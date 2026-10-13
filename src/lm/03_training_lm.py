# %% [markdown]
# # LM 3 — Training Dynamics of a Language Model
#
# *Companion to Lecture 3 — Optimisation and Initialisation.*
#
# LM 2 trained the MLP language model with a sensible initialisation and plain SGD, and it
# worked. Here we use the same model as a **laboratory** to see *why* it worked and what
# breaks when we are careless. We look at the loss at initialisation, saturated tanh
# units, Xavier/Kaiming scaling, SGD vs momentum vs Adam, the learning-rate range test,
# warm-up + cosine schedules, the batch size, and the **update-to-data ratio** — the
# diagnostics that practitioners watch when a billion-parameter training run costs
# millions and cannot be restarted on a whim.
#
# > **Language-model track:** 1 Bigram · 2 MLP LM · **3 Training dynamics** · 4 PyTorch · 5 Word-level & regularisation · 6 WaveNet · 7 Char-RNN · 8 GPT · 9 Sequence VAE · 10 Diffusion LM

# %%
import random
import time
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (5, 4)
plt.rcParams["axes.grid"] = True

# %%
from pathlib import Path
import urllib.request

def load_text(name):
    """Load a corpus from ../data (repository checkout) or download it (e.g. on Colab)."""
    for folder in (Path("data"), Path("../data")):
        if (folder / name).exists():
            return (folder / name).read_text(encoding="utf-8")
    url = f"https://raw.githubusercontent.com/shuishen112/neural-networks/main/data/{name}"
    return urllib.request.urlopen(url).read().decode("utf-8")

# %% [markdown]
# ## 0. Data and model (compact version of LM 2)
#
# Same split, vocabulary and context length $k=3$ as in LM 2.

# %%
words = load_text("names.txt").split()
random.Random(42).shuffle(words)
n1, n2 = int(0.8 * len(words)), int(0.9 * len(words))
train_words, val_words, test_words = words[:n1], words[n1:n2], words[n2:]

chars = "." + "abcdefghijklmnopqrstuvwxyz"   # '.' = start/end token, index 0
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
V, block_size = len(chars), 3

def build_dataset(word_list, k=block_size):
    X, Y = [], []
    for w in word_list:
        context = [0] * k
        for ch in w + ".":
            X.append(context); Y.append(stoi[ch])
            context = context[1:] + [stoi[ch]]
    return np.array(X), np.array(Y)

X_tr, Y_tr = build_dataset(train_words)
X_va, Y_va = build_dataset(val_words)
print("train", X_tr.shape, "val", X_va.shape)

# %% [markdown]
# The model is LM 2's $E = [C_{x_{t-3}};C_{x_{t-2}};C_{x_{t-1}}]$, $H = \tanh(EW_1 + \mathbf b_1)$,
# $Z = HW_2 + \mathbf b_2$, with $d = 10$ and $h = 200$. This time the initialisation is
# configurable: every weight matrix is drawn from $\mathcal N(0, \sigma^2)$ with its own $\sigma$.

# %%
def init_params(w1_std, w2_std, b_std=0.0, d=10, h=200, k=block_size, seed=0):
    r = np.random.default_rng(seed)
    return {"C": r.normal(0, 1, (V, d)),
            "W1": r.normal(0, w1_std, (k * d, h)), "b1": r.normal(0, b_std, h),
            "W2": r.normal(0, w2_std, (h, V)), "b2": r.normal(0, b_std, V)}


def forward_backward(p, X, Y):
    B, k = X.shape
    d = p["C"].shape[1]
    E = p["C"][X].reshape(B, k * d)
    H = np.tanh(E @ p["W1"] + p["b1"])
    Z = H @ p["W2"] + p["b2"]
    Z = Z - Z.max(axis=1, keepdims=True)
    logP = Z - np.log(np.exp(Z).sum(axis=1, keepdims=True))
    loss = -logP[np.arange(B), Y].mean()
    dZ = np.exp(logP); dZ[np.arange(B), Y] -= 1; dZ /= B
    g = {"W2": H.T @ dZ, "b2": dZ.sum(axis=0)}
    dA = (dZ @ p["W2"].T) * (1 - H ** 2)
    g["W1"], g["b1"] = E.T @ dA, dA.sum(axis=0)
    g["C"] = np.zeros_like(p["C"])
    np.add.at(g["C"], X.ravel(), (dA @ p["W1"].T).reshape(B * k, d))
    return loss, g, H


def nll(p, X, Y):
    H = np.tanh(p["C"][X].reshape(len(X), -1) @ p["W1"] + p["b1"])
    Z = H @ p["W2"] + p["b2"]
    Z = Z - Z.max(axis=1, keepdims=True)
    return -(Z[np.arange(len(Y)), Y] - np.log(np.exp(Z).sum(axis=1))).mean()

# %% [markdown]
# One training function serves the whole notebook. The optimisers are the update rules
# of Lecture 3, written for a dictionary of parameters, and the function also records the
# **update-to-data ratio** $\lVert\Delta\theta\rVert/\lVert\theta\rVert$ of every parameter
# (Section 6).

# %%
def make_optimizer(kind, mu=0.9, b1=0.9, b2=0.999, eps=1e-8):
    """Return update(name, grad, lr, t) -> parameter change Delta theta (Lecture 3, Sections 4-5)."""
    state = {}
    def update(name, g, lr, t):
        s = state.setdefault(name, {"m": np.zeros_like(g), "v": np.zeros_like(g)})
        if kind == "sgd":
            return -lr * g
        if kind == "momentum":
            s["m"] = mu * s["m"] + g
            return -lr * s["m"]
        if kind == "adam":
            s["m"] = b1 * s["m"] + (1 - b1) * g
            s["v"] = b2 * s["v"] + (1 - b2) * g ** 2
            m_hat, v_hat = s["m"] / (1 - b1 ** t), s["v"] / (1 - b2 ** t)
            return -lr * m_hat / (np.sqrt(v_hat) + eps)
        raise ValueError(kind)
    return update


def train(p, steps, opt="sgd", lr=0.1, schedule=None, batch_size=64, eval_every=500, seed=0):
    """Mini-batch training. `schedule(t)` overrides the constant learning rate `lr`."""
    r = np.random.default_rng(seed)
    update = make_optimizer(opt)
    log = {"loss": [], "val": [], "ud": {name: [] for name in p}, "lr": []}
    for t in range(steps):
        j = r.integers(0, len(X_tr), batch_size)
        loss, g, _ = forward_backward(p, X_tr[j], Y_tr[j])
        lr_t = schedule(t) if schedule else lr
        for name in p:
            delta = update(name, g[name], lr_t, t + 1)
            if t % 10 == 0:
                log["ud"][name].append(np.linalg.norm(delta) / (np.linalg.norm(p[name]) + 1e-12))
            p[name] += delta
        log["loss"].append(loss); log["lr"].append(lr_t)
        if (t + 1) % eval_every == 0 or t == 0:
            log["val"].append((t + 1, nll(p, X_va, Y_va)))
    log["final_val"] = nll(p, X_va, Y_va)
    return log


def smooth(x, k=100):
    return np.convolve(x, np.ones(k) / k, mode="valid")

results = {}   # name -> final validation NLL, for the table at the end
STEPS = 5000   # budget per configuration in Sections 1-3

# %% [markdown]
# ## 1. What should the loss be at initialisation?
#
# Before training, the network knows nothing, so the most sensible prediction is the
# **uniform distribution** over the 27 characters, with loss
#
# $$ \mathcal L_\text{init} = -\log\tfrac1{27} = \ln 27 \approx 3.30 . $$
#
# The softmax produces (nearly) uniform probabilities when all logits are (nearly)
# equal, e.g. all close to zero. A *naive* initialisation that draws every parameter from
# $\mathcal N(0,1)$ does something very different: each logit $Z_j = \sum_{u=1}^{200}H_uW_{2,uj} + b_{2,j}$
# is a sum of 200 terms of size $\approx 1$, so its standard deviation is about $\sqrt{200}\approx14$.
# The softmax of such logits is essentially one-hot on a random character — the network is
# **confidently wrong**, and the loss is huge.

# %%
p_naive = init_params(w1_std=1.0, w2_std=1.0, b_std=1.0)
_, _, H0 = forward_backward(p_naive, X_tr[:1000], Y_tr[:1000])
Z0 = H0 @ p_naive["W2"] + p_naive["b2"]
print(f"naive init: logit std {Z0.std():.1f}, initial loss {nll(p_naive, X_va, Y_va):.2f}  "
      f"(ln 27 = {np.log(V):.2f})")

# %% [markdown]
# The fix is simple: scale the output layer down ($W_2\sim\mathcal N(0, 0.01^2)$) and set
# $\mathbf b_2 = 0$. Then all logits start near zero and the initial loss is $\approx \ln 27$.
# (Why not $W_2 = 0$ exactly? It would work here, because the gradient of $W_2$ is
# $H^\top\bar Z\ne0$, but the hidden layer would receive zero gradient in the first step;
# a small random $W_2$ is the safe habit.) Let us train both for the same number of steps.

# %%
p_fixed = {name: v.copy() for name, v in p_naive.items()}   # identical, except for the output layer:
p_fixed["W2"] *= 0.01
p_fixed["b2"][:] = 0.0
print(f"scaled output layer: initial loss {nll(p_fixed, X_va, Y_va):.3f}")

log_naive = train(p_naive, STEPS)
log_fixed = train(p_fixed, STEPS)
results["naive N(0,1) init"] = log_naive["final_val"]
results["+ small output layer"] = log_fixed["final_val"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for log, name in [(log_naive, "naive N(0,1)"), (log_fixed, "output layer scaled by 0.01")]:
    axes[0].plot(log["loss"], lw=0.6, alpha=0.8, label=name)
    axes[1].plot(np.arange(99, STEPS), smooth(log["loss"]), label=f"{name} (val {log['final_val']:.3f})")
for ax in axes:
    ax.axhline(np.log(V), color="k", ls=":", lw=1, label="ln 27")
    ax.set_xlabel("step"); ax.set_ylabel("training loss")
axes[0].set_title("Raw mini-batch loss: the hockey stick"); axes[0].legend(fontsize=8)
axes[1].set_ylim(2.0, 3.5); axes[1].set_title("Smoothed, zoomed in"); axes[1].legend(fontsize=8)
plt.tight_layout(); plt.show()

# %% [markdown]
# The naive run shows the famous **hockey-stick** curve: the first few hundred steps are
# spent doing nothing useful except shrinking the logits from "confidently wrong" to
# "uncertain" — and the oversized gradients of that phase also knock the rest of the
# network around. With the scaled output layer the loss starts at $\ln 27$ and every
# step is spent learning structure, and we end lower for the same budget.
#
# **Rule of thumb:** always compute the loss at initialisation and compare it with the
# value you expect ($\ln V$ for a $V$-way softmax). It is the cheapest bug detector there is.
#
# ## 2. Saturated tanh units
#
# The loss is now right, but the hidden layer is still in trouble. With $C\sim\mathcal N(0,1)$
# and $W_1\sim\mathcal N(0,1)$, each pre-activation $A_u = \sum_{i=1}^{30}E_iW_{1,iu} + b_{1,u}$ has
# variance $\approx 30 + 1$, i.e. standard deviation $\approx 5.6$. Most of them land on the flat
# tails of tanh, where $h = \pm1$ and $\tanh'(a) = 1-h^2 \approx 0$. From Lecture 2, the
# backward pass multiplies by exactly this factor,
#
# $$ \bar A = \bar H \odot (1 - H^2), $$
#
# so a **saturated unit passes (almost) no gradient** to $W_1$, $\mathbf b_1$ and the embeddings $C$ —
# the vanishing-gradient problem of Lecture 3, Section 7, in a single layer. A unit that is
# saturated for *every* input is effectively dead: it never learns.
#
# The variance argument of Lecture 3 says: choose $\operatorname{Var}(W_1) = \text{gain}^2/n_\text{in}$ with
# fan-in $n_\text{in} = kd = 30$. Because tanh *contracts* its input (|tanh a| < |a|), the standard
# recommendation (used by `torch.nn.init`) adds a **gain of 5/3** for tanh so that the
# activations neither shrink layer after layer nor saturate:
#
# $$ W_1\sim\mathcal N\!\Big(0,\ \big(\tfrac{5/3}{\sqrt{n_\text{in}}}\big)^2\Big),\qquad \tfrac{5/3}{\sqrt{30}} \approx 0.30 . $$
#
# Let us look at the hidden activations for several scales of $W_1$ (with $\mathbf b_1 = 0$,
# except for the naive case).

# %%
kaiming = (5 / 3) / np.sqrt(block_size * 10)
scales = [("naive: N(0,1), b1 ~ N(0,1)", 1.0, 1.0), ("σ = 0.5", 0.5, 0.0),
          (f"Kaiming tanh: σ = {kaiming:.2f}", kaiming, 0.0), ("σ = 0.03", 0.03, 0.0)]
Xb, Yb = X_tr[:2000], Y_tr[:2000]

fig, axes = plt.subplots(2, 4, figsize=(16, 6.5))
for col, (name, s, bs) in enumerate(scales):
    p = init_params(w1_std=s, w2_std=0.01)
    p["b1"] = np.random.default_rng(0).normal(0, bs, 200)
    _, g, H = forward_backward(p, Xb, Yb)
    sat = np.mean(np.abs(H) > 0.97)
    pass_through = np.mean(1 - H ** 2)          # average local derivative tanh'(a)
    axes[0, col].hist(H.ravel(), bins=50, range=(-1, 1), color="C0")
    axes[0, col].set_title(f"{name}\n|h| > 0.97: {sat:.0%},  mean tanh'(a) = {pass_through:.2f}", fontsize=9)
    axes[0, col].set_yticks([])
    axes[1, col].imshow(np.abs(H[:40]) > 0.97, cmap="gray_r", aspect="auto", interpolation="nearest")
    axes[1, col].set_xlabel("hidden unit"); axes[1, col].grid(False)
    axes[1, col].set_title(f"‖∇W1‖ = {np.linalg.norm(g['W1']):.3f}, ‖∇C‖ = {np.linalg.norm(g['C']):.3f}", fontsize=9)
axes[1, 0].set_ylabel("example (black: |h| > 0.97)")
plt.tight_layout(); plt.show()

# %% [markdown]
# * **Naive**: the histogram piles up at $\pm1$; two-thirds of the activations are saturated
#   (black in the image), the average factor $\tanh'(a) = 1-h^2$ by which the backward signal
#   is multiplied is small, and the gradient reaching $W_1$ is less than half of that with
#   Kaiming scaling.
# * **Kaiming with gain 5/3**: a broad distribution — the units use the whole nonlinear
#   range of tanh, with a minority of saturated values, and $W_1$ receives two and a half
#   times the gradient it gets under the naive initialisation.
# * **Too small** ($\sigma = 0.03$): activations are tiny and tanh is essentially linear, so
#   the hidden layer starts out as a wasted linear map. $W_1$ itself gets a large gradient
#   (nothing is saturated), but the signal that continues down to the embeddings is
#   multiplied by the tiny $W_1^\top$ and almost vanishes: $\lVert\nabla C\rVert$ is five times
#   smaller. This is the backward half of Lecture 3's variance argument.
#
# What does it cost in training? Same optimiser, same budget:

# %%
init_logs = {}
for name, s, bs in scales:
    p = init_params(w1_std=s, w2_std=0.01)
    p["b1"] = np.random.default_rng(0).normal(0, bs, 200)
    init_logs[name] = train(p, STEPS)
    print(f"{name:28s} val NLL after {STEPS} steps: {init_logs[name]['final_val']:.4f}")
results["+ Kaiming tanh init for W1 (SGD, lr 0.1)"] = init_logs[scales[2][0]]["final_val"]

plt.figure(figsize=(7, 4))
for name, log in init_logs.items():
    plt.plot(np.arange(99, STEPS), smooth(log["loss"]), label=f"{name} (val {log['final_val']:.3f})")
plt.ylim(2.0, 3.0); plt.xlabel("step"); plt.ylabel("training loss (smoothed)")
plt.legend(fontsize=8); plt.title("Initialisation scale of $W_1$, SGD lr = 0.1"); plt.show()

# %% [markdown]
# The Kaiming-scaled network learns fastest, the too-small initialisation slowest. The
# differences are modest because a network with a single hidden layer recovers from a
# poor initialisation fairly quickly, but the damage compounds with depth (Lecture 3,
# Section 7.2): with 10 or 100 layers each with saturated or shrinking activations, the
# signal never reaches the bottom. That is why modern networks combine careful
# initialisation with **normalisation layers** (LayerNorm in transformers, Lecture 8)
# and residual connections.
#
# From now on we use the Kaiming-tanh initialisation with a small output layer.

# %%
def good_init(seed=0):
    return init_params(w1_std=kaiming, w2_std=0.01, seed=seed)

# %% [markdown]
# ## 3. Optimisers: SGD, momentum, Adam
#
# Lecture 3 derived three update rules (here with $\mathbf g_t$ the mini-batch gradient):
#
# | | update |
# |---|---|
# | SGD | $\theta \leftarrow \theta - \eta\,\mathbf g_t$ |
# | momentum | $\mathbf v \leftarrow \mu\mathbf v + \mathbf g_t$, $\ \theta \leftarrow \theta - \eta\,\mathbf v$ |
# | Adam | $\mathbf m \leftarrow \beta_1\mathbf m + (1-\beta_1)\mathbf g_t$, $\ \mathbf v \leftarrow \beta_2\mathbf v + (1-\beta_2)\mathbf g_t^2$, $\ \theta \leftarrow \theta - \eta\,\hat{\mathbf m}/(\sqrt{\hat{\mathbf v}}+\epsilon)$ |
#
# A language model is a good place to see Adam's per-parameter step sizes at work: the
# embedding rows of rare characters (`q`, `x`) appear in few mini-batches and receive
# small, infrequent gradients; Adam normalises each parameter's step by the RMS of *its
# own* gradients, so rare rows still move. We give each method the same budget. For
# momentum we use $\eta = 0.01$, $\mu = 0.9$, so that its effective step $\eta/(1-\mu) = 0.1$
# (Lecture 3, Section 4) equals the SGD learning rate. Adam gets its usual $\eta = 3\cdot10^{-3}$.

# %%
opt_configs = [("SGD, lr 0.1", "sgd", 0.1), ("momentum 0.9, lr 0.01", "momentum", 0.01),
               ("Adam, lr 3e-3", "adam", 3e-3)]
opt_logs = {"SGD, lr 0.1": init_logs[scales[2][0]]}       # already trained in Section 2
for name, kind, lr in opt_configs[1:]:
    opt_logs[name] = train(good_init(), STEPS, opt=kind, lr=lr)
    results[name] = opt_logs[name]["final_val"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for name, log in opt_logs.items():
    axes[0].plot(np.arange(99, STEPS), smooth(log["loss"]), label=name)
    v = np.array(log["val"])
    axes[1].plot(v[:, 0], v[:, 1], "o-", ms=3, label=f"{name}: {log['final_val']:.3f}")
axes[0].set_ylim(2.0, 2.8); axes[0].set_title("training loss (smoothed)")
axes[1].set_ylim(2.1, 2.6); axes[1].set_title("validation NLL")
for ax in axes:
    ax.set_xlabel("step"); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()

# %% [markdown]
# Momentum and SGD with the same effective step size give practically the same curve.
# That is not a bug: with noisy mini-batch gradients and a small learning rate, momentum
# mainly averages the noise over the last $1/(1-\mu)$ steps, which is roughly what SGD with
# a $1/(1-\mu)$ times larger learning rate does anyway (the acceleration of Lecture 3 shows
# on ill-conditioned, low-noise problems such as Rosenbrock). Raising momentum's step
# further did not help here (try it). **Adam** clearly beats both on the same budget: its
# per-parameter normalisation gives every tensor — embeddings, biases, weight matrices — a
# well-scaled step (Section 6 makes this visible). It is also the least sensitive to the
# exact learning rate, which is why Adam (in its AdamW form) trains essentially every LLM.
# But none of these numbers mean much without asking: *was the learning rate right?*
#
# ## 4. Finding the learning rate, and scheduling it
#
# ### 4.1 The learning-rate range test
#
# Smith (2017) proposed a cheap way to find a good learning rate: train for a few hundred
# steps while increasing $\eta$ **exponentially** from tiny to huge, and plot the loss
# against $\eta$. Too small: the loss barely moves. Too large: the loss explodes. A good
# learning rate is in the region of steepest descent, a bit below the point where the
# curve turns up.

# %%
def lr_range_test(opt, lr_min, lr_max, steps=1500):
    lrs = lr_min * (lr_max / lr_min) ** (np.arange(steps) / (steps - 1))
    with np.errstate(all="ignore"):
        log = train(good_init(), steps, opt=opt, schedule=lambda t: lrs[t], eval_every=10 ** 9)
    return lrs, np.array(log["loss"])

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (opt, lo, hi, chosen) in zip(axes, [("sgd", 1e-3, 30, 0.1), ("adam", 1e-5, 1.0, 3e-3)]):
    lrs, losses = lr_range_test(opt, lo, hi)
    s = smooth(losses, 25)
    ax.semilogx(lrs[24:], s)
    ax.axvline(chosen, color="k", ls=":", label=f"lr used above: {chosen:g}")
    ax.set_ylim(1.9, 4.5); ax.set_xlabel("learning rate (increasing during the run)")
    ax.set_ylabel("training loss (smoothed)"); ax.set_title(f"LR range test, {opt.upper()}"); ax.legend()
    print(f"{opt}: lowest smoothed loss at lr ≈ {lrs[24 + np.nanargmin(s)]:.3g}")
plt.tight_layout(); plt.show()

# %% [markdown]
# Both curves have the expected shape: flat on the left, a steep descent, a minimum, and
# then divergence. A common heuristic is to pick a learning rate somewhat below the
# minimum of the curve (the minimum itself is reached in a regime that is already on the
# edge of instability, and the test is biased because the loss at step $t$ also reflects all
# earlier, smaller learning rates). The test costs a fraction of one training run — a
# sensible first thing to do with any new model.
#
# ### 4.2 Warm-up + cosine decay
#
# As in Lecture 3, Section 6, the learning rate should be **large early** (fast progress)
# and **small late** (settle into the minimum, shrink the noise ball). The recipe used for
# almost every transformer LM is a short **linear warm-up** followed by **cosine decay**:
#
# $$ \eta_t = \begin{cases} \eta_\text{max}\, \dfrac{t+1}{T_\text{w}}, & t < T_\text{w}, \\[2mm]
# \eta_\text{min} + \tfrac12(\eta_\text{max} - \eta_\text{min})\Big(1 + \cos\big(\pi\,\tfrac{t - T_\text{w}}{T - T_\text{w}}\big)\Big), & t\ge T_\text{w}. \end{cases} $$
#
# Warm-up protects the first steps, when Adam's second-moment estimates rest on very few
# gradients and the network is far from any reasonable region. Because a schedule lets
# us start with a larger step, we raise the peak learning rate of Adam to $10^{-2}$.

# %%
def warmup_cosine(T, lr_max, lr_min=None, warmup=None):
    lr_min = lr_max / 20 if lr_min is None else lr_min
    warmup = warmup if warmup is not None else T // 20
    def schedule(t):
        if t < warmup:
            return lr_max * (t + 1) / warmup
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * (t - warmup) / (T - warmup)))
    return schedule

sched_logs = {"Adam, constant 3e-3": opt_logs["Adam, lr 3e-3"],
              "Adam, constant 1e-2": train(good_init(), STEPS, opt="adam", lr=1e-2),
              "Adam, warm-up + cosine, peak 1e-2": train(good_init(), STEPS, opt="adam",
                                                         schedule=warmup_cosine(STEPS, 1e-2))}
results["Adam, constant 1e-2"] = sched_logs["Adam, constant 1e-2"]["final_val"]
results["Adam, warm-up + cosine 1e-2"] = sched_logs["Adam, warm-up + cosine, peak 1e-2"]["final_val"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for name, log in sched_logs.items():
    axes[0].plot(log["lr"], label=name)
    axes[1].plot(np.arange(99, STEPS), smooth(log["loss"]), label=f"{name}: val {log['final_val']:.3f}")
axes[0].set_xlabel("step"); axes[0].set_ylabel("learning rate"); axes[0].legend(fontsize=8)
axes[1].set_ylim(2.0, 2.5); axes[1].set_xlabel("step"); axes[1].set_ylabel("training loss (smoothed)")
axes[1].legend(fontsize=8)
plt.tight_layout(); plt.show()

# %% [markdown]
# With a constant large learning rate the loss plateaus at a noisy level; the cosine
# schedule uses the large rate when it helps and anneals it at the end, producing the
# characteristic late drop in loss and the best validation NLL of the three.
#
# ## 5. Batch size
#
# A larger mini-batch gives a less noisy gradient (variance $\propto 1/B$, Lecture 3) but costs
# more per step. We train with Adam and a warm-up + cosine schedule for the same
# **number of steps** with $B = 16, 64, 256$ and plot validation NLL against steps and
# against the number of training examples (here: characters) processed.

# %%
BS_STEPS = 3000
bs_logs = {}
for B in (16, 64, 256):
    t0 = time.time()
    bs_logs[B] = train(good_init(), BS_STEPS, opt="adam", schedule=warmup_cosine(BS_STEPS, 1e-2),
                       batch_size=B, eval_every=250)
    bs_logs[B]["time"] = time.time() - t0
    results[f"Adam + cosine, batch {B}, {BS_STEPS} steps"] = bs_logs[B]["final_val"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for B, log in bs_logs.items():
    v = np.array(log["val"])
    label = f"B = {B}: val {log['final_val']:.3f}, {log['time']:.1f} s"
    axes[0].plot(v[:, 0], v[:, 1], "o-", ms=3, label=label)
    axes[1].semilogx(v[:, 0] * B, v[:, 1], "o-", ms=3, label=label)
axes[0].set_xlabel("step"); axes[1].set_xlabel("training examples seen")
for ax in axes:
    ax.set_ylabel("validation NLL"); ax.set_ylim(2.1, 2.7); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()

# %% [markdown]
# Per **step**, larger batches win: each update is better informed. Per **example**
# processed, small batches are more efficient: $B=16$ reaches a given loss after fewer
# examples, because it makes many more (noisier) updates per epoch. Which axis matters
# depends on the hardware. On a GPU a batch of 256 costs barely more wall-clock time than
# a batch of 16 (the matrix multiplications run in parallel), so we take the larger batch;
# on our CPU the cost grows with $B$. LLM pre-training uses batches of millions of tokens
# and increases the learning rate with the batch size — up to a "critical batch size"
# beyond which larger batches stop paying off (McCandlish et al., 2018).
#
# ## 6. The update-to-data ratio
#
# How do you know, *during* a long run, whether the learning rate is sensible for every
# part of the network? A practical diagnostic is the ratio of the size of each update to
# the size of the parameter it updates,
#
# $$ \rho_t = \frac{\lVert\Delta\theta_t\rVert}{\lVert\theta_t\rVert}, \qquad \Delta\theta_t = -\eta\,\mathbf g_t \ \text{(SGD)}, $$
#
# plotted on a $\log_{10}$ scale, one curve per parameter tensor. If $\log_{10}\rho\approx-3$, each step
# changes the parameter by about 0.1 % — a commonly quoted healthy range is roughly $-3.5$ to
# $-2.5$. Much lower (e.g. $-5$) means the tensor is effectively frozen (learning rate too
# small for it); much higher (e.g. $-1$) means each step overwrites a sizeable fraction of
# the weights and training is likely unstable. Different tensors with very different
# ratios point to a scaling problem — which is precisely what Adam's per-parameter
# normalisation and per-layer learning rates address. Our `train` function has recorded
# $\rho$ every 10 steps for all runs.

# %%
ud_runs = {"SGD, lr 0.1, naive init": init_logs[scales[0][0]],
           "SGD, lr 0.1, Kaiming init": init_logs[scales[2][0]],
           "SGD, lr 0.01, Kaiming init": train(good_init(), 2000, lr=0.01),
           "Adam, warm-up + cosine 1e-2": sched_logs["Adam, warm-up + cosine, peak 1e-2"]}
fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
for ax, (name, log) in zip(axes, ud_runs.items()):
    for pname, ratios in log["ud"].items():
        ax.plot(np.arange(len(ratios)) * 10, np.log10(np.array(ratios) + 1e-12), lw=0.8, label=pname)
    ax.axhspan(-3.5, -2.5, color="C2", alpha=0.12)
    ax.set_title(name, fontsize=10); ax.set_xlabel("step")
axes[0].set_ylabel(r"$\log_{10}\ \|\Delta\theta\| / \|\theta\|$"); axes[0].legend(fontsize=8)
axes[0].set_ylim(-6, -0.5)
plt.tight_layout(); plt.show()

for name, log in ud_runs.items():
    last = {k: np.log10(np.mean(v[-50:])) for k, v in log["ud"].items()}
    print(f"{name:30s}", "  ".join(f"{k}: {v:5.2f}" for k, v in last.items()))

# %% [markdown]
# Reading the plots (the green band is $-3.5$ to $-2.5$; the final values are printed above):
#
# * With **SGD at lr 0.01**, the embeddings $C$ and the hidden weights $W_1$ sit around
#   $10^{-4}$: they are practically frozen, while the output layer $W_2$ moves a hundred
#   times faster in relative terms. The layers learn at very different speeds.
# * With **SGD at lr 0.1** and Kaiming init, $C$ and $W_1$ are inside the band. The output
#   layer and the biases are above it: $W_2$ started tiny (std 0.01), so even moderate updates
#   are large relative to its norm. That is tolerable for a last layer, but in a deep network
#   it is exactly the kind of imbalance to watch.
# * With the **naive initialisation**, $W_1$ and $\mathbf b_1$ update almost an order of
#   magnitude more slowly than with Kaiming: their gradients pass through saturated
#   tanh units (Section 2).
# * **Adam** keeps all tensors within about one order of magnitude of each other, and the
#   ratios follow the schedule: up during warm-up, then down with the cosine — the
#   biases and embeddings move, relatively, about as much as the big weight matrices.
#
# In LLM training these per-layer statistics (together with gradient norms and activation
# statistics) are logged continuously; a layer whose ratio drifts out of range is often
# the first sign of an impending loss spike.
#
# ## 7. Putting it together
#
# Our best recipe: Kaiming-tanh initialisation with a small output layer, Adam, warm-up +
# cosine with peak $10^{-2}$, and — following Section 5 — batch 256. We train it for 6 000
# steps (1.5 M examples, about 8 epochs) and compare with everything above and with LM 2
# (SGD, batch 64, 25 000 steps, val NLL ≈ 2.135).

# %%
FINAL_STEPS = 6000
t0 = time.time()
p_best = good_init()
log_best = train(p_best, FINAL_STEPS, opt="adam", schedule=warmup_cosine(FINAL_STEPS, 1e-2),
                 batch_size=256, eval_every=1000)
print(f"{FINAL_STEPS} steps in {time.time() - t0:.1f} s; train NLL {nll(p_best, X_tr, Y_tr):.4f}, "
      f"val NLL {log_best['final_val']:.4f}")
results[f"best recipe: batch 256, {FINAL_STEPS} steps"] = log_best["final_val"]

print(f"\n{'configuration (5000 steps, batch 64, unless stated)':52s} val NLL")
print("-" * 60)
for name, v in results.items():
    print(f"{name:52s} {v:.4f}")

# %% [markdown]
# And, as always, samples from the final model.

# %%
def sample(p, n, seed=0):
    r = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        context, name = [0] * block_size, ""
        while True:
            h = np.tanh(p["C"][context].reshape(-1) @ p["W1"] + p["b1"])
            z = h @ p["W2"] + p["b2"]
            prob = np.exp(z - z.max()); prob /= prob.sum()
            ix = r.choice(V, p=prob)
            if ix == 0:
                break
            name += itos[ix]
            context = context[1:] + [ix]
        out.append(name)
    return out

print(", ".join(sample(p_best, 20)))
print(f"val NLL (nats/char): {log_best['final_val']:.3f}")

# %% [markdown]
# ## Summary
#
# | Diagnostic / technique | What to look for |
# |---|---|
# | Loss at init | $\approx \ln V$; much larger ⇒ output layer too large, hockey-stick curve |
# | Hidden activations | few units with $\lvert h\rvert > 0.97$; no unit saturated on all inputs |
# | Init scale | $\sigma = \text{gain}/\sqrt{n_\text{in}}$, gain $5/3$ for tanh, small last layer |
# | Optimiser | Adam(W) is the robust default for language models |
# | Learning rate | range test, then warm-up + cosine decay |
# | Batch size | better per step, worse per example; choose by hardware |
# | Update-to-data ratio | $\log_{10}(\lVert\Delta\theta\rVert/\lVert\theta\rVert)\approx -3$ for every tensor |
#
# ## From here to LLMs
#
# * **Initialisation of transformers** follows the same logic: GPT-2 initialises weights with
#   std 0.02 and scales the output projections of residual branches by $1/\sqrt{2L}$ for $L$
#   layers, so the residual stream does not grow with depth. Checking that the initial loss
#   is $\approx\ln|\mathcal V|$ ($\ln 50\,257 \approx 10.8$ for GPT-2) is a standard first test.
# * **AdamW + warm-up + cosine** is the default recipe for LLM pre-training (GPT-3, Llama),
#   typically with $\beta_2 = 0.95$, gradient clipping at norm 1, a warm-up of a few thousand
#   steps and decay to 10 % of the peak learning rate. Newer schedules ("warmup-stable-decay")
#   keep the rate constant and decay only at the end, which makes it easy to continue training.
# * **Learning-rate sweeps on small models** — our range test — are how large runs are
#   tuned: hyperparameters are found on small proxies and transferred, e.g. with
#   µP (maximal update parametrisation), which rescales initialisation and per-layer
#   learning rates so that the optimal learning rate stays the same as the model grows.
#   µP is, in essence, a principled way to keep the update-to-data ratio of every layer constant.
# * **Batch size** in LLM training is measured in tokens (GPT-3: 3.2 M tokens per batch) and
#   is often ramped up during training; the critical batch size grows as the loss falls.
# * **Monitoring.** Large runs log the loss, gradient norm, update-to-data ratios and
#   activation statistics per layer. Loss spikes are typically preceded by exploding
#   activations or ratios in a few layers — the same health checks as in this notebook,
#   at scale.
#
# ## Try it yourself
#
# 1. Use the LR range test to pick learning rates for momentum SGD. Do your picks beat
#    the values used in Section 3?
# 2. Add **gradient clipping** (rescale the full gradient if its norm exceeds 1) and retrain
#    the naive initialisation with a large learning rate. Does it help?
# 3. Make the network deeper (three tanh layers of 100 units). How do the activation
#    histograms of the deeper layers look with the naive init and with Kaiming?
#    (This is where initialisation really starts to matter.)
# 4. Implement **AdamW** (decoupled weight decay, Lecture 5) and compare the validation NLL with
#    Adam + L2 penalty at the same strength.
# 5. Scale the batch size from 64 to 256 and the learning rate by 2 or 4. Which rule —
#    linear scaling or square-root scaling — works better for Adam here?
# 6. Plot the update-to-data ratio of the embedding *rows* for frequent (`a`, `e`) and rare
#    (`q`, `x`) characters separately, with SGD and with Adam.
#
# **Further reading.**
# Glorot & Bengio (2010), "Understanding the difficulty of training deep feedforward neural networks";
# He et al. (2015), "Delving deep into rectifiers";
# Kingma & Ba (2015), "Adam: a method for stochastic optimization";
# Smith (2017), "Cyclical learning rates for training neural networks" (LR range test);
# Loshchilov & Hutter (2017, 2019), "SGDR" and "Decoupled weight decay regularization";
# McCandlish et al. (2018), "An empirical model of large-batch training";
# Yang et al. (2022), "Tensor Programs V: tuning large neural networks via zero-shot hyperparameter transfer" (µP);
# Andrej Karpathy, "Building makemore Part 3: activations & gradients, BatchNorm" (video lecture, 2022).
