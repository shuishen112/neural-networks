# %% [markdown]
# # LM 2 — A Neural Probabilistic Language Model
#
# *Companion to Lecture 2 — Multilayer Perceptrons and Backpropagation.*
#
# Counting does not scale: an $n$-gram table grows exponentially with the context length
# and most of its entries are never observed. Bengio et al. (2003) proposed to replace the
# table by a **neural network** that maps the previous characters to **learned embedding
# vectors** and predicts the next character with an MLP. Similar characters get similar
# vectors, so what the model learns about one context transfers to others. We implement
# the model in NumPy with a **hand-written backward pass** — including the gradient of the
# embedding lookup, the one operation Lecture 2 did not cover. The embedding table and the
# "concatenate, transform, softmax" pipeline are exactly the first and last layers of
# every modern LLM.
#
# > **Language-model track:** 1 Bigram · **2 MLP LM** · 3 Training dynamics · 4 PyTorch · 5 Word-level & regularisation · 6 WaveNet · 7 Char-RNN · 8 GPT · 9 Sequence VAE · 10 Diffusion LM

# %%
import random
import time
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
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
# Same data, split and vocabulary as in LM 1.

# %%
words = load_text("names.txt").split()
random.Random(42).shuffle(words)
n1, n2 = int(0.8 * len(words)), int(0.9 * len(words))
train_words, val_words, test_words = words[:n1], words[n1:n2], words[n2:]

chars = "." + "abcdefghijklmnopqrstuvwxyz"   # '.' = start/end token, index 0
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
V = len(chars)
print(f"train {len(train_words)}, val {len(val_words)}, test {len(test_words)} names; V = {V}")

# %% [markdown]
# ## 1. Why not just count longer contexts?
#
# The bigram model of LM 1 reached a validation NLL of about 2.45 nats/char. The obvious
# improvement is to condition on more history: an $n$-gram model uses the last $k = n-1$
# characters as context, $p(x_t\mid x_{t-k},\dots,x_{t-1})$. Its table has $27^k$ rows of 27
# probabilities each:
#
# | context $k$ | rows $27^k$ | parameters $27^{k+1}$ |
# |---|---|---|
# | 1 | 27 | 729 |
# | 2 | 729 | 19 683 |
# | 3 | 19 683 | 531 441 |
# | 5 | 14 348 907 | 387 420 489 |
#
# while our training set has only about 182 000 next-character events. Let us measure how
# many of the possible contexts we actually observe, and how often a validation context
# was **never seen** in training — in which case a count model has nothing to go on
# but its smoothing.

# %%
def ngram_events(word_list, k):
    """All (context, next) pairs with a context of k characters, padded with '.' at the start."""
    out = []
    for w in word_list:
        s = "." * k + w + "."
        out += [(s[i:i + k], s[i + k]) for i in range(len(s) - k)]
    return out

ks = [1, 2, 3, 4, 5]
seen_frac, unseen_val = [], []
for k in ks:
    train_ctx = {c for c, _ in ngram_events(train_words, k)}
    val_ev = ngram_events(val_words, k)
    seen_frac.append(len(train_ctx) / 27 ** k)
    unseen_val.append(np.mean([c not in train_ctx for c, _ in val_ev]))
    print(f"k = {k}: {len(train_ctx):>6} of {27 ** k:>10,} contexts seen in training "
          f"({seen_frac[-1]:7.3%}); validation events with unseen context: {unseen_val[-1]:6.2%}")

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.semilogy(ks, seen_frac, "o-", label="fraction of possible contexts seen in training")
ax.semilogy(ks, unseen_val, "s-", label="validation events with an unseen context")
ax.set_xlabel("context length $k$"); ax.legend(fontsize=8); ax.set_title("The curse of dimensionality for $n$-grams")
plt.show()

# %% [markdown]
# With five characters of context, only a tiny fraction of all possible contexts is ever
# observed, and a sizeable share of validation events have a context the count model has
# literally never seen. Classical $n$-gram models fight this with **back-off** and
# **interpolation** (fall back to shorter contexts when the long one is rare), but the
# fundamental problem remains: a count table treats `"ann"` and `"anm"` as completely
# unrelated contexts. It cannot **generalise** across similar contexts, because it has no
# notion of similarity.
#
# ## 2. The neural probabilistic language model (Bengio et al., 2003)
#
# Bengio's idea has three parts:
#
# 1. **Embed** every token $v$ as a learned vector $C_v\in\mathbb R^{d}$ — a row of an
#    **embedding table** $C\in\mathbb R^{V\times d}$, with $d \ll V$.
# 2. **Concatenate** the embeddings of the $k$ context tokens into one vector
#    $\mathbf e = [C_{x_{t-k}}; \dots; C_{x_{t-1}}]\in\mathbb R^{kd}$.
# 3. Feed $\mathbf e$ into an **MLP** (Lecture 2) whose output layer is a softmax over the vocabulary.
#
# For a mini-batch of $B$ contexts, with integer inputs $X\in\{0..26\}^{B\times k}$:
#
# $$
# \begin{aligned}
# E &= \big[C_{X_{:,1}}; \dots; C_{X_{:,k}}\big] \in\mathbb R^{B\times kd} &&\text{(lookup + concatenate)}\\
# A &= E\,W_1 + \mathbf b_1, \qquad H = \tanh(A) \in\mathbb R^{B\times h} &&\text{(hidden layer)}\\
# Z &= H\,W_2 + \mathbf b_2 \in\mathbb R^{B\times V} &&\text{(logits)}\\
# \mathcal L &= -\frac1B\sum_{n}\log\operatorname{softmax}(Z_{n,:})_{y_n} &&\text{(cross-entropy = NLL)}
# \end{aligned}
# $$
#
# ```
#   x_{t-3}  x_{t-2}  x_{t-1}         integer token ids
#      |        |        |
#    C[.]     C[.]     C[.]           shared embedding table C (V x d)
#      \________|________/
#               |  concat (k*d)
#        tanh(E W1 + b1)              hidden layer (h units)
#               |
#          H W2 + b2  -> softmax      p(x_t | context)  (V outputs)
# ```
#
# The embedding table is **shared** across positions: the same vector represents `a`
# whether it is one, two or three characters back. Generalisation comes for free: if the
# training signal pushes `a` and `e` to similar vectors (because they are followed by
# similar characters), then everything the MLP learns about contexts containing `a`
# applies to contexts containing `e`. The number of parameters,
# $Vd + kd\,h + h + hV + V$, grows only **linearly** in $k$.
#
# *Connection to LM 1.* A lookup $C_{x}$ equals the matrix product $\mathbf x^\top C$ with a
# one-hot $\mathbf x$, so an embedding layer is a linear layer without bias applied to one-hot
# inputs — just like the bigram logit table $W$, but with $d$ instead of $V$ outputs.
# The bigram model is the special case $k=1$, $d=V$, no hidden layer.
#
# ## 3. The dataset
#
# We use a context of $k = 3$ characters, padding the start of each name with `.`. Every
# name of length $L$ gives $L+1$ training examples.

# %%
block_size = 3   # context length k

def build_dataset(word_list, k=block_size):
    X, Y = [], []
    for w in word_list:
        context = [0] * k
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context); Y.append(ix)
            context = context[1:] + [ix]
    return np.array(X), np.array(Y)

X_tr, Y_tr = build_dataset(train_words)
X_va, Y_va = build_dataset(val_words)
print("train", X_tr.shape, "val", X_va.shape)
for x, y in zip(X_tr[:7], Y_tr[:7]):
    print("".join(itos[i] for i in x), "->", itos[y])

# %% [markdown]
# ## 4. Backpropagation through the embedding lookup
#
# Everything after $E$ is the MLP of Lecture 2, so its backward pass is known:
#
# $$
# \bar Z = \tfrac1B(P - Y_\text{1hot}),\quad
# \bar W_2 = H^\top\bar Z,\quad \bar{\mathbf b}_2 = \mathbf 1^\top\bar Z,\quad
# \bar A = (\bar Z W_2^\top)\odot(1 - H^2),\quad
# \bar W_1 = E^\top\bar A,\quad \bar{\mathbf b}_1 = \mathbf 1^\top\bar A,\quad
# \bar E = \bar A\,W_1^\top .
# $$
#
# The new piece is $\bar C$. Reshape $\bar E\in\mathbb R^{B\times kd}$ to $B\times k\times d$: entry
# $\bar E_{n,p,:}$ is the gradient with respect to the embedding used at position $p$ of
# example $n$, which is row $X_{n,p}$ of $C$. A row of $C$ is **used many times** in a batch
# (the letter `a` appears in many contexts, sometimes several times in one), and by the
# "sum over paths" rule of Lecture 2 the gradients of all its uses must be **added**:
#
# $$ \bar C_v = \sum_{n,p\,:\,X_{n,p} = v} \bar E_{n,p,:} . $$
#
# This is a **scatter-add**. In NumPy the tempting `dC[X] += dE` is wrong: with repeated
# indices, buffered fancy-index assignment writes each row only once and the
# contributions overwrite instead of accumulate. `np.add.at` performs the unbuffered
# version. (Equivalently $\bar C = X_\text{1hot}^\top\bar E$, the transpose of the lookup — the
# gradient of the one-hot product in LM 1.)

# %%
idx = np.array([1, 1, 2])            # row 1 is used twice
g = np.ones((3, 2))
wrong = np.zeros((4, 2)); wrong[idx] += g
right = np.zeros((4, 2)); np.add.at(right, idx, g)
print("dC[idx] += g     ->", wrong[:3].tolist())
print("np.add.at(dC,...) ->", right[:3].tolist())

# %% [markdown]
# Now the whole model. Parameters are kept in a dictionary; `forward_backward` returns
# the loss and the gradients (the compact style of Lecture 3), `nll` only runs the forward pass.

# %%
def init_params(d=10, h=200, k=block_size, seed=0):
    """Embedding table + MLP. Scaled Gaussian weights (Lecture 2); LM 3 examines this choice."""
    r = np.random.default_rng(seed)
    return {
        "C": r.normal(0, 1, (V, d)),
        "W1": r.normal(0, 1 / np.sqrt(k * d), (k * d, h)), "b1": np.zeros(h),
        "W2": r.normal(0, 1 / np.sqrt(h), (h, V)), "b2": np.zeros(V),
    }


def forward_backward(p, X, Y):
    B, k = X.shape
    d = p["C"].shape[1]
    # forward
    E = p["C"][X].reshape(B, k * d)                  # lookup + concatenate
    H = np.tanh(E @ p["W1"] + p["b1"])
    Z = H @ p["W2"] + p["b2"]
    Z = Z - Z.max(axis=1, keepdims=True)
    logP = Z - np.log(np.exp(Z).sum(axis=1, keepdims=True))
    loss = -logP[np.arange(B), Y].mean()
    # backward
    dZ = np.exp(logP); dZ[np.arange(B), Y] -= 1; dZ /= B
    g = {"W2": H.T @ dZ, "b2": dZ.sum(axis=0)}
    dA = (dZ @ p["W2"].T) * (1 - H ** 2)
    g["W1"], g["b1"] = E.T @ dA, dA.sum(axis=0)
    dE = dA @ p["W1"].T
    g["C"] = np.zeros_like(p["C"])
    np.add.at(g["C"], X.ravel(), dE.reshape(B * k, d))   # scatter-add into the table
    return loss, g


def nll(p, X, Y, chunk=50_000):
    """Mean NLL (nats/char), forward pass only."""
    total = 0.0
    for s in range(0, len(X), chunk):
        x, y = X[s:s + chunk], Y[s:s + chunk]
        H = np.tanh(p["C"][x].reshape(len(x), -1) @ p["W1"] + p["b1"])
        Z = H @ p["W2"] + p["b2"]
        Z = Z - Z.max(axis=1, keepdims=True)
        total -= (Z[np.arange(len(y)), y] - np.log(np.exp(Z).sum(axis=1))).sum()
    return total / len(X)

# %% [markdown]
# ## 5. Gradient check
#
# As in Lecture 2, we compare every analytical gradient with a central finite difference
# on a tiny model in float64 ($d=2$, $h=8$, 16 examples). The batch is chosen so that some
# characters appear several times — exactly the case the scatter-add has to get right.

# %%
def grad_check(p, X, Y, eps=1e-6):
    _, g = forward_backward(p, X, Y)
    for name, W in p.items():
        num = np.zeros_like(W)
        for i in np.ndindex(W.shape):
            old = W[i]
            W[i] = old + eps; lp, _ = forward_backward(p, X, Y)
            W[i] = old - eps; lm, _ = forward_backward(p, X, Y)
            W[i] = old
            num[i] = (lp - lm) / (2 * eps)
        rel = np.linalg.norm(num - g[name]) / (np.linalg.norm(num + g[name]) + 1e-12)
        print(f"{name:3s} shape {str(W.shape):8s} relative error {rel:.2e}")

p_small = init_params(d=2, h=8, seed=1)
Xg, Yg = X_tr[:16], Y_tr[:16]
counts = np.bincount(Xg.ravel(), minlength=V)
print("characters used more than once in this batch:", {itos[i]: int(c) for i, c in enumerate(counts) if c > 1})
grad_check(p_small, Xg, Yg)

# %% [markdown]
# All relative errors are around $10^{-9}$ or smaller: the backward pass, including the
# scatter-add, is correct. (Try replacing `np.add.at` by `g["C"][X.ravel()] += ...` and
# watch the error for `C` jump.)
#
# ## 6. Training
#
# We use embedding dimension $d = 10$, $h = 200$ hidden units, mini-batches of 64 sampled
# at random, and plain SGD (Lecture 3) with learning rate $0.1$, dropped to $0.01$ for the
# last quarter of training. The model has about 12 000 parameters — compare that with
# the 531 441 entries of a 4-gram table.

# %%
params = init_params(d=10, h=200)
print("parameters:", sum(v.size for v in params.values()))
print(f"loss at initialisation: {nll(params, X_va, Y_va):.3f}  (uniform: ln 27 = {np.log(V):.3f})")

steps, batch_size = 25_000, 64
r = np.random.default_rng(0)
train_loss, val_curve = [], []
t0 = time.time()
for step in range(steps):
    j = r.integers(0, len(X_tr), batch_size)
    loss, grads = forward_backward(params, X_tr[j], Y_tr[j])
    lr = 0.1 if step < 0.75 * steps else 0.01            # learning-rate drop
    for name in params:
        params[name] -= lr * grads[name]
    train_loss.append(loss)
    if step % 1000 == 0 or step == steps - 1:
        val_curve.append((step, nll(params, X_va, Y_va)))
print(f"{steps} steps in {time.time() - t0:.1f} s")

nll_tr, nll_va = nll(params, X_tr, Y_tr), nll(params, X_va, Y_va)
print(f"MLP LM: train NLL {nll_tr:.4f}, val NLL {nll_va:.4f}, val perplexity {np.exp(nll_va):.2f}")

# %% [markdown]
# For comparison we rebuild the add-$\alpha$ bigram model of LM 1 in two lines.

# %%
Nb = np.full((V, V), 0.3)                           # alpha = 0.3, as chosen in LM 1
np.add.at(Nb, (X_tr[:, -1], Y_tr), 1)               # last context character -> next
Pb = Nb / Nb.sum(axis=1, keepdims=True)
nll_bigram = -np.log(Pb[X_va[:, -1], Y_va]).mean()
print(f"bigram val NLL {nll_bigram:.4f}  ->  MLP LM val NLL {nll_va:.4f}")

w = 200
smooth = np.convolve(train_loss, np.ones(w) / w, mode="valid")
vc = np.array(val_curve)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(train_loss, color="C0", alpha=0.15, lw=0.5)
ax.plot(np.arange(w - 1, steps), smooth, color="C0", label="train (mini-batch, smoothed)")
ax.plot(vc[:, 0], vc[:, 1], "o-", color="C1", ms=3, label="validation")
ax.axhline(np.log(V), color="0.5", ls=":", label="uniform (ln 27)")
ax.axhline(nll_bigram, color="k", ls="--", lw=1, label="bigram (LM 1)")
ax.axvline(0.75 * steps, color="C3", lw=0.8, ls=":", label="lr drop")
ax.set_ylim(1.9, 3.6); ax.set_xlabel("step"); ax.set_ylabel("NLL (nats/char)")
ax.legend(fontsize=8); ax.set_title("Training the MLP language model"); plt.show()

# %% [markdown]
# The MLP beats the bigram by about 0.3 nats per character — a drop in perplexity from
# ≈ 11.6 to ≈ 8.5. The learning-rate drop gives a visible step down in loss, as in
# Lecture 3: with a constant learning rate SGD bounces around in a "noise ball", and a
# smaller step lets it settle. Training and validation NLL are close, so with 12 000
# parameters and 180 000 examples we are **not** overfitting — a bigger model would do
# better (see *Try it yourself*).
#
# Two things you may have noticed in the first steps: the initial loss is higher than
# $\ln 27$, and the loss drops very quickly at the start before slowly improving. Both are
# symptoms of the initialisation, and LM 3 is devoted to them.
#
# ## 7. What did the embeddings learn?
#
# Each character is now a point in $\mathbb R^{10}$, which we cannot look at directly. So we
# also train a small companion model with **$d = 2$** (and $h = 100$, 10 000 steps — a few
# seconds) whose embeddings we can plot as they are, and for the main model we project the
# 27 vectors onto their first two principal components (an SVD of the centred table).

# %%
def train(p, steps, batch_size=64, lr=0.1, seed=0):
    r = np.random.default_rng(seed)
    for step in range(steps):
        j = r.integers(0, len(X_tr), batch_size)
        _, grads = forward_backward(p, X_tr[j], Y_tr[j])
        for name in p:
            p[name] -= (lr if step < 0.75 * steps else lr / 10) * grads[name]
    return p

params_2d = train(init_params(d=2, h=100), 10_000)
print(f"d = 2 model: val NLL {nll(params_2d, X_va, Y_va):.4f} (vs {nll_va:.4f} for d = 10)")

Cc = params["C"] - params["C"].mean(axis=0)
U, S, Vt = np.linalg.svd(Cc, full_matrices=False)
C_pca = Cc @ Vt[:2].T
print(f"d = 10 model: the first two principal components explain {(S[:2] ** 2).sum() / (S ** 2).sum():.0%} of the variance")

vowels = set("aeiouy")
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
for ax, Z, title in [(axes[0], params_2d["C"], "d = 2 model: embeddings as learned"),
                     (axes[1], C_pca, "d = 10 model: first two principal components")]:
    for i, ch in enumerate(chars):
        col = "C3" if ch in vowels else ("C2" if ch == "." else "C0")
        ax.scatter(Z[i, 0], Z[i, 1], s=250, color=col, alpha=0.3)
        ax.text(Z[i, 0], Z[i, 1], ch, ha="center", va="center", fontsize=12)
    ax.set_title(title)
fig.suptitle("Character embeddings (red: vowels incl. y, green: '.')"); plt.tight_layout(); plt.show()

def vowel_separation(C):
    """Mean distance between vowels, and between vowels and consonants."""
    D = np.linalg.norm(C[:, None] - C[None], axis=-1)
    vw = [stoi[c] for c in "aeiouy"]; cs = [stoi[c] for c in chars[1:] if c not in vowels]
    return D[np.ix_(vw, vw)].sum() / (len(vw) * (len(vw) - 1)), D[np.ix_(vw, cs)].mean()

for name, C in [("d = 2 ", params_2d["C"]), ("d = 10", params["C"])]:
    vv, vc_ = vowel_separation(C)
    print(f"{name}: mean distance vowel-vowel {vv:.2f}, vowel-consonant {vc_:.2f}")

# %% [markdown]
# Nobody told the network what a vowel is, yet in the 2-D model the vowels form their own
# group, well away from the consonants, and `.` (the boundary token) sits on its own. In
# the 10-D model the structure is spread over more dimensions, so a 2-D projection shows
# it less cleanly; the distances confirm the trend, if less dramatically: vowels are closer
# to each other than to consonants. The reason is distributional: characters that are preceded and followed by
# similar characters receive similar gradients and drift together. This is the
# **distributional hypothesis** ("you shall know a word by the company it keeps", Firth
# 1957) emerging from plain next-token prediction — the same mechanism that makes word
# embeddings such as word2vec, and the token embeddings inside LLMs, meaningful.
#
# ## 8. Sampling
#
# Generation works as in LM 1, except that the model now sees the last three characters.

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

samples = sample(params, 300)
print(", ".join(samples[:24]))
train_set = set(train_words)
print(f"\n{np.mean([s in train_set for s in samples]):.0%} of 300 samples are names from the training set; "
      f"mean length {np.mean([len(s) for s in samples]):.1f} (data: {np.mean([len(w) for w in words]):.1f})")

# %% [markdown]
# The samples are clearly more name-like than the bigram's: plausible syllables,
# realistic endings and lengths. Most of them are *new* names — the model has learned
# the patterns rather than memorised the list.

# %%
print(f"bigram (LM 1) val NLL {nll_bigram:.3f} | MLP LM val NLL {nll_va:.3f}")
print(f"val NLL (nats/char): {nll_va:.3f}")

# %% [markdown]
# ## From here to LLMs
#
# * **Embedding tables.** Every LLM starts with exactly our `C[X]` lookup: a table of
#   shape (vocabulary × model width), e.g. $128\,000\times 4\,096$ in Llama 3 8B. Its backward pass
#   is the same scatter-add (`index_add_` / `embedding_backward` in PyTorch), and for large
#   vocabularies the embedding table is one of the largest single parameter matrices in the model.
# * **Weight tying.** The output layer $W_2$ also has one column per token. Many LLMs
#   (GPT-2, Gemma) *tie* it to the embedding table, $W_2 = C^\top$, so that a token's input and
#   output representations coincide.
# * **Fixed context → unbounded context.** Our model sees exactly $k=3$ characters and
#   concatenates them, so its parameter count grows with $k$. WaveNet (LM 6) grows the
#   context hierarchically, RNNs (LM 7) compress it into a state, and transformers
#   (LM 8) let every position *attend* to all previous ones with shared weights.
# * **The MLP is still there.** Each transformer block contains an MLP that is the hidden
#   layer of this notebook — wider (typically $4\times$ the model width) and with GELU or
#   SwiGLU instead of tanh. Roughly two-thirds of an LLM's parameters are in these MLPs.
# * **Embeddings carry meaning.** The clustering of vowels in Section 7 is the
#   character-level version of the semantic structure found in LLM token embeddings and
#   hidden states, which interpretability research probes with exactly the linear
#   projections we used.
#
# ## Try it yourself
#
# 1. Increase the context to $k = 5$ or $8$. Does the validation NLL improve? How does the
#    parameter count change compared with an $n$-gram table?
# 2. Train with embedding dimension $d = 2$ and plot the embeddings directly, without PCA.
#    Which groups of characters form?
# 3. Make the model bigger ($h = 400$, $d = 24$) and train longer. When do training and validation
#    NLL start to diverge (overfitting)?
# 4. Remove the hidden layer (logits $= E W + \mathbf b$). How much worse is this
#    "linear" model, and how does it relate to LM 1's softmax regression?
# 5. Implement **weight tying** with $d = h$: replace $W_2$ by $C^\top$ (you need to add the two
#    gradient contributions to $C$). Does it help with this tiny vocabulary?
# 6. Implement the bias-free **direct connections** from the embeddings to the output that
#    Bengio et al. also used, $Z = HW_2 + EW_3 + \mathbf b_2$.
#
# **Further reading.**
# Bengio, Ducharme, Vincent & Jauvin (2003), "A neural probabilistic language model", *JMLR* 3;
# Jurafsky & Martin, *Speech and Language Processing* (3rd ed. draft), ch. 3 (n-grams and smoothing) and ch. 6–7 (embeddings, neural LMs);
# Mikolov et al. (2013), "Efficient estimation of word representations in vector space" (word2vec);
# Andrej Karpathy, "Building makemore Part 2: MLP" (video lecture, 2022).
