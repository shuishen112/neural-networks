# %% [markdown]
# # LM 4 — The MLP language model in PyTorch
#
# *Companion to Lecture 4 — From NumPy to PyTorch.*
#
# In LM 2 and LM 3 we built the Bengio et al. (2003) MLP language model in NumPy,
# with a hand-written backward pass. Here we rebuild exactly the same model with
# PyTorch: `nn.Embedding` for the lookup table, `F.cross_entropy` for the loss,
# an `nn.Module` for the model, AdamW for the updates and a `generate()` function
# for sampling. Every one of these pieces reappears, unchanged, inside a GPT
# (LM 8) — after this notebook, an LLM training script will look familiar.
#
# > **Language-model track:** 1 Bigram · 2 MLP LM · 3 Training dynamics · **4 PyTorch** · 5 Word-level & regularisation · 6 WaveNet · 7 Char-RNN · 8 GPT · 9 Sequence VAE · 10 Diffusion LM
#
# **Plan.**
# 1. Data: (context → next character) examples as `long` tensors
# 2. `nn.Embedding` is a one-hot matrix product (checked numerically)
# 3. `F.cross_entropy` on logits, and why not `log(softmax(...))`
# 4. The model as an `nn.Module`
# 5. Mini-batches: `Dataset`/`DataLoader` versus random indexing
# 6. Training with AdamW; train/eval mode; `@torch.no_grad` evaluation
# 7. Sampling with temperature, reproducibly
# 8. Saving and loading a `state_dict`
# 9. A longer context: `block_size` 3 versus 8

# %%
import math
import os
import random
import tempfile
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(0)
plt.rcParams["figure.figsize"] = (5, 4)
plt.rcParams["axes.grid"] = True

device = "cuda" if torch.cuda.is_available() else "cpu"
print("PyTorch", torch.__version__, "| device:", device)

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
# ## 1. Data: the same split and vocabulary as LM 1–3
#
# We use the shared conventions of the track: shuffle the names with
# `random.Random(42)`, split 80 / 10 / 10, and use the 27-symbol vocabulary with
# `"."` (index 0) as start and end token. Because the split is made with Python's
# `random`, it is *identical* to the one in the NumPy notebooks, so all validation
# numbers are comparable.

# %%
words = load_text("names.txt").split()
random.Random(42).shuffle(words)
n1, n2 = int(0.8 * len(words)), int(0.9 * len(words))
train_words, val_words, test_words = words[:n1], words[n1:n2], words[n2:]

chars = sorted(set("".join(words)))
stoi = {ch: i + 1 for i, ch in enumerate(chars)}
stoi["."] = 0
itos = {i: ch for ch, i in stoi.items()}
V = len(stoi)
print(f"{len(words)} names | train {len(train_words)} / val {len(val_words)} / test {len(test_words)} | V = {V}")
assert V == 27 and itos[0] == "." and itos[1] == "a" and itos[26] == "z"

# %% [markdown]
# A name such as `emma` yields one training example per character *plus* the end
# token. With a context ("block") of $3$ characters, padded on the left with `"."`:
#
# | context | target |
# |---|---|
# | `...` | `e` |
# | `..e` | `m` |
# | `.em` | `m` |
# | `emm` | `a` |
# | `mma` | `.` |
#
# In PyTorch, the inputs and targets are **integer** tensors of dtype `torch.long`
# (= `int64`): integers index the embedding table, and `F.cross_entropy` expects
# class indices of this dtype.

# %%
def build_dataset(words, block_size):
    """Return X: (N, block_size) long tensor of contexts, Y: (N,) long tensor of next characters."""
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]          # slide the window
    return torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)

Xex, Yex = build_dataset(["emma"], block_size=3)
for x, y in zip(Xex.tolist(), Yex.tolist()):
    print("".join(itos[i] for i in x), "->", itos[y])

block_size = 3
Xtr, Ytr = build_dataset(train_words, block_size)
Xva, Yva = build_dataset(val_words, block_size)
Xte, Yte = build_dataset(test_words, block_size)
print("train", tuple(Xtr.shape), Xtr.dtype, "| val", tuple(Xva.shape), "| test", tuple(Xte.shape))
assert Xtr.shape == (sum(len(w) + 1 for w in train_words), block_size)

# %% [markdown]
# ## 2. `nn.Embedding` is a matrix product with one-hot vectors
#
# In LM 2 the first layer was a lookup `C[X]` into a matrix $C\in\mathbb R^{V\times d}$.
# Mathematically, the lookup of token $i$ is the product of a one-hot row vector
# with $C$:
#
# $$\mathbf e_i^\top C = C_{i,:}, \qquad \mathbf e_i = (0,\dots,0,\underbrace{1}_{i},0,\dots,0)^\top .$$
#
# So an embedding layer is just a linear layer without bias whose input happens to
# be one-hot — but *implemented* as indexing, which costs $O(d)$ instead of
# $O(Vd)$ per token. For an LLM with $V\approx 10^5$ that difference is
# enormous. `nn.Embedding(V, d)` stores $C$ as `.weight` (shape `(V, d)`,
# initialised $\mathcal N(0,1)$) and accepts integer tensors of *any* shape,
# appending a dimension of size $d$.

# %%
torch.manual_seed(0)
emb = nn.Embedding(V, 10)
idx = Xtr[:4]                                             # (4, 3) long
by_lookup = emb(idx)                                      # (4, 3, 10)
one_hot = F.one_hot(idx, num_classes=V).float()           # (4, 3, 27)
by_matmul = one_hot @ emb.weight                          # (4, 3, 27) @ (27, 10) -> (4, 3, 10)
print("lookup:", tuple(by_lookup.shape), "| one-hot @ W:", tuple(by_matmul.shape))
print("max |difference|:", (by_lookup - by_matmul).abs().max().item())
assert torch.allclose(by_lookup, by_matmul)

# %% [markdown]
# The gradient tells the same story: the derivative of the loss with respect to
# $C$ is $\text{one\_hot}^\top\,\partial\mathcal L/\partial E$, which is non-zero only
# in the rows of the tokens that occurred in the batch (and a row that occurs
# several times accumulates several contributions — exactly the `np.add.at` of
# LM 2).

# %%
emb.zero_grad()
emb(idx).sum().backward()
rows_with_grad = (emb.weight.grad.abs().sum(dim=1) > 0).nonzero().flatten().tolist()
print("tokens in the batch     :", sorted(set(idx.flatten().tolist())))
print("rows with non-zero grad :", rows_with_grad)
assert rows_with_grad == sorted(set(idx.flatten().tolist()))

# %% [markdown]
# ## 3. The loss: `F.cross_entropy` takes logits
#
# The model outputs **logits** $\mathbf z\in\mathbb R^V$. The negative
# log-likelihood of the correct next character $y$ is
#
# $$\ell(\mathbf z, y) = -\log\operatorname{softmax}(\mathbf z)_y
# = -z_y + \log\sum_{k} e^{z_k} .$$
#
# `F.cross_entropy(logits, targets)` computes the right-hand side directly with
# the log-sum-exp trick ($\log\sum_k e^{z_k} = m + \log\sum_k e^{z_k-m}$ with
# $m=\max_k z_k$). The tempting two-step version `torch.log(F.softmax(z))` first
# rounds probabilities to float32 and then takes their log. When the model is
# confident — and LMs routinely are, e.g. after `q` comes `u` — the probability
# of an unlikely target underflows to $0$ and its log is $-\infty$:

# %%
logits = torch.tensor([[30.0, -80.0, 0.0], [2.0, 1.0, 0.0]])
targets = torch.tensor([1, 0])
two_step = -torch.log(F.softmax(logits, dim=1))[torch.arange(2), targets]
fused = F.cross_entropy(logits, targets, reduction="none")
print("-log(softmax(z))_y :", two_step.tolist())
print("F.cross_entropy    :", fused.tolist())
print("F.log_softmax route:", (-F.log_softmax(logits, dim=1)[torch.arange(2), targets]).tolist())

# %% [markdown]
# The fused loss is also faster and has the simple gradient we derived by hand in
# LM 2, $\partial\ell/\partial\mathbf z = \operatorname{softmax}(\mathbf z) - \mathbf e_y$:

# %%
z = torch.randn(5, V, requires_grad=True)
y = torch.randint(0, V, (5,))
F.cross_entropy(z, y, reduction="sum").backward()
manual = F.softmax(z.detach(), dim=1) - F.one_hot(y, V).float()
print("grad = softmax - one_hot:", torch.allclose(z.grad, manual, atol=1e-6))
assert torch.allclose(z.grad, manual, atol=1e-6)

# %% [markdown]
# Shape convention: for $N$ examples, `F.cross_entropy` wants logits of shape
# $(N, V)$ and integer targets of shape $(N,)$. (For sequences it also accepts
# $(N, V, T)$ with targets $(N, T)$ — we use that in LM 6.) By default it returns
# the **mean** over the examples: exactly the "nats per character" we report.
#
# ## 4. The model as an `nn.Module`
#
# The Bengio MLP language model:
#
# $$\mathbf x = [C_{c_1}; C_{c_2}; \dots; C_{c_B}] \in \mathbb R^{Bd},\qquad
# \mathbf h = \tanh(W_1\mathbf x + \mathbf b_1),\qquad
# \mathbf z = W_2\mathbf h + \mathbf b_2 \in\mathbb R^{V},$$
#
# where $c_1,\dots,c_B$ are the $B$ = `block_size` previous characters and
# $[\,\cdot\,;\,\cdot\,]$ is concatenation. In PyTorch the concatenation is a
# `view`/`flatten` of the `(batch, block_size, d)` embedding tensor to
# `(batch, block_size·d)`.
#
# Following LM 3, we shrink the initial output weights so that the untrained
# model predicts a nearly uniform distribution (initial loss $\approx\ln 27$
# instead of a confidently wrong, much larger value).

# %%
class MLPLM(nn.Module):
    """Bengio et al. (2003) MLP language model: embed, concatenate, tanh hidden layer, logits."""

    def __init__(self, vocab_size, block_size, n_embd=16, n_hidden=256):
        super().__init__()
        self.block_size = block_size
        self.emb = nn.Embedding(vocab_size, n_embd)
        self.hidden = nn.Linear(block_size * n_embd, n_hidden)
        self.out = nn.Linear(n_hidden, vocab_size)
        with torch.no_grad():                    # LM 3: start close to the uniform distribution
            self.out.weight.mul_(0.1)
            self.out.bias.zero_()

    def forward(self, idx):                      # idx: (B, block_size) long
        e = self.emb(idx)                        # (B, block_size, n_embd)
        h = torch.tanh(self.hidden(e.flatten(start_dim=1)))   # (B, n_hidden)
        return self.out(h)                       # (B, vocab_size) logits


torch.manual_seed(0)
model = MLPLM(V, block_size).to(device)
print(model)
for name, p in model.named_parameters():
    print(f"{name:15s} {str(tuple(p.shape)):12s} {p.numel():6d}")
print("parameters:", sum(p.numel() for p in model.parameters()))

with torch.no_grad():
    out = model(Xtr[:32].to(device))
    print("logits:", tuple(out.shape), "| initial loss %.3f (ln 27 = %.3f)"
          % (F.cross_entropy(out, Ytr[:32].to(device)).item(), math.log(27)))
assert out.shape == (32, V)

# %% [markdown]
# ## 5. Mini-batches: `DataLoader` or random indexing?
#
# Lecture 4 introduced `Dataset` and `DataLoader`. For the names data a dataset is
# two lines of code. A `DataLoader` then shuffles, fetches `batch_size` examples
# **one by one** through `__getitem__` and *collates* (stacks) them.
#
# The alternative used in many LM codebases (e.g. Karpathy's makemore and nanoGPT)
# is to keep the whole training set in one tensor and draw a batch of random row
# indices: `ix = torch.randint(0, N, (batch_size,))`, then `X[ix]` — a single
# vectorised gather. It samples *with* replacement instead of going through
# epochs, which makes no practical difference with 180 000 examples.

# %%
class NextCharDataset(Dataset):
    def __init__(self, X, Y):
        self.X, self.Y = X, Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]


loader = DataLoader(NextCharDataset(Xtr, Ytr), batch_size=64, shuffle=True,
                    generator=torch.Generator().manual_seed(0))
xb, yb = next(iter(loader))
print("DataLoader batch:", tuple(xb.shape), xb.dtype, tuple(yb.shape))

n_batches = 300
t0 = time.perf_counter()
for k, (xb, yb) in enumerate(loader):
    if k == n_batches:
        break
t_loader = (time.perf_counter() - t0) / n_batches

g = torch.Generator().manual_seed(0)
t0 = time.perf_counter()
for k in range(n_batches):
    ix = torch.randint(0, Xtr.shape[0], (64,), generator=g)
    xb, yb = Xtr[ix], Ytr[ix]
t_index = (time.perf_counter() - t0) / n_batches

opt_tmp = torch.optim.AdamW(model.parameters(), lr=1e-3)
t0 = time.perf_counter()
for k in range(n_batches):
    loss = F.cross_entropy(model(xb.to(device)), yb.to(device))
    opt_tmp.zero_grad()
    loss.backward()
    opt_tmp.step()
t_step = (time.perf_counter() - t0) / n_batches

print(f"DataLoader          : {1e3 * t_loader:7.3f} ms per batch")
print(f"random indexing     : {1e3 * t_index:7.3f} ms per batch")
print(f"one training step   : {1e3 * t_step:7.3f} ms (forward + backward + AdamW)")

# %% [markdown]
# For a model this small, the `DataLoader` (64 Python-level `__getitem__` calls and
# a collate per batch) can easily cost as much as — or more than — the training
# step itself, while random indexing is almost free. For large models the
# balance flips: the step dominates, and `DataLoader`'s worker processes,
# shuffling-by-epoch and pinned memory become valuable (LLM pipelines stream
# pre-tokenised shards through exactly such loaders). We use random indexing from
# now on.
#
# ## 6. Training with AdamW
#
# The canonical loop of Lecture 4, with three LM-specific details:
#
# * **AdamW** (Adam with decoupled weight decay, Lecture 5) is the standard
#   optimiser for language models; we use a small decay of $0.01$ (PyTorch's default).
# * A simple **step learning-rate schedule**: divide the rate by 10 for the last
#   quarter of training (LM 3).
# * **Evaluation** in `model.eval()` mode under `@torch.no_grad()`. This MLP has
#   no dropout or batch norm, so the mode does not change its output — but forgetting
#   it is a classic bug once such layers appear (LM 5), so we form the habit now.
#   `@torch.no_grad()` stops autograd from recording the graph, saving memory and time.
#
# The evaluation loss is a sum over all examples divided by their number, i.e. the
# exact mean NLL in **nats per character**.

# %%
def get_batch(X, Y, batch_size, generator):
    ix = torch.randint(0, X.shape[0], (batch_size,), generator=generator)
    return X[ix].to(device), Y[ix].to(device)


@torch.no_grad()
def evaluate(model, X, Y, batch_size=8192):
    """Mean negative log-likelihood (nats per character) over a whole split."""
    was_training = model.training
    model.eval()
    total = 0.0
    for i in range(0, X.shape[0], batch_size):
        xb, yb = X[i:i + batch_size].to(device), Y[i:i + batch_size].to(device)
        total += F.cross_entropy(model(xb), yb, reduction="sum").item()
    model.train(was_training)
    return total / X.shape[0]


def train(model, Xtr, Ytr, Xva, Yva, steps=15000, batch_size=128, lr=3e-3,
          weight_decay=0.01, eval_every=1000, seed=0):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    g = torch.Generator().manual_seed(seed)
    history = {"step": [], "train": [], "val": []}
    model.train()
    for step in range(steps + 1):
        if step % eval_every == 0 or step == steps:
            history["step"].append(step)
            history["train"].append(evaluate(model, Xtr, Ytr))
            history["val"].append(evaluate(model, Xva, Yva))
        if step == steps:
            break
        for group in optimizer.param_groups:              # step schedule
            group["lr"] = lr if step < 0.75 * steps else lr / 10
        xb, yb = get_batch(Xtr, Ytr, batch_size, g)
        loss = F.cross_entropy(model(xb), yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return history


torch.manual_seed(0)
model = MLPLM(V, block_size=3).to(device)
t0 = time.perf_counter()
hist3 = train(model, Xtr, Ytr, Xva, Yva)
print(f"trained in {time.perf_counter() - t0:.1f} s")
for s, tr, va in zip(hist3["step"], hist3["train"], hist3["val"]):
    if s % 3000 == 0 or s == hist3["step"][-1]:
        print(f"step {s:6d} | train {tr:.3f} | val {va:.3f}")

# %%
plt.plot(hist3["step"], hist3["train"], label="train")
plt.plot(hist3["step"], hist3["val"], label="validation")
plt.axhline(math.log(27), color="gray", ls=":", label="uniform (ln 27)")
plt.ylim(1.9, 3.4)
plt.xlabel("step"); plt.ylabel("NLL (nats / char)"); plt.title("MLP LM, block_size = 3")
plt.legend(); plt.show()

# %% [markdown]
# The drop at 75 % of the steps is the learning-rate decay. Train and validation
# curves stay close: with ~20 k parameters and ~180 k training characters this
# model is far from overfitting (LM 5 shows what happens when it is not).
#
# ## 7. Sampling with temperature
#
# To generate a name we start from the context `...`, compute the next-character
# distribution, **sample** from it with `torch.multinomial`, slide the window and
# repeat until `"."` is drawn. A **temperature** $\tau$ rescales the logits before
# the softmax,
#
# $$p_\tau(k) = \frac{e^{z_k/\tau}}{\sum_j e^{z_j/\tau}},$$
#
# so $\tau<1$ sharpens the distribution (safer, more repetitive samples),
# $\tau>1$ flattens it (more surprising, more errors) and $\tau\to0$ becomes
# greedy arg-max decoding. LLM APIs expose exactly this knob.
#
# For reproducible samples we pass an explicit `torch.Generator` to
# `torch.multinomial`: the same seed then gives the same names, independently of
# any other random numbers drawn elsewhere in the notebook.

# %%
@torch.no_grad()
def generate(model, n_samples=10, temperature=1.0, generator=None, max_len=30):
    model.eval()
    names = []
    for _ in range(n_samples):
        context = [0] * model.block_size
        out = []
        while len(out) < max_len:
            x = torch.tensor([context], dtype=torch.long, device=device)   # (1, block_size)
            logits = model(x)[0] / temperature                             # (V,)
            probs = F.softmax(logits, dim=-1)
            ix = torch.multinomial(probs, num_samples=1, generator=generator).item()
            if ix == 0:
                break
            out.append(itos[ix])
            context = context[1:] + [ix]
        names.append("".join(out))
    return names


def make_gen(seed):
    return torch.Generator(device=device).manual_seed(seed)

for tau in (0.5, 1.0, 1.5):
    print(f"temperature {tau}:", ", ".join(generate(model, 10, tau, make_gen(42))))

same_a = generate(model, 5, 1.0, make_gen(123))
same_b = generate(model, 5, 1.0, make_gen(123))
print("same seed, same samples:", same_a == same_b)
assert same_a == same_b

# %% [markdown]
# How does temperature change the distribution itself? Below: the model's
# next-character distribution after the context `mar`, at three temperatures,
# together with its entropy $H = -\sum_k p_k\ln p_k$ (in nats; $\ln 27 \approx 3.30$
# is the maximum, reached by the uniform distribution).

# %%
ctx = torch.tensor([[stoi[c] for c in "mar"]], device=device)
with torch.no_grad():
    z = model(ctx)[0].cpu()
fig, axes = plt.subplots(1, 3, figsize=(13, 3), sharey=True)
for ax, tau in zip(axes, (0.5, 1.0, 1.5)):
    p = F.softmax(z / tau, dim=0)
    H = torch.special.entr(p).sum().item()          # -sum p log p, with 0 log 0 = 0
    ax.bar(range(V), p.numpy())
    ax.set_xticks(range(V)); ax.set_xticklabels([itos[i] for i in range(V)], fontsize=7)
    ax.set_title(f"p(next | 'mar'),  τ = {tau},  H = {H:.2f} nats")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 8. Saving and loading
#
# As in Lecture 4 we save the `state_dict` — a dictionary from parameter names to
# tensors — and restore it into a freshly constructed model of the same
# architecture. The restored model must give *exactly* the same validation loss.

# %%
print({k: tuple(v.shape) for k, v in model.state_dict().items()})
val_before = evaluate(model, Xva, Yva)

with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "mlp_lm.pt")
    torch.save(model.state_dict(), path)
    print(f"saved {os.path.getsize(path) / 1024:.1f} kB")
    restored = MLPLM(V, block_size=3).to(device)          # fresh random weights ...
    print(f"fresh model val NLL   : {evaluate(restored, Xva, Yva):.4f}")
    restored.load_state_dict(torch.load(path, map_location=device, weights_only=True))

val_after = evaluate(restored, Xva, Yva)
print(f"original model val NLL: {val_before:.6f}")
print(f"restored model val NLL: {val_after:.6f}")
assert abs(val_before - val_after) < 1e-6
print("restored samples:", ", ".join(generate(restored, 5, 1.0, make_gen(123))))

# %% [markdown]
# Same weights, same generator seed ⇒ the restored model also produces the same
# names as above. Note that the hyper-parameters (`block_size`, `n_embd`,
# `n_hidden`) are *not* in the `state_dict`; real checkpoints store them alongside
# (e.g. nanoGPT saves a `model_args` dictionary), together with the optimiser
# state if training is to be resumed.
#
# ## 9. A longer context: `block_size` 3 versus 8
#
# With three characters of context the model cannot know how long the name
# already is, or what its first letters were. In the MLP, extending the context is
# trivial — the first layer just gets wider ($8\cdot16=128$ instead of $48$ inputs).
# Everything else, including the training code, stays the same.

# %%
block_size = 8
Xtr8, Ytr8 = build_dataset(train_words, block_size)
Xva8, Yva8 = build_dataset(val_words, block_size)
Xte8, Yte8 = build_dataset(test_words, block_size)
assert Xtr8.shape == (Xtr.shape[0], 8) and torch.equal(Ytr8, Ytr)   # same targets, longer contexts

torch.manual_seed(0)
model8 = MLPLM(V, block_size=8).to(device)
print("parameters:", sum(p.numel() for p in model8.parameters()))
t0 = time.perf_counter()
hist8 = train(model8, Xtr8, Ytr8, Xva8, Yva8)
print(f"trained in {time.perf_counter() - t0:.1f} s")

# %%
plt.plot(hist3["step"], hist3["val"], label="block_size 3 (val)")
plt.plot(hist8["step"], hist8["val"], label="block_size 8 (val)")
plt.plot(hist8["step"], hist8["train"], ls="--", color="C1", alpha=0.6, label="block_size 8 (train)")
plt.axhline(2.45, color="gray", ls=":", label="bigram (LM 1)")
plt.ylim(1.7, 2.8)
plt.xlabel("step"); plt.ylabel("NLL (nats / char)"); plt.legend()
plt.title("Longer context"); plt.show()

print(f"{'model':16s} {'params':>7s} {'train NLL':>10s} {'val NLL':>8s} {'val ppl':>8s}")
for name, m, h in [("block_size 3", model, hist3), ("block_size 8", model8, hist8)]:
    n_par = sum(p.numel() for p in m.parameters())
    print(f"{name:16s} {n_par:7d} {h['train'][-1]:10.3f} {h['val'][-1]:8.3f} {math.exp(h['val'][-1]):8.2f}")

# %% [markdown]
# A longer context helps — but look at the price. The first layer is now more
# than twice as large, and the gap between training and validation loss has
# grown: the MLP has to learn *separately* what the character two positions back
# means and what the character six positions back means. Position $j$ of the
# context has its own block of weights in $W_1$, and nothing is shared between
# positions, so every extra character of context costs parameters and data. The next notebooks
# address exactly this: convolutions share weights across positions (LM 6),
# recurrent networks process any length with one cell (LM 7), and attention lets
# every position look up whatever it needs (LM 8).
#
# Samples from the block-8 model, and the final numbers. The test set is touched
# once, for the model chosen on validation data.

# %%
print("block_size 8:", ", ".join(generate(model8, 12, 1.0, make_gen(7))))

hists = {3: hist3, 8: hist8}
best_bs = min(hists, key=lambda b: hists[b]["val"][-1])
best_model, Xte_b, Yte_b = (model, Xte, Yte) if best_bs == 3 else (model8, Xte8, Yte8)
val_nll = hists[best_bs]["val"][-1]
print(f"best block_size on validation: {best_bs}")
print(f"test NLL (nats/char): {evaluate(best_model, Xte_b, Yte_b):.3f}")
print(f"val NLL (nats/char): {val_nll:.3f}")
assert val_nll < 2.6

# %% [markdown]
# ## From here to LLMs
#
# Every line of this notebook has a direct counterpart in an LLM training script:
#
# * **`nn.Embedding`** is the first layer of every LLM (GPT-2: a $50\,257\times768$
#   table; Llama-3 8B: $128\,256\times4096$). It is a one-hot product implemented as
#   a lookup, and its gradient is sparse — only the rows of tokens in the batch change.
# * **`F.cross_entropy` on logits** is *the* pre-training loss: next-token
#   prediction, averaged over all positions. Reported per token it is the NLL;
#   $e^{\text{NLL}}$ is the perplexity quoted in papers. LLM codebases go to great
#   lengths to keep it numerically stable (and in float32) even when the rest of
#   the model runs in bfloat16.
# * **AdamW** with a learning-rate schedule (warm-up, then cosine decay) is still the
#   default optimiser for pre-training.
# * **Random-offset batching** (`torch.randint` into one long token tensor) is how
#   nanoGPT samples training windows; production systems stream tokenised shards
#   through a `DataLoader`.
# * **`generate()` with temperature** is how chatbots decode, usually combined with
#   *top-k* or *top-p* (nucleus) truncation of the distribution. Seeded generators
#   make samples reproducible.
# * **`state_dict` checkpoints** are what you download when you download "open
#   weights": a dictionary of named tensors (today usually in the `safetensors`
#   format) plus a configuration file with the hyper-parameters.
#
# What does *not* survive is the fixed context with position-specific weights:
# its parameter count grows linearly with `block_size`, and nothing learned about
# position 3 transfers to position 7.
#
# ## Try it yourself
#
# 1. Replace the step schedule by `torch.optim.lr_scheduler.CosineAnnealingLR` and
#    compare. Does a warm-up phase help?
# 2. Add *top-k* sampling to `generate()`: keep only the $k$ most probable
#    characters (`torch.topk`), renormalise, sample. How do $k=3$ and $k=27$ differ?
# 3. Swap `tanh` for `F.relu` or `F.gelu`. Does the output-layer scaling from LM 3
#    still matter? Remove it and look at the first evaluation.
# 4. Make the embedding 2-dimensional (`n_embd=2`), train, and scatter-plot
#    `model.emb.weight` with the characters as labels. Which letters cluster?
# 5. Train with `block_size=16`. Does the gap between train and validation loss grow?
# 6. Time a training step on a GPU (Colab) for batch sizes 32 … 4096. Where does
#    the GPU start to beat the CPU?
#
# **Further reading.**
# Bengio, Ducharme, Vincent & Jauvin, *A Neural Probabilistic Language Model*, JMLR 2003;
# Loshchilov & Hutter, *Decoupled Weight Decay Regularization*, ICLR 2019;
# Karpathy, *makemore* (github.com/karpathy/makemore) and the lecture "Building makemore
# Part 2: MLP" (Neural Networks: Zero to Hero);
# Holtzman et al., *The Curious Case of Neural Text Degeneration*, ICLR 2020 (temperature,
# top-k and nucleus sampling);
# the PyTorch documentation of `nn.Embedding`, `F.cross_entropy` and `torch.multinomial`.
