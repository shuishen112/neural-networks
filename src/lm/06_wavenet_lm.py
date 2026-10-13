# %% [markdown]
# # LM 6 — WaveNet: causal dilated convolutions for language modelling
#
# *Companion to Lecture 6 — Convolutional Neural Networks.*
#
# Lecture 6 slid small, shared filters over images. Text is a 1-D sequence, and a
# convolution slid along it is a natural language model — provided it never looks
# into the future. We build **causal** convolutions, stack them with **dilations**
# so that the receptive field grows exponentially (WaveNet, van den Oord et al.
# 2016), and train the network to predict the next character at **every position
# of a name at once**. That last idea — one forward pass, a loss at every position,
# masks to forbid peeking — is exactly how GPT is trained (LM 8).
#
# > **Language-model track:** 1 Bigram · 2 MLP LM · 3 Training dynamics · 4 PyTorch · 5 Word-level & regularisation · **6 WaveNet** · 7 Char-RNN · 8 GPT · 9 Sequence VAE · 10 Diffusion LM
#
# **Plan.**
# 1. Data: whole names as padded sequences, targets shifted by one
# 2. Causal 1-D convolution by left padding — and a check that it cannot peek
# 3. Dilations: exponential receptive-field growth
# 4. The WaveNet language model (gated residual blocks)
# 5. Training at every position in parallel, with a masked loss
# 6. Sampling
# 7. The same computation as a hierarchical MLP
# 8. Validation loss versus receptive field

# %%
import math
import random
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

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
# ## 1. Data: names as sequences
#
# Same split and vocabulary as the rest of the track. Until now (LM 2–4) every
# training example was one (context → next character) pair, so a name of length
# $n$ became $n+1$ separate examples, each recomputing the embeddings of its
# context. Now a whole name is **one** example. For `emma`:
#
# | position $t$ | 0 | 1 | 2 | 3 | 4 | 5 … 15 |
# |---|---|---|---|---|---|---|
# | input $x_t$ | `.` | `e` | `m` | `m` | `a` | padding |
# | target $y_t$ | `e` | `m` | `m` | `a` | `.` | *ignored* (−1) |
#
# The target sequence is the input sequence **shifted left by one**: at position
# $t$ the model has read $x_0,\dots,x_t$ and must predict $x_{t+1}$. Names have
# different lengths, so we pad to a common length $T$. Padded *targets* get the
# value $-1$, which we will tell the loss to ignore; padded *inputs* can be anything
# (we use `.`), because — as we will make sure — a causal model's predictions at
# real positions never see them.

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
assert V == 27 and itos[0] == "."

T_MAX = max(len(w) for w in words) + 1          # +1 for the start token
IGNORE = -1

def build_sequences(words, T=T_MAX):
    """X: (N, T) inputs '.'+name, Y: (N, T) targets name+'.', padded with IGNORE; lengths: (N,)."""
    X = torch.zeros((len(words), T), dtype=torch.long)          # padded inputs = '.' (index 0)
    Y = torch.full((len(words), T), IGNORE, dtype=torch.long)
    for i, w in enumerate(words):
        ids = [stoi[c] for c in w]
        X[i, :len(ids) + 1] = torch.tensor([0] + ids)
        Y[i, :len(ids) + 1] = torch.tensor(ids + [0])
    lengths = torch.tensor([len(w) + 1 for w in words])
    return X, Y, lengths

Xtr, Ytr, Ltr = build_sequences(train_words)
Xva, Yva, Lva = build_sequences(val_words)
Xte, Yte, Lte = build_sequences(test_words)
print("T =", T_MAX, "| train", tuple(Xtr.shape), "| val", tuple(Xva.shape))
print("input :", "".join(itos[i] for i in Xtr[0].tolist()))
print("target:", "".join(itos[i] if i >= 0 else "_" for i in Ytr[0].tolist()))

# shifted-by-one property and target counts (the same 22 655 val predictions as the MLP LMs)
assert torch.equal(Xtr[:, 1:][Ytr[:, :-1] > 0], Ytr[:, :-1][Ytr[:, :-1] > 0])
assert (Yva != IGNORE).sum().item() == sum(len(w) + 1 for w in val_words)
print("val predictions:", (Yva != IGNORE).sum().item())

# %% [markdown]
# ## 2. Causal convolution
#
# A 1-D convolution (cross-correlation, Lecture 6) with kernel size $k$, $C_{in}$
# input and $C_{out}$ output channels computes, at position $t$,
#
# $$y_t = \mathbf b + \sum_{j=0}^{k-1} W_j\, x_{t+j-p},$$
#
# where $p$ is the amount of padding added on the left. `nn.Conv1d` expects
# tensors of shape **(batch, channels, time)** and has weight shape
# $(C_{out}, C_{in}, k)$. With the usual symmetric "same" padding ($p=(k-1)/2$ on
# both sides) the output at $t$ depends on $x_{t+1},\dots$ — the future. For a
# language model that is cheating: during training it would simply copy the next
# character, which it can see, and at generation time that character does not
# exist yet.
#
# A **causal** convolution pads with $p=k-1$ zeros **on the left only**
# (`F.pad(x, (k - 1, 0))`) and none on the right. Then
# $y_t = \mathbf b + \sum_j W_j x_{t-(k-1)+j}$ uses only $x_{t-k+1},\dots,x_t$, and
# the output has the same length $T$ as the input.

# %%
class CausalConv1d(nn.Module):
    """1-D convolution whose output at time t depends only on inputs at times <= t."""

    def __init__(self, c_in, c_out, kernel_size, dilation=1):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, kernel_size, dilation=dilation)   # no built-in padding

    def forward(self, x):                                  # x: (B, C_in, T)
        return self.conv(F.pad(x, (self.left_pad, 0)))     # (B, C_out, T)


torch.manual_seed(0)
causal = CausalConv1d(4, 4, kernel_size=3)
same = nn.Conv1d(4, 4, kernel_size=3, padding=1)            # symmetric padding: NOT causal
x = torch.randn(1, 4, 10)
print("causal output:", tuple(causal(x).shape), "| 'same' output:", tuple(same(x).shape))

# %% [markdown]
# **Checking causality.** Two equivalent tests:
#
# 1. *Perturbation:* change the inputs after position $t$; the outputs up to $t$
#    must not change.
# 2. *Gradient:* $\partial y_t/\partial x_s$ must be zero for all $s>t$. Its
#    non-zero entries also show the **receptive field** directly.

# %%
def future_leak(layer, t=5, T=10, C=4):
    """Max change of outputs at positions <= t when inputs after t are replaced."""
    x = torch.randn(1, C, T)
    x2 = x.clone()
    x2[:, :, t + 1:] = torch.randn(1, C, T - t - 1)
    with torch.no_grad():
        return (layer(x)[:, :, :t + 1] - layer(x2)[:, :, :t + 1]).abs().max().item()

def input_gradient(layer, t, T=16, C=4):
    """|d sum(y_t) / d x_s| summed over channels, for every input position s."""
    x = torch.randn(1, C, T, requires_grad=True)
    layer(x)[0, :, t].sum().backward()
    return x.grad[0].abs().sum(dim=0)

print(f"causal conv: max change of past outputs = {future_leak(causal):.2e}")
print(f"'same' conv: max change of past outputs = {future_leak(same):.2e}   <- peeks into the future")
assert future_leak(causal) < 1e-6

t = 9
fig, axes = plt.subplots(1, 2, figsize=(10, 2.2), sharey=True)
for ax, layer, title in [(axes[0], CausalConv1d(4, 4, 3), "causal, k = 3"), (axes[1], nn.Conv1d(4, 4, 3, padding=1), "'same' padding, k = 3")]:
    g = input_gradient(layer, t)
    ax.bar(range(16), (g > 0).float().numpy(), color=["C0" if s <= t else "C3" for s in range(16)])
    ax.axvline(t + 0.5, color="k", ls=":")
    ax.set_title(f"{title}: inputs that affect output t = {t}"); ax.set_xlabel("input position s")
axes[0].set_yticks([0, 1]); axes[0].set_yticklabels(["no", "yes"])
plt.tight_layout(); plt.show()
assert input_gradient(CausalConv1d(4, 4, 3), t)[t + 1:].sum().item() == 0.0

# %% [markdown]
# ## 3. Dilated convolutions: exponential receptive fields
#
# A stack of $L$ causal convolutions with kernel size 2 sees only $L+1$ characters:
# the receptive field grows *linearly* with depth. A **dilated** convolution with
# dilation $d$ skips $d-1$ inputs between its taps:
#
# $$y_t = \mathbf b + W_0\, x_{t-d} + W_1\, x_t \qquad (k=2).$$
#
# The causal left padding becomes $(k-1)\,d$. Doubling the dilation in every layer,
# $d_l = 2^{l-1}$, makes the receptive field **double** with every layer:
#
# $$r_L = 1 + \sum_{l=1}^{L} (k-1)\, d_l \;\overset{k=2,\ d_l=2^{l-1}}{=}\; 2^L .$$
#
# This is the receptive-field formula of Lecture 6 with dilation instead of
# stride: dilation grows the receptive field like striding does, but without
# downsampling, so we still get an output at **every** position. WaveNet used this
# to model raw audio at 16 000 samples per second with receptive fields of
# thousands of samples.
#
# ```text
# layer 3, d = 4:                       y7
#                             ┌─────────┴─────────┐
# layer 2, d = 2:            h3                  h7
#                        ┌────┴────┐         ┌────┴────┐
# layer 1, d = 1:       g1        g3        g5        g7
#                      ┌─┴─┐     ┌─┴─┐     ┌─┴─┐     ┌─┴─┐
# input:              x0   x1   x2   x3   x4   x5   x6   x7
# ```
#
# (Only the activations that $y_7$ depends on are drawn; every layer of course
# produces an output at every position.)

# %%
def receptive_field(kernel_size, dilations):
    return 1 + sum((kernel_size - 1) * d for d in dilations)

print(f"{'layers':>6s} {'no dilation':>12s} {'dilation 2^l':>13s}")
for L in range(1, 9):
    print(f"{L:6d} {receptive_field(2, [1] * L):12d} {receptive_field(2, [2 ** l for l in range(L)]):13d}")

# %% [markdown]
# Let us measure the receptive field of a dilated stack with the gradient test,
# layer by layer.

# %%
torch.manual_seed(0)
T_demo = 24
dil_stack = nn.ModuleList([CausalConv1d(4, 4, 2, dilation=2 ** l) for l in range(4)])
fig, axes = plt.subplots(4, 1, figsize=(9, 4.2), sharex=True)
for L in range(1, 5):
    layer = nn.Sequential(*[nn.Sequential(conv, nn.Tanh()) for conv in dil_stack[:L]])
    g = input_gradient(layer, t=T_demo - 1, T=T_demo)
    support = (g > 0).nonzero().flatten()
    measured = (support.max() - support.min() + 1).item()
    formula = receptive_field(2, [2 ** l for l in range(L)])
    axes[L - 1].bar(range(T_demo), g.numpy(), color="C0")
    axes[L - 1].set_ylabel(f"L={L}", rotation=0, labelpad=20)
    axes[L - 1].set_yticks([])
    print(f"L = {L}: dilations {[2 ** l for l in range(L)]}, receptive field formula {formula}, measured {measured}")
    assert measured <= formula
axes[0].set_title(f"|d y_t / d x_s| for the last output t = {T_demo - 1}")
axes[-1].set_xlabel("input position s")
plt.tight_layout(); plt.show()

# %% [markdown]
# Four layers with kernel size 2 already see 16 characters — more than the longest
# name plus its start token ($T=16$), so the model below conditions on the entire
# prefix, like an RNN or a Transformer would.
#
# ## 4. The WaveNet language model
#
# Each **residual block** applies WaveNet's gated activation unit to a dilated
# causal convolution and adds the result back to its input:
#
# $$\mathbf h = \tanh(W_f * \mathbf x) \odot \sigma(W_g * \mathbf x), \qquad
# \mathbf x \leftarrow \mathbf x + W_r\,\mathbf h ,$$
#
# where $*$ is a dilated causal convolution, $\sigma$ the logistic sigmoid (a
# learned "gate" between 0 and 1, an idea from LSTMs, LM 7) and $W_r$ a $1\times1$
# convolution — a linear layer applied at every position independently. The
# residual connection (ResNet, Lecture 6) keeps gradients flowing through the stack.
# The full model:
#
# 1. `nn.Embedding` → `(B, T, C)`, transposed to `(B, C, T)` for `nn.Conv1d`;
# 2. $L$ residual blocks with dilations $1, 2, 4, \dots, 2^{L-1}$;
# 3. a head of $1\times1$ convolutions mapping $C$ channels to $V$ logits **at every
#    position**: output shape `(B, V, T)`.
#
# Everything except the dilated convolutions acts on each position separately, so
# causality of the whole model follows from causality of the convolutions.
#
# *Note on normalisation.* Lecture 5's batch normalisation would compute statistics
# over the batch *and over time*, mixing in padded positions and (at training
# time) statistics of future positions. Our network is shallow enough to train
# without it; deeper causal models use a per-position normalisation (LayerNorm,
# as in GPT) instead.

# %%
class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation, kernel_size=2):
        super().__init__()
        self.filter = CausalConv1d(channels, channels, kernel_size, dilation)
        self.gate = CausalConv1d(channels, channels, kernel_size, dilation)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):                                   # (B, C, T) -> (B, C, T)
        h = torch.tanh(self.filter(x)) * torch.sigmoid(self.gate(x))
        return x + self.proj(h)


class WaveNetLM(nn.Module):
    def __init__(self, vocab_size, n_levels=4, channels=64, kernel_size=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, channels)
        self.blocks = nn.ModuleList(
            [ResidualBlock(channels, 2 ** l, kernel_size) for l in range(n_levels)])
        self.head = nn.Sequential(
            nn.ReLU(), nn.Conv1d(channels, channels, 1),
            nn.ReLU(), nn.Conv1d(channels, vocab_size, 1))
        self.receptive_field = receptive_field(kernel_size, [2 ** l for l in range(n_levels)])
        with torch.no_grad():                              # start near the uniform distribution
            self.head[-1].weight.mul_(0.1)
            self.head[-1].bias.zero_()

    def forward(self, idx):                                # idx: (B, T) long
        x = self.emb(idx).transpose(1, 2)                  # (B, T, C) -> (B, C, T)
        for block in self.blocks:
            x = block(x)
        return self.head(x)                                # (B, V, T) logits


torch.manual_seed(0)
model = WaveNetLM(V, n_levels=4).to(device)
print(model)
print("parameters:", sum(p.numel() for p in model.parameters()), "| receptive field:", model.receptive_field)
out = model(Xtr[:8].to(device))
print("input", tuple(Xtr[:8].shape), "-> logits", tuple(out.shape))
assert out.shape == (8, V, T_MAX)

# %% [markdown]
# **Causality of the whole model.** Replace every token after position $t$ by random
# characters: the logits at positions $\le t$ must be unchanged. (This is also why
# the padding value does not matter.)

# %%
torch.manual_seed(1)
idx = Xtr[:16].clone().to(device)
t = 4
idx2 = idx.clone()
idx2[:, t + 1:] = torch.randint(0, V, (16, T_MAX - t - 1), device=device)
with torch.no_grad():
    diff_past = (model(idx)[:, :, :t + 1] - model(idx2)[:, :, :t + 1]).abs().max().item()
    diff_future = (model(idx)[:, :, t + 1:] - model(idx2)[:, :, t + 1:]).abs().max().item()
print(f"max change of logits at positions <= {t}: {diff_past:.2e}")
print(f"max change of logits at positions >  {t}: {diff_future:.2e}")
assert diff_past < 1e-5

# %% [markdown]
# ## 5. Training: a loss at every position
#
# For a batch of $B$ names the model returns logits of shape $(B, V, T)$ and the
# targets have shape $(B, T)$. `F.cross_entropy` accepts exactly this layout
# (classes in dimension 1). With `ignore_index=-1` the padded positions contribute
# nothing, and the mean is taken over the real positions only:
#
# $$\mathcal L = \frac{1}{\#\{(b,t): y_{bt}\neq -1\}}\sum_{(b,t):\,y_{bt}\neq -1} -\log p_\theta(y_{bt}\mid x_{b,0},\dots,x_{b,t}).$$
#
# This is the same per-character NLL as before — the validation set contains the
# same 22 655 predictions as for the MLP LMs — so the numbers are directly
# comparable. Let us check that `ignore_index` does what we think:

# %%
logits = torch.randn(3, V, 5)
targets = torch.tensor([[1, 2, 0, -1, -1], [4, 0, -1, -1, -1], [7, 8, 9, 0, -1]])
masked = F.cross_entropy(logits, targets, ignore_index=IGNORE)
keep = targets != IGNORE
by_hand = F.cross_entropy(logits.transpose(1, 2)[keep], targets[keep])   # (n_real, V) vs (n_real,)
print(f"ignore_index: {masked.item():.5f} | only real positions: {by_hand.item():.5f}")
assert torch.allclose(masked, by_hand, atol=1e-6)

# %% [markdown]
# **Why this matters.** The MLP LM of LM 4 made one prediction per forward pass of
# an example and recomputed the embedding of every context character for each of
# the $n+1$ examples of a name. Here a single forward pass over a name produces all
# $n+1$ predictions, and every intermediate activation is computed once and
# shared by all later positions that need it. Training a GPT works the same way:
# a sequence of 1 024 tokens gives 1 024 next-token losses from one forward pass,
# with a causal **mask** in the attention playing the role of our left padding.
#
# Each batch is trimmed to the longest name it contains (padding at the end only
# costs compute). Evaluation sums the loss over all real positions of a split.

# %%
def get_batch(X, Y, L, batch_size, generator):
    ix = torch.randint(0, X.shape[0], (batch_size,), generator=generator)
    T = int(L[ix].max())                                     # trim padding beyond the longest name
    return X[ix, :T].to(device), Y[ix, :T].to(device)


@torch.no_grad()
def evaluate(model, X, Y, batch_size=2048):
    """Mean NLL in nats per character over all real (non-padded) positions."""
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    for i in range(0, X.shape[0], batch_size):
        xb, yb = X[i:i + batch_size].to(device), Y[i:i + batch_size].to(device)
        total += F.cross_entropy(model(xb), yb, ignore_index=IGNORE, reduction="sum").item()
        count += (yb != IGNORE).sum().item()
    model.train(was_training)
    return total / count


def train(model, steps=3000, batch_size=64, lr=3e-3, eval_every=250, seed=0, verbose=True):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    g = torch.Generator().manual_seed(seed)
    history = {"step": [], "train": [], "val": []}
    model.train()
    for step in range(steps + 1):
        if step % eval_every == 0 or step == steps:
            history["step"].append(step)
            history["train"].append(evaluate(model, Xtr[:5000], Ytr[:5000]))
            history["val"].append(evaluate(model, Xva, Yva))
            if verbose:
                print(f"step {step:5d} | train {history['train'][-1]:.3f} | val {history['val'][-1]:.3f}")
        if step == steps:
            break
        for group in optimizer.param_groups:
            group["lr"] = lr if step < 0.75 * steps else lr / 10
        xb, yb = get_batch(Xtr, Ytr, Ltr, batch_size, g)
        logits = model(xb)                                   # (B, V, T)
        loss = F.cross_entropy(logits, yb, ignore_index=IGNORE)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return history


torch.manual_seed(0)
model = WaveNetLM(V, n_levels=4).to(device)
print(f"initial val NLL {evaluate(model, Xva, Yva):.3f} (ln 27 = {math.log(27):.3f})")
t0 = time.perf_counter()
hist = train(model)
print(f"trained in {time.perf_counter() - t0:.1f} s")

# %%
plt.plot(hist["step"], hist["train"], label="train (5 000 names)")
plt.plot(hist["step"], hist["val"], label="validation")
plt.axhline(2.45, color="gray", ls=":", label="bigram (LM 1)")
plt.ylim(1.8, 3.4)
plt.xlabel("step"); plt.ylabel("NLL (nats / char)"); plt.title("WaveNet LM, receptive field 16")
plt.legend(); plt.show()

# %% [markdown]
# Where in a name is the model most uncertain? Averaging the loss **per position**
# (another thing the all-positions view gives us for free) shows that the first
# character is hardest — any name can start with almost any letter — and that the
# loss drops once the model has seen a few characters.

# %%
with torch.no_grad():
    model.eval()
    lv = F.cross_entropy(model(Xva.to(device)), Yva.to(device), ignore_index=IGNORE, reduction="none").cpu()  # (N, T)
mask = Yva != IGNORE
per_pos = (lv * mask).sum(0) / mask.sum(0).clamp(min=1)
counts = mask.sum(0)
pos = [t for t in range(T_MAX) if counts[t] >= 50]
plt.bar(pos, per_pos[pos].numpy())
plt.xlabel("position t (0 = predict the first letter)"); plt.ylabel("mean NLL (nats)")
plt.title("Validation loss by position"); plt.show()

# %% [markdown]
# ## 6. Sampling
#
# Generation is still sequential: start with `"."`, run the model on the sequence
# so far, take the logits at the **last** position, sample, append, repeat. We
# recompute the whole prefix at every step — simple, and cheap for names. (Real
# WaveNet implementations cache the activations of earlier positions, the
# "fast WaveNet" trick; for Transformers the same idea is the **KV cache**.)

# %%
@torch.no_grad()
def generate(model, n_samples=10, temperature=1.0, generator=None, max_len=T_MAX + 10):
    model.eval()
    names = []
    for _ in range(n_samples):
        seq = [0]
        while len(seq) <= max_len:
            x = torch.tensor([seq], dtype=torch.long, device=device)   # (1, t+1)
            logits = model(x)[0, :, -1] / temperature                   # (V,) at the last position
            ix = torch.multinomial(F.softmax(logits, dim=-1), 1, generator=generator).item()
            if ix == 0:
                break
            seq.append(ix)
        names.append("".join(itos[i] for i in seq[1:]))
    return names

g = torch.Generator(device=device).manual_seed(42)
samples = generate(model, 20, 1.0, g)
print(", ".join(samples))
print(f"{sum(s in set(train_words) for s in samples)} of {len(samples)} samples are names from the training set")

# %% [markdown]
# ## 7. The same computation as a hierarchical MLP
#
# Karpathy's "makemore part 5" builds the same model from the MLP side: with a
# block of 8 characters, instead of concatenating all 8 embeddings at once, it
# concatenates **pairs** (a `FlattenConsecutive(2)` layer reshaping
# `(B, 8, C) → (B, 4, 2C)`) and applies a linear layer, then pairs of pairs
# `(B, 4, C) → (B, 2, 2C)`, then `(B, 2, C) → (B, 1, 2C)`: a binary tree that fuses
# information slowly.
#
# A kernel-2 convolution with dilation $d$ computes $W_0 x_{t-d} + W_1 x_t$ — a
# linear layer applied to the pair $[x_{t-d};\,x_t]$. So the tree is exactly a
# dilated causal convolution stack with dilations 1, 2, 4, **evaluated only at the
# last position**. The convolution evaluates the same tree at *every* position and
# shares the intermediate results between overlapping trees. We verify this by
# copying the convolution weights into the hierarchical MLP:
# `Linear.weight = [W_0, W_1]` concatenated along the input dimension.

# %%
def flatten_consecutive(x, n=2):
    """(B, T, C) -> (B, T // n, n * C): concatenate n neighbouring time steps."""
    B, T, C = x.shape
    return x.reshape(B, T // n, n * C)

torch.manual_seed(0)
C_demo = 8
convs = [CausalConv1d(C_demo, C_demo, 2, dilation=d) for d in (1, 2, 4)]
x = torch.randn(3, C_demo, 8)                                # (B, C, T = 8)

with torch.no_grad():
    h = x
    for conv in convs:                                       # dilated causal conv stack
        h = torch.tanh(conv(h))
    conv_last = h[:, :, -1]                                  # (B, C) at the last position

    z = x.transpose(1, 2)                                    # hierarchical MLP: (B, T, C)
    for conv in convs:
        W = conv.conv.weight                                 # (C_out, C_in, 2)
        W_lin = torch.cat([W[:, :, 0], W[:, :, 1]], dim=1)   # (C_out, 2 C_in): [x_{t-d} ; x_t]
        z = torch.tanh(flatten_consecutive(z) @ W_lin.T + conv.conv.bias)
    tree_out = z[:, 0, :]                                    # (B, C)

print("tree shapes: (3, 8, 8) -> (3, 4, 8) -> (3, 2, 8) -> (3, 1, 8)")
print("max |conv stack (last position) - hierarchical MLP| =", (conv_last - tree_out).abs().max().item())
assert torch.allclose(conv_last, tree_out, atol=1e-5)

# %% [markdown]
# ## 8. Validation loss versus receptive field
#
# How much does seeing further back help? We train WaveNets with
# $L = 0,1,2,3,4$ levels — receptive fields $1, 2, 4, 8, 16$ — for the same, short
# budget. With $L=0$ the model sees only the current character: it is a (neural)
# **bigram** model, and should land near LM 1's 2.45 nats.

# %%
sweep = {}
t0 = time.perf_counter()
for n_levels in range(5):
    torch.manual_seed(0)
    m = WaveNetLM(V, n_levels=n_levels).to(device)
    h = train(m, steps=1000, eval_every=1000, verbose=False)
    sweep[m.receptive_field] = h["val"][-1]
    print(f"levels {n_levels} | receptive field {m.receptive_field:2d} | "
          f"params {sum(p.numel() for p in m.parameters()):6d} | val NLL {h['val'][-1]:.3f}")
print(f"sweep took {time.perf_counter() - t0:.1f} s")

# %%
rf = list(sweep)
plt.plot(rf, [sweep[r] for r in rf], "o-", label="WaveNet, 1 000 steps")
plt.axhline(2.45, color="gray", ls=":", label="bigram (LM 1)")
plt.axhline(hist["val"][-1], color="C1", ls="--", label="WaveNet RF 16, 3 000 steps")
plt.xscale("log", base=2); plt.xticks(rf, [str(r) for r in rf])
plt.xlabel("receptive field (characters)"); plt.ylabel("val NLL (nats / char)")
plt.title("Seeing further back"); plt.legend(); plt.show()

# %% [markdown]
# The biggest gains come from the first few characters of context; beyond about
# eight characters there is little left to gain for names, which are short.
# For long documents the story is very different — and the cost of reaching far
# back with convolutions (one more layer per doubling, fixed maximum context)
# is one reason attention took over.
#
# Final numbers for the main model:

# %%
val_nll = evaluate(model, Xva, Yva)
print(f"test NLL (nats/char): {evaluate(model, Xte, Yte):.3f}")
print(f"val perplexity: {math.exp(val_nll):.2f}")
print(f"val NLL (nats/char): {val_nll:.3f}")
assert val_nll < 2.6

# %% [markdown]
# ## From here to LLMs
#
# * **Teacher forcing at every position.** We fed the true prefix and scored the
#   prediction at every position of every sequence in one forward pass. GPT
#   pre-training is precisely this, with inputs `tokens[:-1]` and targets
#   `tokens[1:]`. (In LLM code `ignore_index=-100`, PyTorch's default, is the
#   conventional "do not train on this token" label — used for padding, and in
#   instruction tuning to skip the loss on the prompt.)
# * **Causality by construction.** Left padding made our convolution causal; in a
#   Transformer a triangular **attention mask** does the same job. The
#   perturbation test we used is a standard unit test for LLM implementations.
# * **Receptive field ↔ context length.** Our model cannot use information more
#   than 16 characters back, however long the input. A Transformer's "receptive
#   field" is its context window (thousands to millions of tokens), reached in a
#   single layer by attention.
# * **Convolutions are not gone.** Modern efficient sequence models — H3, Hyena,
#   Mamba and other state-space models — contain short causal 1-D convolutions,
#   and several of them are long convolutions in disguise. Gated units like
#   $\tanh\odot\sigma$ live on as the **GLU/SwiGLU** feed-forward layers of Llama-style
#   Transformers.
# * **Caching for generation.** Sampling recomputed the whole prefix each step; real
#   systems cache past activations — the KV cache is the Transformer version.
#
# ## Try it yourself
#
# 1. Replace the gated unit by a plain `ReLU` (one convolution per block). Is gating
#    worth the extra parameters?
# 2. Use kernel size 3 with dilations 1, 3, 9. What is the receptive field? Compare
#    the validation loss at a similar parameter count.
# 3. Remove the residual connection (`return self.proj(h)`). Train 4 and 8 levels.
#    What happens to the deeper network?
# 4. Break causality on purpose: use `nn.Conv1d(..., padding=1)` with kernel size 3.
#    Watch the *training* loss — and then look at the samples.
# 5. Add WaveNet's *skip connections*: sum a $1\times1$-projected output of every
#    block and feed that to the head.
# 6. Implement cached generation: keep, for each layer, the last $d_l$ inputs it
#    needs, so each new character costs one step per layer instead of a full pass.
#
# **Further reading.**
# van den Oord et al., *WaveNet: A Generative Model for Raw Audio*, 2016;
# Yu & Koltun, *Multi-Scale Context Aggregation by Dilated Convolutions*, ICLR 2016;
# Kalchbrenner et al., *Neural Machine Translation in Linear Time* (ByteNet), 2016;
# Dauphin et al., *Language Modeling with Gated Convolutional Networks*, ICML 2017;
# Bai, Kolter & Koltun, *An Empirical Evaluation of Generic Convolutional and Recurrent
# Networks for Sequence Modeling* (TCN), 2018;
# Karpathy, "Building makemore Part 5: Building a WaveNet" (Neural Networks: Zero to Hero);
# Poli et al., *Hyena Hierarchy*, ICML 2023.
