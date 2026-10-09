# %% [markdown]
# # Lecture 4 — From NumPy to PyTorch
#
# **Neural Networks · Master's course**
#
# In Lectures 1–3 we wrote everything by hand: the forward pass, the backward
# pass, the parameter updates and the initialisation. That was deliberate — you
# now know exactly what a deep learning framework has to do. From this lecture
# on we let **PyTorch** do the bookkeeping, so we can focus on models and ideas.
#
# PyTorch gives us four things that NumPy does not:
#
# 1. **Tensors** — n-dimensional arrays (almost identical to NumPy arrays) that
#    can live on a GPU;
# 2. **autograd** — automatic differentiation that records the operations of the
#    forward pass and runs backpropagation for us;
# 3. **`torch.nn`** — building blocks (layers, losses, containers) that manage
#    parameters for us;
# 4. **`torch.optim` and `torch.utils.data`** — optimisers and data pipelines.
#
# By the end of the lecture we will have written the *canonical training loop*
# that every later lecture reuses, and trained an MLP on the digits dataset.
#
# **Plan.**
# 1. Tensors: creation, dtype, device, broadcasting, views vs copies, NumPy interop
# 2. Autograd: the dynamic computational graph
# 3. Autograd vs. the manual backprop of Lecture 2
# 4. `nn.Module`, `nn.Parameter`, `nn.Sequential`
# 5. Loss functions and optimisers
# 6. `Dataset` and `DataLoader`
# 7. The training loop, train/eval modes, validation
# 8. Saving and loading models

# %%
import os
import tempfile

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.manual_seed(0)
np.random.seed(0)
plt.rcParams["figure.figsize"] = (5, 4)
plt.rcParams["axes.grid"] = True

device = "cuda" if torch.cuda.is_available() else "cpu"
print("PyTorch", torch.__version__, "| device:", device)

# %% [markdown]
# ## 1. Tensors
#
# A `torch.Tensor` is an n-dimensional array with a **shape**, a **dtype** and a
# **device**. Most of the NumPy API carries over with small spelling changes
# (`axis` → `dim`, `np.concatenate` → `torch.cat`, `astype` → `.to(dtype)`).
#
# ### 1.1 Creation

# %%
a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])   # from Python data
z = torch.zeros(2, 3)
o = torch.ones_like(a)
r = torch.randn(2, 3)                 # standard normal
u = torch.rand(2, 3)                  # uniform on [0, 1)
i = torch.arange(6)                   # 0, 1, ..., 5  (integer dtype!)
l = torch.linspace(0, 1, 5)
e = torch.eye(3)

for name, t in [("a", a), ("z", z), ("r", r), ("i", i), ("l", l), ("e", e)]:
    print(f"{name}: shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}")

# %% [markdown]
# ### 1.2 dtype: the float32 convention
#
# Deep learning almost always uses **32-bit floats**. PyTorch's default floating
# dtype is `torch.float32`, whereas NumPy's default is `float64`. This is the
# most common source of confusing errors when moving from NumPy to PyTorch:
# a layer has `float32` weights, and feeding it a `float64` tensor raises
# `RuntimeError: mat1 and mat2 must have the same dtype`.
#
# Class labels for classification losses must be **integers** of type
# `torch.int64` (a.k.a. `torch.long`).

# %%
x_np = np.random.rand(3, 2)                     # float64 in NumPy
x_t = torch.from_numpy(x_np)
print("from NumPy:", x_t.dtype)                 # stays float64!
print("converted :", x_t.float().dtype, "|", x_t.to(torch.float32).dtype)
print("default   :", torch.get_default_dtype())

layer = nn.Linear(2, 1)
try:
    layer(x_t)                                  # float64 input, float32 weights
except RuntimeError as err:
    print("RuntimeError:", str(err).splitlines()[0])
print("works after .float():", layer(x_t.float()).shape)

# %% [markdown]
# ### 1.3 Device
#
# Tensors live on a device: the CPU or an accelerator such as a CUDA GPU. All
# operands of an operation must be on the same device; moving data is explicit
# with `.to(device)`. The idiom below makes code run on a GPU when one is
# available and on the CPU otherwise (e.g. in our CI or on a laptop). On Colab
# you can switch to a GPU runtime and the same notebook will use it.
#
# ```python
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = model.to(device)          # moves all parameters
# xb, yb = xb.to(device), yb.to(device)
# ```

# %%
t = torch.randn(3, device=device)
print(t.device, "->", t.cpu().device)

# %% [markdown]
# ### 1.4 Broadcasting
#
# Broadcasting follows exactly the NumPy rules: align shapes from the right;
# two dimensions are compatible if they are equal or one of them is 1; size-1
# dimensions are (virtually) repeated. A classic use: pairwise squared distances
# between $N$ points and $M$ centres without a Python loop,
#
# $$D_{ij} = \lVert \mathbf{x}_i - \mathbf{c}_j \rVert^2, \qquad
# \underbrace{(N,1,d)}_{\mathbf{x}} - \underbrace{(1,M,d)}_{\mathbf{c}} \to (N,M,d).$$

# %%
X_pts = torch.randn(5, 2)
C = torch.randn(3, 2)
D = ((X_pts[:, None, :] - C[None, :, :]) ** 2).sum(dim=-1)   # (5, 3)
print(D.shape)
print("matches torch.cdist:", torch.allclose(D, torch.cdist(X_pts, C) ** 2, atol=1e-5))

# %% [markdown]
# ### 1.5 Views versus copies
#
# Many operations (`view`, slicing, `transpose`, `t`, `expand`, `unsqueeze`) return
# a **view**: a new tensor object that shares memory with the original.
# Writing through a view changes the original. `clone()` makes a real copy.
#
# A view may be **non-contiguous**: after `transpose`, the elements are no longer
# laid out row by row in memory. `view` requires a compatible (e.g. contiguous)
# layout and fails otherwise; `reshape` returns a view when it can and a copy
# when it must.

# %%
a = torch.arange(6)
b = a.view(2, 3)            # view: shares memory with a
c = a.clone()               # copy
b[0, 0] = 100
print("a after writing into b:", a.tolist())
print("c is unaffected       :", c.tolist())

bt = b.t()                  # transpose -> non-contiguous view
print("bt contiguous?", bt.is_contiguous())
try:
    bt.view(6)
except RuntimeError as err:
    print("view() failed:", str(err)[:60], "...")
print("reshape works:", bt.reshape(6).tolist())
print("so does .contiguous().view:", bt.contiguous().view(6).tolist())

# %% [markdown]
# ### 1.6 NumPy interoperability
#
# * `torch.from_numpy(arr)` and `tensor.numpy()` **share memory** (CPU only) — no copy.
# * `torch.tensor(arr)` always **copies**.
# * A tensor that is part of an autograd graph must be detached first, and a GPU
#   tensor moved to the CPU: `tensor.detach().cpu().numpy()`.

# %%
arr = np.zeros(3, dtype=np.float32)
shared = torch.from_numpy(arr)
copied = torch.tensor(arr)
arr[0] = 7.0
print("shared:", shared.tolist(), "| copied:", copied.tolist())

w = torch.ones(2, requires_grad=True)
print((w * 2).detach().cpu().numpy())

# %% [markdown]
# ## 2. Autograd: automatic differentiation
#
# In Lecture 2 every layer implemented a `forward` and a `backward` method and we
# chained the backward calls in reverse order by hand. PyTorch automates this
# **reverse-mode automatic differentiation**:
#
# * a tensor created with `requires_grad=True` is a **leaf** whose gradient we want;
# * every operation on such tensors is recorded in a **computational graph**
#   (each result stores a `grad_fn` that knows its local derivative);
# * calling `.backward()` on a scalar walks the graph from the output back to the
#   leaves, applying the chain rule, and **accumulates** $\partial L/\partial
#   \theta$ into `theta.grad`.
#
# Consider $y = x^3 + 2x$. Then $dy/dx = 3x^2 + 2$, which is $14$ at $x=2$.

# %%
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3 + 2 * x
print("y =", y.item(), "| grad_fn:", y.grad_fn)
y.backward()
print("dy/dx =", x.grad.item())

# %% [markdown]
# ### 2.1 The graph is dynamic
#
# PyTorch builds the graph **as the code runs** ("define-by-run"). Ordinary
# Python control flow — `if`, `for`, recursion — just works, and the graph can
# be different at every iteration. Here we can peek at the recorded nodes:

# %%
x = torch.tensor(1.5, requires_grad=True)
h = x
n_steps = 0
while h.abs() < 10:          # data-dependent loop length
    h = h * 2 + 1
    n_steps += 1
print(f"loop ran {n_steps} times, h = {h.item():.1f}")
h.backward()
print("dh/dx =", x.grad.item(), "= 2 **", n_steps)

node = h.grad_fn
chain = []
while node is not None:
    chain.append(type(node).__name__)
    node = node.next_functions[0][0] if node.next_functions else None
print(" <- ".join(chain))

# %% [markdown]
# ### 2.2 Gradients accumulate
#
# `.backward()` **adds** to `.grad`; it does not overwrite it. This is a feature
# (it lets us sum gradients over several mini-batches, e.g. to simulate a larger
# batch), but it means we must reset the gradients before every optimisation
# step — that is what `optimizer.zero_grad()` does.

# %%
x = torch.tensor(2.0, requires_grad=True)
for k in range(3):
    y = x ** 3 + 2 * x
    y.backward()
    print(f"after backward #{k + 1}: x.grad = {x.grad.item()}")
x.grad.zero_()                                    # or: x.grad = None
print("after zero_():", x.grad.item())

# %% [markdown]
# Two more rules:
#
# * `.backward()` without arguments requires a **scalar** output (a loss). For a
#   vector output you must say which vector–Jacobian product you want, e.g.
#   `y.sum().backward()` or `y.backward(gradient=v)`.
# * By default the graph is **freed** after `.backward()` to save memory; a second
#   backward through the same graph raises an error unless you pass
#   `retain_graph=True`. In a training loop we rebuild the graph in every
#   forward pass anyway.

# %% [markdown]
# ### 2.3 Switching autograd off: `torch.no_grad()` and `detach()`
#
# Recording the graph costs memory and time. When we only *use* a model —
# evaluation, inference, or a manual parameter update — we wrap the code in
# `with torch.no_grad():`. Results computed inside do not require gradients.
#
# `t.detach()` returns a view of `t` that is cut off from the graph: gradients
# will not flow through it. Use it to treat a quantity as a constant (e.g. a
# target value) or to convert to NumPy.

# %%
w = torch.randn(3, requires_grad=True)
inp = torch.randn(3)
out = (w * inp).sum()
print("tracked:", out.requires_grad)
with torch.no_grad():
    out2 = (w * inp).sum()
print("inside no_grad:", out2.requires_grad)
print("detached:", out.detach().requires_grad)

# A manual gradient-descent step must not itself be recorded in the graph:
out.backward()
with torch.no_grad():
    w -= 0.1 * w.grad
w.grad = None
print("w is still a leaf requiring grad:", w.is_leaf, w.requires_grad)

# %% [markdown]
# ## 3. Autograd versus manual backprop
#
# Let us make the connection to Lecture 2 explicit. A two-layer MLP for $K$
# classes, on a mini-batch $X\in\mathbb{R}^{N\times D}$ with integer labels $y$:
#
# $$Z_1 = XW_1 + \mathbf b_1,\quad A_1 = \operatorname{ReLU}(Z_1),\quad
# Z_2 = A_1W_2 + \mathbf b_2,\quad
# \mathcal L = -\frac1N\sum_i \log \operatorname{softmax}(Z_2)_{i,y_i}.$$
#
# The backward pass of Lecture 2, written with $P=\operatorname{softmax}(Z_2)$ and
# one-hot targets $Y$:
#
# $$\begin{aligned}
# \frac{\partial\mathcal L}{\partial Z_2} &= \tfrac1N (P - Y), &
# \frac{\partial\mathcal L}{\partial W_2} &= A_1^\top \frac{\partial\mathcal L}{\partial Z_2}, &
# \frac{\partial\mathcal L}{\partial \mathbf b_2} &= \textstyle\sum_i \big(\frac{\partial\mathcal L}{\partial Z_2}\big)_{i,:},\\
# \frac{\partial\mathcal L}{\partial A_1} &= \frac{\partial\mathcal L}{\partial Z_2} W_2^\top, &
# \frac{\partial\mathcal L}{\partial Z_1} &= \frac{\partial\mathcal L}{\partial A_1}\odot \mathbb 1[Z_1>0], &
# \frac{\partial\mathcal L}{\partial W_1} &= X^\top \frac{\partial\mathcal L}{\partial Z_1}.
# \end{aligned}$$
#
# We compute these by hand and compare with autograd. We use `float64` so that
# the comparison is exact up to round-off.

# %%
torch.manual_seed(0)
N, D, H, K = 16, 64, 32, 10
X = torch.randn(N, D, dtype=torch.float64)
y = torch.randint(0, K, (N,))
W1 = (0.1 * torch.randn(D, H, dtype=torch.float64)).requires_grad_()
b1 = torch.zeros(H, dtype=torch.float64, requires_grad=True)
W2 = (0.1 * torch.randn(H, K, dtype=torch.float64)).requires_grad_()
b2 = torch.zeros(K, dtype=torch.float64, requires_grad=True)

# forward pass (recorded by autograd)
Z1 = X @ W1 + b1
A1 = torch.relu(Z1)
Z2 = A1 @ W2 + b2
loss = F.cross_entropy(Z2, y)          # softmax + negative log-likelihood, averaged
loss.backward()                        # autograd fills W1.grad, b1.grad, W2.grad, b2.grad

# manual backward pass (Lecture 2), no graph needed
with torch.no_grad():
    P = torch.softmax(Z2, dim=1)
    Y = F.one_hot(y, K).to(P.dtype)
    dZ2 = (P - Y) / N
    dW2 = A1.T @ dZ2
    db2 = dZ2.sum(dim=0)
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (Z1 > 0).to(dA1.dtype)
    dW1 = X.T @ dZ1
    db1 = dZ1.sum(dim=0)

for name, manual, param in [("W1", dW1, W1), ("b1", db1, b1), ("W2", dW2, W2), ("b2", db2, b2)]:
    err = (manual - param.grad).abs().max().item()
    print(f"{name}: max |manual - autograd| = {err:.2e}")
    assert torch.allclose(manual, param.grad, atol=1e-10)
print("Autograd reproduces our hand-written backprop.")

# %% [markdown]
# Autograd does *exactly* what we did in Lecture 2 — it just keeps track of the
# intermediate values ($Z_1$, $A_1$, …) and the order of operations for us. The
# price is memory: every intermediate needed for the backward pass is stored
# until `.backward()` runs. This is why training needs much more memory than
# inference, and why `torch.no_grad()` matters at evaluation time.

# %% [markdown]
# ## 4. `nn.Module`: models as objects
#
# Writing `W1, b1, W2, b2` by hand does not scale. `torch.nn.Module` is the base
# class for every layer and model. A module
#
# * registers every `nn.Parameter` (and every sub-module) assigned as an
#   attribute in `__init__`, so `model.parameters()` finds them recursively;
# * defines the computation in `forward`; we call the module like a function,
#   `model(x)`, which runs `forward` (plus any hooks);
# * can be moved to a device with `.to(device)` and switched between training
#   and evaluation behaviour with `.train()` / `.eval()`;
# * exposes its state as a `state_dict()` for saving and loading.
#
# Note the convention of `nn.Linear(in_features, out_features)`: its weight has
# shape `(out, in)` and it computes $\mathbf y = \mathbf x W^\top + \mathbf b$.

# %%
class TwoLayerMLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()                      # always call this first
        self.fc1 = nn.Linear(d_in, d_hidden)    # sub-modules are registered automatically
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


model = TwoLayerMLP(64, 32, 10)
print(model)
for name, p in model.named_parameters():
    print(f"{name:12s} {tuple(p.shape)}")
print("number of parameters:", sum(p.numel() for p in model.parameters()))

# %% [markdown]
# A plain tensor assigned as an attribute is **not** a parameter — the optimiser
# would never see it. Wrap learnable tensors in `nn.Parameter`. (For non-learnable
# state that should still be saved and moved with the model — e.g. running
# statistics in batch normalisation, Lecture 5 — use `self.register_buffer`.)

# %%
class Scale(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(d))   # learnable, registered
        self.not_registered = torch.ones(d)       # invisible to .parameters()!

    def forward(self, x):
        return x * self.gain

print([name for name, _ in Scale(4).named_parameters()])

# %% [markdown]
# For simple feed-forward stacks, `nn.Sequential` saves us from writing a class.
# Other useful containers are `nn.ModuleList` and `nn.ModuleDict` (a plain Python
# list of modules would *not* register their parameters).

# %%
seq = nn.Sequential(
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 10),
)
print(seq)
print("output shape:", seq(torch.randn(5, 64)).shape)

# %% [markdown]
# ## 5. Loss functions and optimisers
#
# ### 5.1 Losses
#
# | Task | Model output | Loss | Target |
# |---|---|---|---|
# | regression | real values | `nn.MSELoss` | float, same shape |
# | binary classification | one **logit** per example | `nn.BCEWithLogitsLoss` | float in $\{0,1\}$ |
# | $K$-class classification | $K$ **logits** per example | `nn.CrossEntropyLoss` | `long` in $\{0,\dots,K-1\}$ |
#
# **`CrossEntropyLoss` takes logits, not probabilities.** It applies
# `log_softmax` internally (in a numerically stable way, using the log-sum-exp
# trick of Lecture 1) and then the negative log-likelihood:
#
# $$\operatorname{CE}(\mathbf z, y) = -z_y + \log\sum_k e^{z_k}.$$
#
# A very common bug is to put an `nn.Softmax` at the end of the model *and* use
# `CrossEntropyLoss`. Nothing crashes, but softmax is applied twice: the loss can
# never get close to zero and the gradients become small.

# %%
logits = torch.tensor([[4.0, 0.0, -2.0], [0.5, 2.5, 0.0]], requires_grad=True)
target = torch.tensor([0, 1])

ce = F.cross_entropy(logits, target)
manual = F.nll_loss(F.log_softmax(logits, dim=1), target)
print(f"cross_entropy = {ce.item():.4f} | log_softmax + nll = {manual.item():.4f}")

wrong = F.cross_entropy(torch.softmax(logits, dim=1), target)   # the double-softmax bug
print(f"with an extra softmax: {wrong.item():.4f}  (minimum possible ~ {np.log(1 + 2 / np.e):.3f}, not 0)")

g_right = torch.autograd.grad(ce, logits)[0]
g_wrong = torch.autograd.grad(wrong, logits)[0]
print(f"gradient norm: correct {g_right.norm().item():.4f} vs double softmax {g_wrong.norm().item():.4f}")

# %% [markdown]
# Losses come in two flavours with identical maths: classes (`nn.CrossEntropyLoss()`,
# convenient inside `nn.Sequential` or as a configurable object) and functions
# (`F.cross_entropy`). By default they **average** over the batch
# (`reduction="mean"`).
#
# ### 5.2 Optimisers
#
# `torch.optim` implements the update rules of Lecture 3 (SGD, momentum, Adam, …).
# An optimiser is constructed from an iterable of parameters and has two methods:
#
# * `optimizer.zero_grad()` — reset `.grad` of all its parameters (to `None` by default);
# * `optimizer.step()` — apply one update using the current `.grad`.
#
# Let us check that one step of `optim.SGD` is literally
# $\theta \leftarrow \theta - \eta\,\nabla_\theta\mathcal L$:

# %%
torch.manual_seed(0)
lin = nn.Linear(3, 1)
xb, yb = torch.randn(8, 3), torch.randn(8, 1)
opt = torch.optim.SGD(lin.parameters(), lr=0.1)

before = [p.detach().clone() for p in lin.parameters()]
opt.zero_grad()
F.mse_loss(lin(xb), yb).backward()
expected = [p0 - 0.1 * p.grad for p0, p in zip(before, lin.parameters())]
opt.step()
print("SGD step matches the manual update:",
      all(torch.allclose(p, e) for p, e in zip(lin.parameters(), expected)))

# %% [markdown]
# Momentum and Adam are one argument away:
# `torch.optim.SGD(params, lr=0.1, momentum=0.9)`, `torch.optim.Adam(params, lr=1e-3)`.
# Learning-rate schedules live in `torch.optim.lr_scheduler`.
#
# ## 6. `Dataset` and `DataLoader`
#
# A **`Dataset`** is anything with `__len__` and `__getitem__(i)` returning one
# example. A **`DataLoader`** wraps a dataset and produces mini-batches: it
# shuffles the indices each epoch (if `shuffle=True`), fetches the examples, and
# *collates* them into batched tensors (stacking along a new first dimension).
# For data already in memory as tensors, `TensorDataset` does the job.
#
# We now load the digits (8×8 images, 10 classes) and split them into training,
# validation and test sets. The validation set is used to monitor training and to
# choose hyper-parameters; the test set is touched once, at the very end.

# %%
digits = load_digits()
X_all = (digits.data / 16.0).astype(np.float32)     # float32 for PyTorch
y_all = digits.target.astype(np.int64)              # int64 labels for cross-entropy

X_trval, X_test, y_trval, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=0, stratify=y_all)
X_train, X_val, y_train, y_val = train_test_split(
    X_trval, y_trval, test_size=0.2, random_state=0, stratify=y_trval)
print("train", X_train.shape, "| val", X_val.shape, "| test", X_test.shape)

train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

x0, y0 = train_ds[0]
print("one example:", x0.shape, x0.dtype, "| label:", y0.item(), y0.dtype)

# %% [markdown]
# Writing your own `Dataset` is just as easy, and is the place to put
# per-example preprocessing or data augmentation (Lecture 5):

# %%
class DigitsImages(Dataset):
    """Digits as 1x8x8 images (channel-first, as convolutional layers expect)."""

    def __init__(self, X, y):
        self.X = torch.as_tensor(X, dtype=torch.float32).reshape(-1, 1, 8, 8)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

img_ds = DigitsImages(X_train, y_train)
print(len(img_ds), img_ds[0][0].shape)

# %%
g = torch.Generator().manual_seed(0)          # reproducible shuffling
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, generator=g)
val_loader = DataLoader(val_ds, batch_size=256)
test_loader = DataLoader(test_ds, batch_size=256)

xb, yb = next(iter(train_loader))
print("a batch:", xb.shape, yb.shape, "| batches per epoch:", len(train_loader))

fig, axes = plt.subplots(1, 8, figsize=(8, 1.3))
for k, ax in enumerate(axes):
    ax.imshow(xb[k].reshape(8, 8).numpy(), cmap="gray_r"); ax.set_title(yb[k].item()); ax.axis("off")
plt.show()

# %% [markdown]
# ## 7. The canonical training loop
#
# Every supervised training script in PyTorch has the same skeleton:
#
# ```text
# for epoch in range(n_epochs):
#     model.train()
#     for xb, yb in train_loader:
#         xb, yb = xb.to(device), yb.to(device)
#         loss = loss_fn(model(xb), yb)      # 1. forward
#         optimizer.zero_grad()              # 2. reset gradients
#         loss.backward()                    # 3. backward
#         optimizer.step()                   # 4. update
#     model.eval()
#     with torch.no_grad():
#         ... evaluate on the validation set ...
# ```
#
# **Train vs. eval mode.** `model.train()` and `model.eval()` set the flag
# `module.training` on every sub-module. Some layers behave differently in the two
# modes — dropout and batch normalisation (Lecture 5). For our plain MLP the mode
# makes no difference, but setting it correctly is a habit worth forming now.
# Note that `model.eval()` does **not** switch off autograd; that is the job of
# `torch.no_grad()`. We use both when evaluating.

# %%
def train_one_epoch(model, loader, loss_fn, optimizer):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(yb)          # .item(): tensor -> Python float
        correct += (logits.argmax(dim=1) == yb).sum().item()
        n += len(yb)
    return total_loss / n, correct / n


@torch.no_grad()                                     # decorator form of `with torch.no_grad()`
def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        total_loss += loss_fn(logits, yb).item() * len(yb)
        correct += (logits.argmax(dim=1) == yb).sum().item()
        n += len(yb)
    return total_loss / n, correct / n

# %% [markdown]
# Why `loss.item() * len(yb)`? The loss is averaged within a batch, and the last
# batch of an epoch may be smaller; weighting by batch size gives the exact
# per-example average over the epoch. Accumulating `loss` itself (without
# `.item()`) would keep every batch's graph alive — a classic memory leak.
#
# Now we train a one-hidden-layer MLP with Adam.

# %%
torch.manual_seed(0)

def make_mlp():
    return nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )

model = make_mlp().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
n_epochs = 40
for epoch in range(n_epochs):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, loss_fn, optimizer)
    va_loss, va_acc = evaluate(model, val_loader, loss_fn)
    for key, val in zip(history, (tr_loss, tr_acc, va_loss, va_acc)):
        history[key].append(val)
    if epoch % 10 == 0 or epoch == n_epochs - 1:
        print(f"epoch {epoch:3d} | train loss {tr_loss:.3f} acc {tr_acc:.3f} "
              f"| val loss {va_loss:.3f} acc {va_acc:.3f}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].plot(history["train_loss"], label="train"); axes[0].plot(history["val_loss"], label="validation")
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("cross-entropy"); axes[0].legend(); axes[0].set_title("Loss")
axes[1].plot(history["train_acc"], label="train"); axes[1].plot(history["val_acc"], label="validation")
axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy"); axes[1].legend(); axes[1].set_title("Accuracy")
plt.tight_layout(); plt.show()

# %% [markdown]
# Only now, with all choices made, do we look at the test set:

# %%
test_loss, test_acc = evaluate(model, test_loader, loss_fn)
print(f"test accuracy: {test_acc:.3f}")
assert test_acc >= 0.90

# %% [markdown]
# Compare with softmax regression in Lecture 1: on this easy dataset the hidden
# layer buys us only a modest improvement — the linear model is already good. To get predictions for new inputs,
# run the model in eval mode without gradients and convert logits to
# probabilities with `softmax` only if you actually need probabilities:

# %%
model.eval()
with torch.no_grad():
    probs = torch.softmax(model(torch.from_numpy(X_test[:5]).to(device)), dim=1).cpu()
print("predicted:", probs.argmax(dim=1).tolist(), "| true:", y_test[:5].tolist())
print("max probability:", [round(p, 3) for p in probs.max(dim=1).values.tolist()])

# %% [markdown]
# ## 8. Saving and loading
#
# The recommended way to save a model is its **`state_dict`**: an ordered
# dictionary mapping parameter (and buffer) names to tensors. To load it, build
# the same architecture in code and call `load_state_dict`. (Saving the whole
# model object with pickle also works but ties the file to your exact source code.)
# Optimisers have a `state_dict` too — save it if you want to resume training.

# %%
print({k: tuple(v.shape) for k, v in model.state_dict().items()})

with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "digits_mlp.pt")
    torch.save(model.state_dict(), path)
    print(f"saved {os.path.getsize(path) / 1024:.1f} kB")

    restored = make_mlp().to(device)
    restored.load_state_dict(torch.load(path, map_location=device, weights_only=True))

print("restored model, test accuracy:", round(evaluate(restored, test_loader, loss_fn)[1], 3))

# %% [markdown]
# `map_location` lets you load a model trained on a GPU on a CPU-only machine;
# `weights_only=True` restricts unpickling to tensors and simple containers, which
# is safer for files from untrusted sources.
#
# ## Summary
#
# | NumPy (Lectures 1–3) | PyTorch |
# |---|---|
# | `np.ndarray`, float64 | `torch.Tensor`, float32 by default, on a `device` |
# | hand-written `backward` | `loss.backward()` (autograd, dynamic graph) |
# | parameters in variables | `nn.Parameter` registered inside an `nn.Module` |
# | `W -= lr * dW` | `optimizer.zero_grad(); loss.backward(); optimizer.step()` |
# | index shuffling by hand | `Dataset` + `DataLoader` |
# | softmax + cross-entropy | `nn.CrossEntropyLoss` on **logits** |
#
# Checklist for every training loop: data and model on the same device and with
# float32 inputs / int64 labels; `zero_grad` before `backward`; `model.train()`
# for training, `model.eval()` + `torch.no_grad()` for evaluation; `.item()` when
# logging; select hyper-parameters on validation data, report on test data.
#
# **Further reading.**
# Paszke et al., *PyTorch: An Imperative Style, High-Performance Deep Learning
# Library*, NeurIPS 2019;
# Baydin et al., *Automatic Differentiation in Machine Learning: a Survey*, JMLR 2018;
# the official PyTorch tutorials "Learn the Basics" and "Autograd mechanics"
# (pytorch.org/tutorials, pytorch.org/docs/stable/notes/autograd.html);
# Zhang et al., *Dive into Deep Learning*, ch. 2 and 5–6.
