# %% [markdown]
# # Exercises 6 — Convolutional Neural Networks
#
# Fill in every cell marked `# YOUR CODE HERE` and answer the written questions.
# The `assert` cells check your work: if a cell runs without an error, you are
# probably on the right track.

# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

# %% [markdown]
# ## Exercise 6.1 — Convolution arithmetic
#
# Implement `conv_output_size(n, k, s=1, p=0, d=1)`, the output size of a
# convolution (or pooling) along one dimension with input size `n`, kernel size
# `k`, stride `s`, padding `p` and dilation `d`:
#
# $$n_{\text{out}} = \left\lfloor \frac{n + 2p - d(k-1) - 1}{s} \right\rfloor + 1 .$$
#
# Then use it to fill in the spatial size after each layer of the network below.

# %%
def conv_output_size(n, k, s=1, p=0, d=1):
    ### BEGIN SOLUTION
    return (n + 2 * p - d * (k - 1) - 1) // s + 1
    ### END SOLUTION

# %%
for (n, k, s, p, d) in [(28, 3, 1, 0, 1), (28, 3, 1, 1, 1), (28, 5, 2, 2, 1), (32, 4, 2, 1, 1),
                        (31, 3, 2, 0, 1), (28, 3, 1, 2, 2), (224, 7, 2, 3, 1), (7, 7, 1, 0, 1)]:
    expected = nn.Conv2d(1, 1, k, stride=s, padding=p, dilation=d)(torch.zeros(1, 1, n, n)).shape[-1]
    assert conv_output_size(n, k, s, p, d) == expected, (n, k, s, p, d)
print("conv_output_size OK")

# %% [markdown]
# A network for $32\times32$ inputs: conv 5×5 (stride 1, no padding) → max-pool 2×2
# (stride 2) → conv 3×3 (stride 1, padding 1) → conv 3×3 (stride 2, padding 1) →
# max-pool 2×2 (stride 2). Use `conv_output_size` to compute the spatial size after
# each layer and store them in the list `sizes` (5 numbers).

# %%
### BEGIN SOLUTION
sizes = []
n = 32
for k, s, p in [(5, 1, 0), (2, 2, 0), (3, 1, 1), (3, 2, 1), (2, 2, 0)]:
    n = conv_output_size(n, k, s, p)
    sizes.append(n)
### END SOLUTION
print(sizes)

# %%
net = nn.Sequential(nn.Conv2d(3, 8, 5), nn.MaxPool2d(2), nn.Conv2d(8, 8, 3, padding=1),
                    nn.Conv2d(8, 8, 3, stride=2, padding=1), nn.MaxPool2d(2))
x = torch.zeros(1, 3, 32, 32)
true_sizes = []
for layer in net:
    x = layer(x); true_sizes.append(x.shape[-1])
assert sizes == true_sizes, true_sizes
print("sizes OK")

# %% [markdown]
# ## Exercise 6.2 — Convolution as a matrix multiplication (im2col)
#
# The loop implementation in the lecture is far too slow for real use. Libraries
# instead rearrange all input patches into the columns of a matrix ("im2col") and
# perform **one** big matrix multiplication. PyTorch exposes the rearrangement as
# `F.unfold`: for an input of shape `(N, C, H, W)` it returns a tensor of shape
# `(N, C·kh·kw, L)`, where `L = H_out·W_out` is the number of patch positions and each
# column is one flattened patch (in the same `(c, u, v)` order as the weight tensor).
#
# Implement `conv2d_im2col(x, weight, bias, stride, padding)` using `F.unfold`, a
# matrix multiplication with the reshaped weight `(C_out, C_in·kh·kw)`, the bias, and a
# final reshape to `(N, C_out, H_out, W_out)`. Do not call `F.conv2d`.

# %%
def conv2d_im2col(x, weight, bias=None, stride=1, padding=0):
    N, C_in, H, W = x.shape
    C_out, _, kh, kw = weight.shape
    ### BEGIN SOLUTION
    H_out = (H + 2 * padding - kh) // stride + 1
    W_out = (W + 2 * padding - kw) // stride + 1
    cols = F.unfold(x, kernel_size=(kh, kw), stride=stride, padding=padding)   # (N, C_in*kh*kw, L)
    out = weight.reshape(C_out, -1) @ cols                                       # (N, C_out, L)
    if bias is not None:
        out = out + bias[None, :, None]
    return out.reshape(N, C_out, H_out, W_out)
    ### END SOLUTION

# %%
torch.manual_seed(0)
for (N, C_in, H, W, C_out, k, s, p) in [(2, 3, 8, 8, 4, 3, 1, 1), (1, 1, 7, 9, 2, 3, 2, 0),
                                         (3, 5, 10, 6, 7, 5, 1, 2), (2, 2, 11, 11, 3, 4, 3, 1)]:
    x = torch.randn(N, C_in, H, W, dtype=torch.float64)
    w = torch.randn(C_out, C_in, k, k, dtype=torch.float64)
    b = torch.randn(C_out, dtype=torch.float64)
    ref = F.conv2d(x, w, b, stride=s, padding=p)
    mine = conv2d_im2col(x, w, b, stride=s, padding=p)
    assert mine.shape == ref.shape, (mine.shape, ref.shape)
    assert torch.allclose(mine, ref), (mine - ref).abs().max()
assert torch.allclose(conv2d_im2col(x, w), F.conv2d(x, w))
# it is differentiable through autograd
x.requires_grad_(); w.requires_grad_()
g_mine = torch.autograd.grad(conv2d_im2col(x, w, b, 3, 1).sum(), (x, w))
g_ref = torch.autograd.grad(F.conv2d(x, w, b, stride=3, padding=1).sum(), (x, w))
assert all(torch.allclose(a, c) for a, c in zip(g_mine, g_ref))
print("im2col convolution OK")

# %% [markdown]
# **Question.** For an input with $C_{in}=64$ channels of size $56\times56$, a
# $3\times3$ kernel, stride 1 and padding 1, how many elements does the unfolded
# matrix have per image, compared with the input? What does this say about the
# memory cost of im2col?
#
# <!-- BEGIN SOLUTION -->
# The unfolded matrix has $C_{in}k^2 \times L = 64\cdot9 \times 56\cdot56 = 576\times3136
# \approx 1.8\times10^6$ elements, versus $64\cdot56\cdot56\approx 2.0\times10^5$ in the
# input — $k^2=9$ times more, because every pixel is copied into up to nine patches.
# im2col trades memory for speed (it turns convolution into a single highly optimised
# GEMM); libraries such as cuDNN therefore also use implicit-GEMM, FFT or Winograd
# algorithms that avoid materialising the full matrix.
# <!-- END SOLUTION -->

# %% [markdown]
# ## Exercise 6.3 — Max pooling
#
# Implement `max_pool2d(x, k)` for non-overlapping $k\times k$ windows (stride $k$,
# no padding; assume $H$ and $W$ are divisible by $k$) **without** `F.max_pool2d` or
# `F.unfold`. *Hint:* reshape `(N, C, H, W)` to `(N, C, H//k, k, W//k, k)` and take the
# maximum over the two window dimensions (`amax` accepts a tuple of dims).
#
# Thanks to autograd, your function automatically has the right backward pass:
# the gradient goes to the arg-max of each window.

# %%
def max_pool2d(x, k=2):
    ### BEGIN SOLUTION
    N, C, H, W = x.shape
    return x.reshape(N, C, H // k, k, W // k, k).amax(dim=(3, 5))
    ### END SOLUTION

# %%
torch.manual_seed(0)
x = torch.randn(2, 3, 8, 12, requires_grad=True)
for k in (2, 4):
    assert torch.equal(max_pool2d(x, k), F.max_pool2d(x, k))
g_mine = torch.autograd.grad(max_pool2d(x, 2).sum(), x)[0]
g_ref = torch.autograd.grad(F.max_pool2d(x, 2).sum(), x)[0]
assert torch.equal(g_mine, g_ref)
assert g_mine.sum().item() == 2 * 3 * 4 * 6          # exactly one 1 per window
print("max_pool2d OK")

# %% [markdown]
# ## Exercise 6.4 — A residual block
#
# Implement the ResNet "basic block":
#
# $$\mathbf y = \operatorname{ReLU}\big(\operatorname{BN}_2(\operatorname{conv}_2(\operatorname{ReLU}(\operatorname{BN}_1(\operatorname{conv}_1(\mathbf x))))) + \operatorname{shortcut}(\mathbf x)\big)$$
#
# * `conv1`: $3\times3$, `c_in → c_out`, given `stride`, padding 1, no bias;
# * `conv2`: $3\times3$, `c_out → c_out`, stride 1, padding 1, no bias;
# * `bn1`, `bn2`: `nn.BatchNorm2d(c_out)`;
# * `shortcut`: `nn.Identity()` if the shape does not change, otherwise
#   `nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False), nn.BatchNorm2d(c_out))`.
#
# Use exactly these attribute names.

# %%
class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        ### BEGIN SOLUTION
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False),
                                          nn.BatchNorm2d(c_out))
        else:
            self.shortcut = nn.Identity()
        ### END SOLUTION

    def forward(self, x):
        ### BEGIN SOLUTION
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))
        ### END SOLUTION

# %%
torch.manual_seed(0)
blk = BasicBlock(8, 8)
assert isinstance(blk.shortcut, nn.Identity)
assert blk(torch.randn(2, 8, 6, 6)).shape == (2, 8, 6, 6)
blk2 = BasicBlock(8, 16, stride=2)
assert blk2(torch.randn(2, 8, 6, 6)).shape == (2, 16, 3, 3)
assert not isinstance(blk2.shortcut, nn.Identity)
assert sum(p.numel() for p in blk.parameters()) == 2 * 8 * 8 * 9 + 4 * 8
assert sum(p.numel() for p in blk2.parameters()) == 8 * 16 * 9 + 16 * 16 * 9 + 8 * 16 + 6 * 16
# With conv2 zeroed, the residual branch vanishes and the block is ReLU(x)
# (a fresh block: BN running mean 0, running var 1, gamma 1, beta 0).
blk = BasicBlock(8, 8)
with torch.no_grad():
    blk.conv2.weight.zero_()
blk.eval()
x = torch.randn(2, 8, 6, 6)
assert torch.allclose(blk(x), F.relu(x), atol=1e-6)
print("BasicBlock OK")

# %% [markdown]
# ## Exercise 6.5 — Receptive fields (pen and paper, then check)
#
# **(a)** Compute by hand the receptive field (in input pixels, along one dimension)
# of a unit in the output of each of these stacks:
#
# 1. three $3\times3$ convolutions, stride 1;
# 2. conv $3\times3$ (stride 1) → max-pool $2\times2$ (stride 2) → conv $3\times3$ (stride 1) → max-pool $2\times2$ (stride 2) → conv $3\times3$ (stride 1);
# 3. conv $7\times7$ (stride 2) → max-pool $3\times3$ (stride 2) — the ResNet stem.
#
# Store your three answers in `rf_answers`. Then check them with the autograd trick
# from the lecture (implemented for you below).
#
# <!-- BEGIN SOLUTION -->
# Using $r_L = 1 + \sum_l (k_l-1)\prod_{i<l}s_i$:
# 1. $1 + 2 + 2 + 2 = 7$;
# 2. $1 + 2\cdot1 + 1\cdot1 + 2\cdot2 + 1\cdot2 + 2\cdot4 = 18$;
# 3. $1 + 6\cdot1 + 2\cdot2 = 11$.
# <!-- END SOLUTION -->

# %%
### BEGIN SOLUTION
rf_answers = [7, 18, 11]
### END SOLUTION

# %%
def measured_rf(layers, size=64):
    """Measure the receptive field of the centre output unit with autograd.
    Max-pooling is replaced by average pooling of the same size so that the
    gradient reaches the whole receptive field."""
    mods = []
    for kind, k, s in layers:
        mods.append(nn.Conv2d(1, 1, k, stride=s) if kind == "conv" else nn.AvgPool2d(k, stride=s))
    net = nn.Sequential(*mods)
    with torch.no_grad():
        for m in net:
            if isinstance(m, nn.Conv2d):
                m.weight.fill_(1.0); m.bias.zero_()
    x = torch.ones(1, 1, size, size, requires_grad=True)
    out = net(x)
    out[0, 0, out.shape[2] // 2, out.shape[3] // 2].backward()
    rows = (x.grad[0, 0] != 0).any(dim=1).nonzero()
    return (rows.max() - rows.min() + 1).item()

stacks = [
    [("conv", 3, 1)] * 3,
    [("conv", 3, 1), ("pool", 2, 2), ("conv", 3, 1), ("pool", 2, 2), ("conv", 3, 1)],
    [("conv", 7, 2), ("pool", 3, 2)],
]
measured = [measured_rf(s) for s in stacks]
print("your answers:", rf_answers, "| measured:", measured)
assert rf_answers == measured
print("receptive fields OK")

# %% [markdown]
# **(b)** How many stacked $3\times3$ convolutions (stride 1, no pooling) would be
# needed for a receptive field covering a whole $224\times224$ image? Why do real
# networks reach large receptive fields much faster?
#
# <!-- BEGIN SOLUTION -->
# Each layer adds 2 pixels, so $1 + 2L \ge 224$ requires $L = 112$ layers. Real
# networks downsample (strided convolutions, pooling): after each downsampling by 2,
# every later $3\times3$ layer adds twice as many input pixels, so the receptive field
# grows exponentially with the number of downsampling stages. (Dilated convolutions
# and attention are other ways to enlarge it.) Note also that the *effective*
# receptive field — where the gradient is substantial — is usually much smaller than
# the theoretical one and roughly Gaussian-shaped.
# <!-- END SOLUTION -->

# %% [markdown]
# ## Exercise 6.6 — Your own CNN on the digits
#
# Design and train a CNN for the $8\times8$ digits (input shape `(N, 1, 8, 8)`) that
# reaches at least **93 %** test accuracy with **fewer than 30 000 parameters**.
# Use at least two convolutional layers. You may reuse `BasicBlock`. A training loop
# is provided; you only need to define `model`. Remember: `torch.manual_seed(0)`.

# %%
digits = load_digits()
X_all = (digits.data / 16.0).astype(np.float32).reshape(-1, 1, 8, 8)
y_all = digits.target.astype(np.int64)
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=0, stratify=y_all)
train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)), batch_size=512)


@torch.no_grad()
def accuracy(model, loader):
    model.eval()
    correct, n = 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        correct += (model(xb).argmax(1) == yb).sum().item(); n += len(yb)
    return correct / n


def train(model, n_epochs=25, lr=2e-3):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    loader = DataLoader(train_ds, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(0))
    for epoch in range(n_epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
    return model

# %%
torch.manual_seed(0)
### BEGIN SOLUTION
model = nn.Sequential(
    nn.Conv2d(1, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(),   # 8x8
    BasicBlock(16, 16),                                                          # 8x8
    BasicBlock(16, 32, stride=2),                                                # 4x4
    nn.AdaptiveAvgPool2d(1), nn.Flatten(),
    nn.Linear(32, 10),
)
### END SOLUTION
n_par = sum(p.numel() for p in model.parameters())
train(model)
test_acc = accuracy(model, test_loader)
print(f"parameters: {n_par} | test accuracy: {test_acc:.3f}")

# %%
assert sum(isinstance(m, nn.Conv2d) for m in model.modules()) >= 2
assert n_par < 30_000
assert test_acc >= 0.93
print("CNN OK")

# %% [markdown]
# ## Exercise 6.7 — Written questions
#
# **(a)** A convolutional layer maps $C_{in}$ to $C_{out}$ channels with $k\times k$
# kernels. Give its number of parameters and the number of multiply–adds for an
# $H_{out}\times W_{out}$ output. Which of the two depends on the image size?
#
# <!-- BEGIN SOLUTION -->
# Parameters: $C_{out}(C_{in}k^2 + 1)$, independent of the image size.
# Multiply–adds: $H_{out}W_{out}\,C_{out}\,C_{in}k^2$ — every output value is a dot
# product of length $C_{in}k^2$. The computation scales with the image size, the
# parameter count does not; this is why CNNs are parameter-efficient but can still be
# computationally expensive at high resolution.
# <!-- END SOLUTION -->
#
# **(b)** Is a CNN with a final `Flatten` + `Linear` layer translation invariant?
# What architectural change makes it (more) invariant, and what is lost?
#
# <!-- BEGIN SOLUTION -->
# No. The convolutions are equivariant, but the flatten + linear head has a separate
# weight for every position of the final feature map, so its output depends on
# where features occur. (Strided convolutions/pooling also break exact equivariance
# for shifts that are not multiples of the stride.) Replacing the head with global
# average (or max) pooling followed by a linear layer makes the output invariant to
# where a feature appears. What is lost is explicit spatial information — "where" a
# feature is — which matters for some tasks (e.g. distinguishing a 6 from a 9 needs
# the relative layout of parts, which must then be encoded by the features themselves).
# <!-- END SOLUTION -->
#
# **(c)** Explain why a residual block $\mathbf y = \mathbf x + \mathcal F(\mathbf x)$
# makes it easy for a very deep network to be *at least as good* as a shallower one.
#
# <!-- BEGIN SOLUTION -->
# A deeper network can always emulate a shallower one if the extra blocks compute
# the identity. With residual blocks this only requires $\mathcal F(\mathbf x)=0$, which
# is easy to reach (e.g. weights near zero, or the last BN gain $\gamma=0$), whereas a
# plain stack of nonlinear layers must learn the identity map explicitly — hard for
# a composition of conv + ReLU layers. In addition, the Jacobian $I + \partial\mathcal F/\partial\mathbf x$
# lets gradients pass through many blocks without vanishing.
# <!-- END SOLUTION -->
