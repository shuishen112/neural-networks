# %% [markdown]
# # Exercises 4 — PyTorch Fundamentals
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
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.manual_seed(0)
np.random.seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# %% [markdown]
# ## Exercise 4.1 — Gradient descent with raw autograd
#
# Before using `nn.Module` and `torch.optim`, let us use autograd directly. Fit a
# cubic polynomial $\hat y = a + bx + cx^2 + dx^3$ to noisy samples of $\sin(x)$
# on $[-\pi, \pi]$ by gradient descent on the mean squared error.
#
# Inside the loop:
# 1. compute the prediction and the loss from the leaf tensor `coef` (shape `(4,)`);
# 2. call `.backward()`;
# 3. update `coef` **inside `torch.no_grad()`** and reset its gradient.
#
# Store `loss.item()` in `losses` at every step.

# %%
torch.manual_seed(0)
x = torch.linspace(-np.pi, np.pi, 200)
y = torch.sin(x) + 0.05 * torch.randn(200)
powers = torch.stack([x ** k for k in range(4)], dim=1)     # (200, 4): 1, x, x^2, x^3

coef = torch.zeros(4, requires_grad=True)
lr, losses = 1e-3, []
for step in range(3000):
    ### BEGIN SOLUTION
    pred = powers @ coef
    loss = ((pred - y) ** 2).mean()
    loss.backward()
    with torch.no_grad():
        coef -= lr * coef.grad
    coef.grad.zero_()
    losses.append(loss.item())
    ### END SOLUTION

print("coefficients:", [round(c, 3) for c in coef.tolist()])
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].semilogy(losses); axes[0].set_xlabel("step"); axes[0].set_title("MSE")
axes[1].scatter(x.numpy(), y.numpy(), s=5, label="data")
axes[1].plot(x.numpy(), (powers @ coef).detach().numpy(), "k", label="cubic fit"); axes[1].legend()
plt.show()

# %%
assert len(losses) == 3000 and isinstance(losses[0], float)
assert coef.is_leaf and coef.requires_grad
assert losses[-1] < 0.05, losses[-1]
# Taylor series: sin x ~ x - x^3/6, so the fit should have b > 0 and d < 0.
assert coef[1].item() > 0.5 and coef[3].item() < 0
print("raw autograd OK")

# %% [markdown]
# ## Exercise 4.2 — A custom autograd `Function`
#
# Autograd knows the derivatives of all built-in operations. To add your own
# operation, subclass `torch.autograd.Function` and implement two static methods:
#
# * `forward(ctx, x)` computes the output; use `ctx.save_for_backward(...)` to
#   store tensors needed later;
# * `backward(ctx, grad_output)` receives $\partial L/\partial \text{output}$ and
#   must return $\partial L/\partial x$ (one gradient per input of `forward`).
#
# The function is used through `MyReLU.apply(x)`.
#
# **(a)** Implement `MyReLU`. The test uses `torch.autograd.gradcheck`, which
# compares your `backward` against finite differences (in float64).
#
# **(b)** Implement the **straight-through estimator** for the sign function. The
# true derivative of $\operatorname{sign}(x)$ is zero almost everywhere, which
# would stop all learning. The straight-through estimator (Bengio et al., 2013),
# used to train binarised and quantised networks, pretends in the backward pass
# that the function was the identity, clipped to $|x|\le 1$:
#
# $$\text{forward: } \operatorname{sign}(x), \qquad
# \text{backward: } \frac{\partial L}{\partial x} := \frac{\partial L}{\partial \text{out}}\cdot \mathbb 1\big[|x| \le 1\big].$$

# %%
class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ### BEGIN SOLUTION
        ctx.save_for_backward(x)
        return x.clamp(min=0)
        ### END SOLUTION

    @staticmethod
    def backward(ctx, grad_output):
        ### BEGIN SOLUTION
        (x,) = ctx.saved_tensors
        return grad_output * (x > 0).to(grad_output.dtype)
        ### END SOLUTION


class SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ### BEGIN SOLUTION
        ctx.save_for_backward(x)
        return torch.sign(x)
        ### END SOLUTION

    @staticmethod
    def backward(ctx, grad_output):
        ### BEGIN SOLUTION
        (x,) = ctx.saved_tensors
        return grad_output * (x.abs() <= 1).to(grad_output.dtype)
        ### END SOLUTION

# %%
torch.manual_seed(0)
xt = torch.randn(20, 5, dtype=torch.float64)
xt = (xt + 0.1 * torch.sign(xt)).requires_grad_()      # keep away from the kink at 0
assert torch.equal(MyReLU.apply(xt), torch.relu(xt))
assert torch.autograd.gradcheck(MyReLU.apply, (xt,))

xs = torch.tensor([-2.0, -0.5, 0.3, 1.5], requires_grad=True)
out = SignSTE.apply(xs)
assert torch.equal(out, torch.tensor([-1.0, -1.0, 1.0, 1.0]))
(out * torch.tensor([1.0, 2.0, 3.0, 4.0])).sum().backward()
assert torch.equal(xs.grad, torch.tensor([0.0, 2.0, 3.0, 0.0])), xs.grad
print("custom Functions OK")

# %% [markdown]
# ## Exercise 4.3 — A configurable MLP module
#
# Write an `nn.Module` called `MLP` that takes a list of layer sizes, e.g.
# `MLP([64, 128, 64, 10])`, and builds `Linear` layers between consecutive sizes
# with the given activation **between** them (not after the last layer — the
# model must output logits). Optionally apply `nn.Dropout(dropout)` after each
# hidden activation when `dropout > 0`.
#
# *Hint:* collect the layers in a Python list and wrap it in `nn.Sequential(*layers)`.
# A plain list attribute would not register the parameters!

# %%
class MLP(nn.Module):
    def __init__(self, sizes, activation=nn.ReLU, dropout=0.0):
        super().__init__()
        ### BEGIN SOLUTION
        layers = []
        for i, (d_in, d_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(nn.Linear(d_in, d_out))
            if i < len(sizes) - 2:
                layers.append(activation())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
        ### END SOLUTION

    def forward(self, x):
        ### BEGIN SOLUTION
        return self.net(x)
        ### END SOLUTION

# %%
m = MLP([64, 128, 64, 10])
n_params = sum(p.numel() for p in m.parameters())
assert n_params == 64 * 128 + 128 + 128 * 64 + 64 + 64 * 10 + 10, n_params
assert m(torch.randn(7, 64)).shape == (7, 10)
mods = list(m.modules())
assert sum(isinstance(k, nn.Linear) for k in mods) == 3
assert sum(isinstance(k, nn.ReLU) for k in mods) == 2          # none after the last layer
assert sum(isinstance(k, nn.Tanh) for k in MLP([2, 3, 3, 1], activation=nn.Tanh).modules()) == 2
assert sum(isinstance(k, nn.Dropout) for k in MLP([2, 3, 3, 1], dropout=0.5).modules()) == 2
assert MLP([5, 1])(torch.randn(3, 5)).shape == (3, 1)
print("MLP OK")

# %% [markdown]
# ## Exercise 4.4 — A custom `Dataset`
#
# Implement `DigitsDataset`, which stores the digits and **standardises** every
# feature with a given mean and standard deviation (computed on the training set
# and passed in, so that validation and test data use the *same* statistics).
# `__getitem__` must return a pair `(x, y)` with `x` a `float32` tensor of shape
# `(64,)` and `y` an `int64` scalar tensor. (Some border pixels are always 0 in
# the training set; their standard deviation has already been replaced by 1 below
# to avoid dividing by zero.)

# %%
digits = load_digits()
X_all, y_all = digits.data, digits.target
X_trval, X_test, y_trval, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=0, stratify=y_all)
X_train, X_val, y_train, y_val = train_test_split(X_trval, y_trval, test_size=0.2, random_state=0, stratify=y_trval)
mean, std = X_train.mean(axis=0), X_train.std(axis=0)
std[std == 0] = 1.0


class DigitsDataset(Dataset):
    def __init__(self, X, y, mean, std):
        ### BEGIN SOLUTION
        Xs = (X - mean) / std
        self.X = torch.as_tensor(Xs, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)
        ### END SOLUTION

    def __len__(self):
        ### BEGIN SOLUTION
        return len(self.y)
        ### END SOLUTION

    def __getitem__(self, idx):
        ### BEGIN SOLUTION
        return self.X[idx], self.y[idx]
        ### END SOLUTION


train_ds = DigitsDataset(X_train, y_train, mean, std)
val_ds = DigitsDataset(X_val, y_val, mean, std)
test_ds = DigitsDataset(X_test, y_test, mean, std)

# %%
assert len(train_ds) == len(X_train)
x0, y0 = train_ds[0]
assert x0.shape == (64,) and x0.dtype == torch.float32
assert y0.dtype == torch.int64 and y0.ndim == 0
all_train = torch.stack([train_ds[i][0] for i in range(len(train_ds))])
assert torch.allclose(all_train.mean(0), torch.zeros(64), atol=1e-4)
xb, yb = next(iter(DataLoader(train_ds, batch_size=32, shuffle=True)))
assert xb.shape == (32, 64) and yb.shape == (32,)
print("Dataset OK")

# %% [markdown]
# ## Exercise 4.5 — A reusable `fit` function
#
# Write two functions that we will reuse in the following lectures:
#
# * `evaluate(model, loader, loss_fn)` returns `(mean loss, accuracy)` over the
#   whole loader, computed in **eval mode** and **without gradients**;
# * `fit(model, train_loader, val_loader, loss_fn, optimizer, n_epochs)` runs the
#   canonical training loop (train mode, forward, `zero_grad`, backward, step) and
#   after every epoch appends the validation loss and accuracy (from `evaluate`)
#   and the average training loss to the lists in `history`.
#
# Move every batch to `device`. Weight batch losses by batch size so that the
# averages are exact.

# %%
def evaluate(model, loader, loss_fn):
    ### BEGIN SOLUTION
    model.eval()
    total, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            total += loss_fn(logits, yb).item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            n += len(yb)
    return total / n, correct / n
    ### END SOLUTION


def fit(model, train_loader, val_loader, loss_fn, optimizer, n_epochs):
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    for epoch in range(n_epochs):
        ### BEGIN SOLUTION
        model.train()
        total, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = loss_fn(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(yb)
            n += len(yb)
        va_loss, va_acc = evaluate(model, val_loader, loss_fn)
        history["train_loss"].append(total / n)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        ### END SOLUTION
    return history

# %% [markdown]
# Now train `MLP([64, 128, 10])` with SGD + momentum (lr 0.05, momentum 0.9) for
# 30 epochs, batch size 64, and report the test accuracy.

# %%
torch.manual_seed(0)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(0))
val_loader = DataLoader(val_ds, batch_size=256)
test_loader = DataLoader(test_ds, batch_size=256)

model = MLP([64, 128, 10]).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
loss_fn = nn.CrossEntropyLoss()
history = fit(model, train_loader, val_loader, loss_fn, optimizer, n_epochs=30)
_, test_acc = evaluate(model, test_loader, loss_fn)
print(f"final val acc {history['val_acc'][-1]:.3f} | test acc {test_acc:.3f}")

plt.plot(history["train_loss"], label="train"); plt.plot(history["val_loss"], label="validation")
plt.xlabel("epoch"); plt.ylabel("cross-entropy"); plt.legend(); plt.show()

# %%
assert len(history["val_acc"]) == 30
assert history["train_loss"][-1] < history["train_loss"][0]
assert not model.training, "evaluate() should leave the model in eval mode"
assert test_acc >= 0.90
print("fit OK")

# %% [markdown]
# ## Exercise 4.6 — Written questions
#
# **(a)** A student forgets `optimizer.zero_grad()` in the training loop. The code
# runs without errors. What actually happens to the updates, and why does
# training usually become unstable?
#
# <!-- BEGIN SOLUTION -->
# `.backward()` adds into `.grad`, so after $t$ steps the gradient used is the sum
# of all gradients seen so far, $\sum_{s\le t} \nabla\mathcal L_s$. The update behaves
# like heavy momentum with no decay ($\beta = 1$): the effective step size grows
# linearly with the number of steps and old, stale gradients never fade away, so
# the parameters overshoot and training typically diverges.
# <!-- END SOLUTION -->
#
# **(b)** What is the difference between `model.eval()` and `torch.no_grad()`? Give
# an example where using only one of them during validation gives a wrong or
# wasteful result.
#
# <!-- BEGIN SOLUTION -->
# `model.eval()` changes the *behaviour* of layers such as dropout (disabled) and
# batch normalisation (uses running statistics); it does not affect autograd.
# `torch.no_grad()` stops recording the computational graph; it does not change
# layer behaviour. Using only `no_grad` in a network with dropout gives noisy,
# too-pessimistic validation scores (and with batch norm, it even updates the
# running statistics on validation data). Using only `eval()` gives correct numbers
# but wastes memory and time storing a graph that is never used.
# <!-- END SOLUTION -->
#
# **(c)** Why is it wrong to end a classifier with `nn.Softmax` when training with
# `nn.CrossEntropyLoss`? What should you do if you need probabilities at test time?
#
# <!-- BEGIN SOLUTION -->
# `CrossEntropyLoss` expects logits and applies `log_softmax` itself. With an extra
# softmax the loss sees values in $[0,1]$ instead of unbounded logits, so it cannot
# approach zero, and the extra Jacobian of the softmax shrinks the gradients, which
# slows learning. Keep the model output as logits and apply `torch.softmax(logits, dim=1)`
# separately when probabilities are needed (the argmax is the same either way).
# <!-- END SOLUTION -->
#
# **(d)** In Exercise 4.1 we wrote `with torch.no_grad(): coef -= lr * coef.grad`.
# What would go wrong without the `no_grad` block?
#
# <!-- BEGIN SOLUTION -->
# `coef` is a leaf that requires grad; an in-place modification of such a leaf
# while autograd is recording is forbidden and raises a `RuntimeError`. (Writing
# `coef = coef - lr * coef.grad` instead would avoid the error but would make
# `coef` a non-leaf node of an ever-growing graph whose `.grad` is no longer populated.)
# The update is not part of the model, so it must not be recorded.
# <!-- END SOLUTION -->
