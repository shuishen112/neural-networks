# %% [markdown]
# # Exercises 5 — Regularisation
#
# Fill in every cell marked `# YOUR CODE HERE` and answer the written questions.
# The `assert` cells check your work: if a cell runs without an error, you are
# probably on the right track.

# %%
import copy

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.manual_seed(0)
rng = np.random.default_rng(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

# %% [markdown]
# ## Exercise 5.1 — Inverted dropout
#
# Implement `MyDropout`, an `nn.Module` with the behaviour of `nn.Dropout(p)`:
#
# * in **training** mode (`self.training` is `True`), zero every element
#   independently with probability `p` and scale the survivors by $1/(1-p)$;
# * in **evaluation** mode, return the input unchanged.
#
# Handle the edge case `p = 0` (identity). *Hint:* `torch.rand_like(x) >= p` gives a
# boolean mask whose entries are `True` with probability $1-p$.

# %%
class MyDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        assert 0.0 <= p < 1.0
        self.p = p

    def forward(self, x):
        ### BEGIN SOLUTION
        if not self.training or self.p == 0.0:
            return x
        mask = (torch.rand_like(x) >= self.p).to(x.dtype)
        return x * mask / (1.0 - self.p)
        ### END SOLUTION

# %%
torch.manual_seed(0)
d = MyDropout(0.3)
x = torch.ones(2000, 500)
d.train()
out = d(x)
frac_zero = (out == 0).float().mean().item()
assert abs(frac_zero - 0.3) < 0.01, frac_zero
assert abs(out.mean().item() - 1.0) < 0.01, out.mean().item()              # expectation preserved
assert torch.allclose(out[out != 0], torch.full_like(out[out != 0], 1 / 0.7))
d.eval()
assert torch.equal(d(x), x)
assert torch.equal(MyDropout(0.0).train()(x), x)
# gradients flow only through the kept units, with the same scaling
xg = torch.ones(10, 10, requires_grad=True)
d.train(); y = d(xg); y.sum().backward()
assert torch.allclose(xg.grad, (y != 0).float() / 0.7)
print("MyDropout OK")

# %% [markdown]
# **Question.** Each unit of a layer with $n$ inputs $h_j$ is followed by a linear
# layer computing $a = \sum_j w_j \tilde h_j$. With dropout (rate $p$, inverted
# scaling), what are $\mathbb E[a]$ and $\operatorname{Var}[a]$ over the random mask,
# for fixed $h$ and $w$? What does this say about the "noise" dropout injects?
#
# <!-- BEGIN SOLUTION -->
# With $\tilde h_j = m_j h_j/(1-p)$, $m_j\sim\text{Bernoulli}(1-p)$ independent:
# $\mathbb E[\tilde h_j] = h_j$, so $\mathbb E[a] = \sum_j w_j h_j$ — the same as without
# dropout. $\operatorname{Var}[\tilde h_j] = h_j^2\,\frac{(1-p)p}{(1-p)^2} = \frac{p}{1-p}h_j^2$,
# hence $\operatorname{Var}[a] = \frac{p}{1-p}\sum_j w_j^2 h_j^2$. Dropout is
# multiplicative noise whose size grows with the weights and activations; penalising
# it in expectation acts like a data-dependent $L_2$ penalty
# $\propto\sum_j w_j^2 h_j^2$ (exactly so for linear regression), and it grows quickly
# as $p\to1$.
# <!-- END SOLUTION -->

# %% [markdown]
# ## Exercise 5.2 — Batch normalisation from scratch
#
# Implement `batchnorm_train(x, gamma, beta, running_mean, running_var, momentum, eps)`
# for an input of shape `(B, D)` in **training mode**. It must
#
# 1. normalise every feature with the **biased** batch variance,
# 2. apply the affine transform $\gamma\hat x + \beta$,
# 3. update the running statistics **in place** like PyTorch does:
#    `running = (1 - momentum) * running + momentum * batch_stat`, where the running
#    variance is updated with the **unbiased** batch variance ($\frac{B}{B-1}\sigma^2$),
#
# and return the output. Then implement `batchnorm_eval`, which uses the running
# statistics. We compare against `nn.BatchNorm1d`, including the gradients.

# %%
def batchnorm_train(x, gamma, beta, running_mean, running_var, momentum=0.1, eps=1e-5):
    ### BEGIN SOLUTION
    mu = x.mean(dim=0)
    var = x.var(dim=0, correction=0)
    x_hat = (x - mu) / torch.sqrt(var + eps)
    B = x.shape[0]
    with torch.no_grad():
        running_mean.mul_(1 - momentum).add_(momentum * mu)
        running_var.mul_(1 - momentum).add_(momentum * var * B / (B - 1))
    return gamma * x_hat + beta
    ### END SOLUTION


def batchnorm_eval(x, gamma, beta, running_mean, running_var, eps=1e-5):
    ### BEGIN SOLUTION
    return gamma * (x - running_mean) / torch.sqrt(running_var + eps) + beta
    ### END SOLUTION

# %%
torch.manual_seed(0)
B, D = 32, 5
ref = nn.BatchNorm1d(D)
with torch.no_grad():
    ref.weight.copy_(torch.rand(D) + 0.5)
    ref.bias.copy_(torch.randn(D))
gamma = ref.weight.detach().clone().requires_grad_()
beta = ref.bias.detach().clone().requires_grad_()
rm, rv = torch.zeros(D), torch.ones(D)

ref.train()
for step in range(5):
    x = (3 * torch.randn(B, D) + 2).requires_grad_()
    x2 = x.detach().clone().requires_grad_()
    y_ref = ref(x)
    y_my = batchnorm_train(x2, gamma, beta, rm, rv)
    assert torch.allclose(y_ref, y_my, atol=1e-5)
    w = torch.randn(B, D)
    (y_ref * w).sum().backward()
    (y_my * w).sum().backward()
    assert torch.allclose(x.grad, x2.grad, atol=1e-5)
assert torch.allclose(rm, ref.running_mean, atol=1e-5)
assert torch.allclose(rv, ref.running_var, atol=1e-5)
assert torch.allclose(gamma.grad, ref.weight.grad, atol=1e-4)
assert torch.allclose(beta.grad, ref.bias.grad, atol=1e-4)

ref.eval()
xt = torch.randn(3, D)
assert torch.allclose(ref(xt), batchnorm_eval(xt, gamma, beta, rm, rv), atol=1e-5)
print("batch norm OK")

# %% [markdown]
# **Question.** In training mode, what does batch norm output for a batch of size 1?
# Try `nn.BatchNorm1d(3).train()(torch.randn(1, 3))` and explain the result.
#
# <!-- BEGIN SOLUTION -->
# With $B=1$ the batch mean equals the example and the batch variance is zero, so
# $\hat x = 0$ and the output would be $\beta$ regardless of the input — all information
# is destroyed (and the unbiased running variance would divide by $B-1=0$). PyTorch
# therefore raises `ValueError: Expected more than 1 value per channel when training`.
# Batch norm needs reasonably large batches; this is one motivation for layer norm.
# <!-- END SOLUTION -->

# %% [markdown]
# ## Exercise 5.3 — An early-stopping helper
#
# Implement the class `EarlyStopping`. Calling `stopper.step(val_loss, model)` after
# every epoch should
#
# * if `val_loss < best - min_delta`: record the new best loss, store a **deep copy**
#   of `model.state_dict()` in `self.best_state`, reset the counter, and return `False`;
# * otherwise increase the counter and return `True` once it reaches `patience`
#   (meaning "stop now"), `False` before that.
#
# `stopper.restore(model)` loads the best state back into the model.

# %%
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience, self.min_delta = patience, min_delta
        self.best = float("inf")
        self.best_state = None
        self.counter = 0

    def step(self, val_loss, model):
        ### BEGIN SOLUTION
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience
        ### END SOLUTION

    def restore(self, model):
        ### BEGIN SOLUTION
        model.load_state_dict(self.best_state)
        ### END SOLUTION

# %%
lin = nn.Linear(2, 1)
es = EarlyStopping(patience=3)
decisions = []
for i, loss in enumerate([1.0, 0.8, 0.9, 0.7, 0.75, 0.71, 0.72, 0.6]):
    with torch.no_grad():
        lin.bias.fill_(float(i))               # mark the model with the epoch number
    decisions.append(es.step(loss, lin))
assert decisions == [False, False, False, False, False, False, True, False], decisions
assert es.best == 0.6
es2 = EarlyStopping(patience=3)
for i, loss in enumerate([1.0, 0.5, 0.6, 0.7, 0.8]):
    with torch.no_grad():
        lin.bias.fill_(float(i))
    if es2.step(loss, lin):
        break
assert i == 4
es2.restore(lin)
assert lin.bias.item() == 1.0, "restore() must load the state of the best epoch (a deep copy!)"
print("EarlyStopping OK")

# %% [markdown]
# ## Exercise 5.4 — A regularisation experiment
#
# We use the setting of the lecture: 300 training digits, 20 % of them with random
# labels, and a clean validation set. The functions `make_model` and `run` are given.
#
# Train **three** configurations for 100 epochs each, using your `EarlyStopping`
# (patience 15) to restore the best model at the end:
#
# * `"baseline"`: `make_model(dropout=0.0)` with `Adam(lr=1e-3)`;
# * `"dropout"`: `make_model(dropout=0.5)` with `Adam(lr=1e-3)`;
# * `"dropout+wd"`: `make_model(dropout=0.5)` with `AdamW(lr=1e-3, weight_decay=1.0)`.
#
# Store the validation accuracy of each restored model in the dictionary `results`,
# and the model itself in `trained`. Use `torch.manual_seed(0)` before creating each model.

# %%
digits = load_digits()
X_all = (digits.data / 16.0).astype(np.float32)
y_all = digits.target.astype(np.int64)
X_pool, X_test, y_pool, y_test = train_test_split(X_all, y_all, test_size=0.25, random_state=0, stratify=y_all)
X_rest, X_val, y_rest, y_val = train_test_split(X_pool, y_pool, test_size=0.3, random_state=0, stratify=y_pool)
X_small, y_small = X_rest[:300], y_rest[:300].copy()
flip = rng.random(300) < 0.2
y_small[flip] = rng.integers(0, 10, flip.sum())

train_ds = TensorDataset(torch.from_numpy(X_small), torch.from_numpy(y_small))
val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=512)
test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)), batch_size=512)


def make_model(dropout=0.0):
    return nn.Sequential(
        nn.Linear(64, 256), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(256, 256), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(256, 10),
    ).to(device)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        total += F.cross_entropy(logits, yb, reduction="sum").item()
        correct += (logits.argmax(1) == yb).sum().item()
        n += len(yb)
    return total / n, correct / n


def run(model, optimizer, stopper, n_epochs=100):
    """Train with early stopping; returns the list of validation losses."""
    loader = DataLoader(train_ds, batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(0))
    val_losses = []
    for epoch in range(n_epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.cross_entropy(model(xb), yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        val_loss, _ = evaluate(model, val_loader)
        val_losses.append(val_loss)
        if stopper.step(val_loss, model):
            break
    stopper.restore(model)
    return val_losses

# %%
results, trained, curves = {}, {}, {}
### BEGIN SOLUTION
configs = {
    "baseline": (0.0, lambda p: torch.optim.Adam(p, lr=1e-3)),
    "dropout": (0.5, lambda p: torch.optim.Adam(p, lr=1e-3)),
    "dropout+wd": (0.5, lambda p: torch.optim.AdamW(p, lr=1e-3, weight_decay=1.0)),
}
for name, (p_drop, make_opt) in configs.items():
    torch.manual_seed(0)
    model = make_model(dropout=p_drop)
    curves[name] = run(model, make_opt(model.parameters()), EarlyStopping(patience=15))
    results[name] = evaluate(model, val_loader)[1]
    trained[name] = model
### END SOLUTION

for name, c in curves.items():
    plt.plot(c, label=name)
plt.xlabel("epoch"); plt.ylabel("validation loss"); plt.legend(); plt.show()
for name, acc in results.items():
    print(f"{name:12s} val acc {acc:.3f} | test acc {evaluate(trained[name], test_loader)[1]:.3f}")

# %%
assert set(results) == {"baseline", "dropout", "dropout+wd"}
assert all(0.0 <= a <= 1.0 for a in results.values())
assert all(not m.training for m in trained.values())
best_name = max(results, key=results.get)
best_test = evaluate(trained[best_name], test_loader)[1]
print("best configuration:", best_name, "| test accuracy", round(best_test, 3))
assert best_test >= 0.78
print("experiment OK")

# %% [markdown]
# **Question.** We selected the best configuration by its *validation* accuracy and
# reported its *test* accuracy. Why would reporting the best of the three *test*
# accuracies be optimistic? Also, early stopping itself used the validation set —
# is the validation accuracy of the restored model an unbiased estimate?
#
# <!-- BEGIN SOLUTION -->
# Taking the maximum over several noisy estimates is biased upwards: some of the
# winner's advantage is luck on that particular test sample, and selecting on the
# test set turns it into a validation set. The validation accuracy of the restored
# model is also optimistic, for the same reason — both the epoch and the
# configuration were chosen to minimise the validation loss. Only data that played
# no role in any decision (here the test set, used once) gives an unbiased estimate.
# <!-- END SOLUTION -->

# %% [markdown]
# ## Exercise 5.5 — Written questions
#
# **(a)** Show that for plain SGD (no momentum), adding $\frac{\lambda}{2}\lVert\mathbf
# w\rVert^2$ to the loss is identical to multiplying the weights by $(1-\eta\lambda)$
# before each gradient step. Why is this no longer true for Adam?
#
# <!-- BEGIN SOLUTION -->
# SGD on the penalised loss: $\mathbf w \leftarrow \mathbf w - \eta(\nabla\mathcal L + \lambda\mathbf w)
# = (1-\eta\lambda)\mathbf w - \eta\nabla\mathcal L$, which is exactly decay followed by a
# plain gradient step. In Adam, the term $\lambda\mathbf w$ enters the moment estimates
# and is divided elementwise by $\sqrt{\hat v_t}+\epsilon$; coordinates with large
# gradient history are therefore decayed less and those with small gradients more,
# so the effective regularisation is no longer a uniform shrinkage. AdamW applies the
# decay $\eta\lambda\mathbf w$ outside the adaptive normalisation, restoring the
# intended behaviour.
# <!-- END SOLUTION -->
#
# **(b)** A colleague evaluates a network containing batch norm and dropout but
# forgets `model.eval()`. List two distinct ways in which the reported results are wrong.
#
# <!-- BEGIN SOLUTION -->
# (1) Dropout stays active, so predictions are random and on average worse —
# results are noisy and pessimistic. (2) Batch norm normalises with the statistics
# of the evaluation batch instead of the running averages, so a prediction depends
# on the other examples in the batch (and fails for batch size 1); moreover, each
# forward pass in training mode *updates* the running statistics with test data,
# silently changing the model.
# <!-- END SOLUTION -->
#
# **(c)** Which of these augmentations are appropriate for (i) handwritten digit
# recognition and (ii) classifying photos of animals: horizontal flip, rotation by
# ±10°, rotation by 180°, small translations, colour jitter? Justify briefly.
#
# <!-- BEGIN SOLUTION -->
# Digits: small rotations and small translations preserve the label; horizontal
# flips and 180° rotations do not (2 ↔ mirrored 2, 6 ↔ 9); colour jitter is
# irrelevant for grey-scale ink but brightness/contrast changes are harmless.
# Animals: horizontal flip, small rotations, translations/crops and colour jitter
# are all label-preserving; a 180° rotation produces upside-down animals that do
# not occur in natural photos — it rarely helps and can hurt.
# <!-- END SOLUTION -->
#
# **(d)** In the double-descent experiment of the lecture, the test error peaks when
# the number of random features $p$ equals the number of training examples $n$.
# Explain why, in terms of the singular values of the feature matrix.
#
# <!-- BEGIN SOLUTION -->
# The least-squares solution is $W = \Phi^{+}Y$; along each singular direction the
# targets are multiplied by $1/s_i$. When $\Phi$ is (nearly) square, its smallest
# singular values are very close to zero (a random square matrix is badly
# conditioned), so the noise in $Y$ is amplified enormously and the norm of $W$
# explodes — high variance. For $p\ll n$ or $p\gg n$, the matrix is far from square,
# its smallest singular value is bounded away from zero, and (for $p > n$) the
# pseudo-inverse picks the minimum-norm interpolant, which is smooth.
# <!-- END SOLUTION -->
