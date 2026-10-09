# %% [markdown]
# # Exercises 3 — Optimisation and Initialisation
#
# Fill in every cell marked `# YOUR CODE HERE` and answer the written questions.
# The `assert` cells check your work.

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(0)

# %% [markdown]
# ## Exercise 3.1 — The stability limit of gradient descent
#
# For the quadratic $f(\mathbf w) = \tfrac12 \mathbf w^\top A\,\mathbf w$ with symmetric
# positive-definite $A$, gradient descent $\mathbf w \leftarrow \mathbf w - \eta A\mathbf w$
# converges for every starting point if and only if $\eta < 2/\lambda_\text{max}(A)$.
#
# 1. Implement `max_stable_lr(A)`.
# 2. Implement `run_gd(A, w0, lr, steps)` returning the final iterate.
#
# The test then checks that GD converges just below the limit and diverges just above it.

# %%
def max_stable_lr(A):
    ### BEGIN SOLUTION
    return 2.0 / np.linalg.eigvalsh(A).max()
    ### END SOLUTION


def run_gd(A, w0, lr, steps):
    w = np.array(w0, dtype=float)
    ### BEGIN SOLUTION
    for _ in range(steps):
        w = w - lr * A @ w
    ### END SOLUTION
    return w

# %%
Q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
A = Q @ np.diag([0.1, 0.5, 1.0, 4.0, 10.0]) @ Q.T
lr_max = max_stable_lr(A)
w0 = rng.normal(size=5)
print(f"eta_max = {lr_max:.4f}")
assert np.isclose(lr_max, 0.2)
assert np.linalg.norm(run_gd(A, w0, 0.95 * lr_max, 2000)) < 1e-3
assert np.linalg.norm(run_gd(A, w0, 1.05 * lr_max, 2000)) > 1e3
print("stability limit OK")

# %% [markdown]
# **Question.** With $\eta$ at its best constant value, the error contracts by roughly
# $\frac{\kappa-1}{\kappa+1}$ per step, where $\kappa=\lambda_\text{max}/\lambda_\text{min}$.
# What is $\kappa$ for the matrix above, and roughly how many steps are needed to reduce the
# error by a factor $10^{6}$? What does momentum change?
#
# <!-- BEGIN SOLUTION -->
# $\kappa = 10/0.1 = 100$, so the rate is $99/101\approx 0.980$ and we need
# $\ln 10^{6}/(-\ln 0.980) \approx 690$ steps. With heavy-ball momentum the rate becomes
# $\frac{\sqrt\kappa-1}{\sqrt\kappa+1} = 9/11 \approx 0.818$, i.e. about 70 steps:
# momentum replaces $\kappa$ by $\sqrt\kappa$.
# <!-- END SOLUTION -->

# %% [markdown]
# ## Exercise 3.2 — Implement momentum and Adam
#
# Optimisers keep state between steps. Implement the two classes below. Both act on a
# list of parameter arrays and **update them in place** given a list of gradients.
#
# * `Momentum`: $\mathbf v \leftarrow \mu \mathbf v + \mathbf g$, $\;\theta \leftarrow \theta - \eta \mathbf v$.
# * `Adam`: $\mathbf m \leftarrow \beta_1\mathbf m + (1-\beta_1)\mathbf g$,
#   $\mathbf v \leftarrow \beta_2\mathbf v + (1-\beta_2)\mathbf g^2$,
#   $\hat{\mathbf m} = \mathbf m/(1-\beta_1^t)$, $\hat{\mathbf v} = \mathbf v/(1-\beta_2^t)$,
#   $\theta \leftarrow \theta - \eta\,\hat{\mathbf m}/(\sqrt{\hat{\mathbf v}}+\epsilon)$.

# %%
class Momentum:
    def __init__(self, params, lr=0.01, mu=0.9):
        self.params, self.lr, self.mu = params, lr, mu
        self.v = [np.zeros_like(p) for p in params]

    def step(self, grads):
        ### BEGIN SOLUTION
        for p, g, v in zip(self.params, grads, self.v):
            v *= self.mu
            v += g
            p -= self.lr * v
        ### END SOLUTION


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        ### BEGIN SOLUTION
        self.t += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= self.b1; m += (1 - self.b1) * g
            v *= self.b2; v += (1 - self.b2) * g * g
            m_hat = m / (1 - self.b1 ** self.t)
            v_hat = v / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        ### END SOLUTION

# %%
# Test 1: the very first Adam step has magnitude ~lr in every coordinate, whatever the gradient scale.
p = [np.zeros(4)]
opt = Adam(p, lr=0.1)
opt.step([np.array([1e-4, -3.0, 50.0, -1e3])])
assert np.allclose(p[0], [-0.1, 0.1, -0.1, 0.1], atol=1e-4), p[0]

# Test 2: momentum with constant gradient g accumulates v_t = g (1 - mu^t) / (1 - mu).
p = [np.zeros(1)]
opt = Momentum(p, lr=1.0, mu=0.5)
for _ in range(3):
    opt.step([np.ones(1)])
assert np.isclose(p[0][0], -(1 + 1.5 + 1.75))

# Test 3: both minimise an ill-conditioned quadratic.
for Opt, kw in [(Momentum, dict(lr=0.02, mu=0.9)), (Adam, dict(lr=0.05))]:
    w = [np.array([1.0, 1.0])]
    opt = Opt(w, **kw)
    for _ in range(2000):
        opt.step([np.array([100.0, 1.0]) * w[0]])
    assert np.linalg.norm(w[0]) < 1e-2, (Opt.__name__, w[0])
print("optimisers OK")

# %% [markdown]
# ## Exercise 3.3 — Warm-up followed by cosine decay
#
# Implement the learning-rate schedule
#
# $$\eta_t = \begin{cases} \eta_0\,\dfrac{t+1}{W} & t < W \\[6pt]
# \eta_\text{min} + \tfrac12(\eta_0-\eta_\text{min})\Big(1+\cos\big(\pi\,\tfrac{t-W}{T-W}\big)\Big) & t \ge W \end{cases}$$
#
# for steps $t = 0,\dots,T-1$.

# %%
def warmup_cosine(t, T, lr0, warmup, lr_min=0.0):
    ### BEGIN SOLUTION
    if t < warmup:
        return lr0 * (t + 1) / warmup
    progress = (t - warmup) / (T - warmup)
    return lr_min + 0.5 * (lr0 - lr_min) * (1 + np.cos(np.pi * progress))
    ### END SOLUTION

# %%
T, W = 1000, 100
lrs = np.array([warmup_cosine(t, T, 0.1, W, 1e-4) for t in range(T)])
plt.plot(lrs); plt.xlabel("step"); plt.ylabel("learning rate"); plt.title("Warm-up + cosine"); plt.show()
assert np.isclose(lrs[0], 0.1 / W) and np.isclose(lrs[W - 1], 0.1) and np.isclose(lrs[W], 0.1)
assert np.all(np.diff(lrs[:W]) > 0) and np.all(np.diff(lrs[W:]) < 0)
assert lrs[-1] < 1e-3 and lrs.min() >= 1e-4 - 1e-12
print("schedule OK")

# %% [markdown]
# ## Exercise 3.4 — Signal propagation through a deep ReLU network
#
# Implement `activation_stds(init, ...)`: pass a batch of standard-normal inputs through
# `n_layers` fully-connected ReLU layers (no biases) whose weights are drawn with the given
# initialisation, and return the standard deviation of the pre-activations at every layer.
#
# * `"xavier"`: $W_{ij}\sim\mathcal N\!\big(0, \tfrac{2}{n_\text{in}+n_\text{out}}\big)$
# * `"he"`: $W_{ij}\sim\mathcal N\!\big(0, \tfrac{2}{n_\text{in}}\big)$
#
# Then compare the two on a 30-layer network.

# %%
def activation_stds(init, n_layers=30, width=256, batch=512, seed=0):
    r = np.random.default_rng(seed)
    h = r.normal(size=(batch, width))
    stds = []
    ### BEGIN SOLUTION
    for _ in range(n_layers):
        var = 2.0 / (2 * width) if init == "xavier" else 2.0 / width
        W = r.normal(0.0, np.sqrt(var), size=(width, width))
        z = h @ W
        stds.append(z.std())
        h = np.maximum(z, 0)
    ### END SOLUTION
    return np.array(stds)

# %%
s_x, s_h = activation_stds("xavier"), activation_stds("he")
plt.semilogy(s_x, label="Xavier"); plt.semilogy(s_h, label="He")
plt.xlabel("layer"); plt.ylabel("std of pre-activations"); plt.legend(); plt.show()
assert len(s_h) == 30
assert 0.5 < s_h[-1] / s_h[0] < 2.0            # He keeps the signal alive
assert s_x[-1] / s_x[0] < 1e-3                 # Xavier shrinks it by ~1/sqrt(2) per ReLU layer
print("initialisation OK")

# %% [markdown]
# **Question.** By what factor does the variance shrink per layer with Xavier initialisation
# in this ReLU network (square layers)? Why is Xavier nevertheless a good choice for tanh?
#
# <!-- BEGIN SOLUTION -->
# For square layers Xavier uses $\operatorname{Var}(W)=1/n$, while a ReLU zeroes half of its
# input so $\mathbb E[h^2]=\tfrac12\operatorname{Var}(z)$. Hence $\operatorname{Var}(z_{l+1})=\tfrac12\operatorname{Var}(z_l)$:
# the variance halves every layer (std shrinks by $1/\sqrt2$), giving $2^{-29}$ after 30 layers.
# Tanh is approximately the identity around zero ($\tanh'(0)=1$) and is symmetric, so it
# preserves variance to first order and the factor 2 that He adds is not needed.
# <!-- END SOLUTION -->

# %% [markdown]
# ## Exercise 3.5 — Optimisers on a real network
#
# Below is a small NumPy MLP (the same design as in Lectures 2–3) for the digits dataset.
# Train it twice, for the same number of epochs, once with your `Momentum` (lr 0.05, μ 0.9)
# and once with your `Adam` (lr 1e-3), with mini-batches of 32 and He initialisation.
# Record the training loss per epoch in `history[name]` and the final test accuracy in
# `test_acc[name]`. Adam should reach at least **93 %** test accuracy.

# %%
digits = load_digits()
X_tr, X_te, y_tr, y_te = train_test_split(digits.data / 16.0, digits.target, test_size=0.25,
                                          random_state=0, stratify=digits.target)

def init_params(sizes, seed=0):
    r = np.random.default_rng(seed)
    params = []
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):
        params += [r.normal(0, np.sqrt(2 / n_in), size=(n_in, n_out)), np.zeros(n_out)]
    return params

def loss_and_grads(params, X, y):
    """Forward + backward pass of a ReLU MLP with softmax cross-entropy."""
    hs, h = [X], X
    n_layers = len(params) // 2
    for l in range(n_layers):
        z = h @ params[2 * l] + params[2 * l + 1]
        h = np.maximum(z, 0) if l < n_layers - 1 else z
        hs.append(h)
    Z = hs[-1] - hs[-1].max(1, keepdims=True)
    P = np.exp(Z); P /= P.sum(1, keepdims=True)
    loss = -np.mean(np.log(P[np.arange(len(y)), y] + 1e-12))
    dZ = P; dZ[np.arange(len(y)), y] -= 1; dZ /= len(y)
    grads = [None] * len(params)
    for l in reversed(range(n_layers)):
        grads[2 * l] = hs[l].T @ dZ
        grads[2 * l + 1] = dZ.sum(0)
        if l > 0:
            dZ = (dZ @ params[2 * l].T) * (hs[l] > 0)
    return loss, grads

def accuracy(params, X, y):
    h = X
    for l in range(len(params) // 2):
        z = h @ params[2 * l] + params[2 * l + 1]
        h = np.maximum(z, 0) if l < len(params) // 2 - 1 else z
    return np.mean(h.argmax(1) == y)

# %%
epochs, batch_size = 20, 32
history, test_acc = {}, {}
for name in ["momentum", "adam"]:
    ### BEGIN SOLUTION
    params = init_params((64, 128, 10))
    opt = Momentum(params, lr=0.05, mu=0.9) if name == "momentum" else Adam(params, lr=1e-3)
    r = np.random.default_rng(1)
    history[name] = []
    for epoch in range(epochs):
        idx = r.permutation(len(X_tr))
        losses = []
        for start in range(0, len(idx), batch_size):
            j = idx[start:start + batch_size]
            loss, grads = loss_and_grads(params, X_tr[j], y_tr[j])
            opt.step(grads)
            losses.append(loss)
        history[name].append(np.mean(losses))
    test_acc[name] = accuracy(params, X_te, y_te)
    ### END SOLUTION
    print(f"{name:9s} final training loss {history[name][-1]:.4f}, test accuracy {test_acc[name]:.3f}")

for name, h in history.items():
    plt.semilogy(h, label=name)
plt.xlabel("epoch"); plt.ylabel("training loss"); plt.legend(); plt.show()

# %%
assert len(history["adam"]) == epochs and len(history["momentum"]) == epochs
assert test_acc["adam"] >= 0.93
assert history["adam"][-1] < history["adam"][0]

# %% [markdown]
# ## Exercise 3.6 — Adam and L2 regularisation (pen and paper)
#
# Adding an L2 penalty $\tfrac{\lambda}{2}\lVert\theta\rVert^2$ to the loss adds $\lambda\theta$
# to the gradient. Explain why, under Adam, this is **not** the same as shrinking the weights by
# a constant factor each step ("weight decay"), whereas under plain SGD the two are equivalent.
# (This is the motivation for AdamW, which we meet in Lecture 5.)
#
# <!-- BEGIN SOLUTION -->
# With SGD, $\theta \leftarrow \theta - \eta(\mathbf g + \lambda\theta) = (1-\eta\lambda)\theta - \eta\mathbf g$,
# which is exactly multiplicative weight decay. With Adam the term $\lambda\theta$ enters $\mathbf m$
# and is then divided by $\sqrt{\hat{\mathbf v}}$, so each coordinate is decayed by a different,
# gradient-history-dependent amount: weights with large historical gradients are barely regularised.
# AdamW decouples the two by applying $\theta \leftarrow \theta - \eta\lambda\theta$ outside the
# adaptive update.
# <!-- END SOLUTION -->
