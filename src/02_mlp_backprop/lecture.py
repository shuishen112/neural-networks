# %% [markdown]
# # Lecture 2 — Multilayer Perceptrons and Backpropagation
#
# **Neural Networks · Master's course**
#
# Lecture 1 ended with a failure: no linear model can solve XOR. The fix is to
# *stack* linear models with nonlinearities in between, so that the early layers
# learn features and the last layer is a linear classifier on top of them. In this
# lecture we
#
# 1. define the **multilayer perceptron** (MLP) and the common activation functions,
# 2. build intuition for the **universal approximation** theorem,
# 3. derive **backpropagation** from the chain rule on a computational graph,
# 4. write a small, modular **NumPy deep-learning library** (layers with `forward` and
#    `backward`, a `Sequential` container) and **gradient-check** it,
# 5. train MLPs on XOR, two moons and handwritten digits, and
# 6. look inside the network at what the hidden units have learned.
#
# The code you write today is, in miniature, exactly what PyTorch's `autograd` does.
# Understanding it makes every later lecture easier.

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, load_digits
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(0)
plt.rcParams["figure.figsize"] = (5, 4)
plt.rcParams["axes.grid"] = True

# %% [markdown]
# ## 1. From one layer to many
#
# We keep the row convention of Lecture 1: a mini-batch is a matrix
# $X\in\mathbb{R}^{N\times d}$ with one example per row, and a layer with weights
# $W\in\mathbb{R}^{d_\text{in}\times d_\text{out}}$ computes $XW + \mathbf{b}$.
# An MLP with $L$ layers alternates **affine maps** and elementwise **activation
# functions** $\phi$:
#
# $$
# H^{(0)} = X, \qquad
# Z^{(l)} = H^{(l-1)} W^{(l)} + \mathbf{b}^{(l)}, \qquad
# H^{(l)} = \phi\big(Z^{(l)}\big), \qquad l = 1,\dots,L-1,
# $$
#
# and the last layer produces the scores $Z^{(L)} = H^{(L-1)}W^{(L)} + \mathbf{b}^{(L)}$
# which are fed to a softmax (classification) or used directly (regression).
# The $H^{(l)}$ are the **hidden representations**; their widths are hyperparameters.
#
# **Why the nonlinearity is essential.** Without $\phi$, the network collapses:
# $(XW^{(1)} + \mathbf b^{(1)})W^{(2)} + \mathbf b^{(2)} = X\underbrace{W^{(1)}W^{(2)}}_{W'} + \underbrace{\mathbf b^{(1)}W^{(2)} + \mathbf b^{(2)}}_{\mathbf b'}$
# is again a linear model, no matter how many layers we stack. All the extra
# expressive power comes from $\phi$.
#
# ## 2. Activation functions
#
# | Name | $\phi(z)$ | $\phi'(z)$ | Range |
# |---|---|---|---|
# | Sigmoid | $\sigma(z) = 1/(1+e^{-z})$ | $\sigma(z)\,(1-\sigma(z))$ | $(0,1)$ |
# | Tanh | $\tanh z$ | $1-\tanh^2 z$ | $(-1,1)$ |
# | ReLU | $\max(0,z)$ | $\mathbb{1}[z>0]$ | $[0,\infty)$ |
# | Leaky ReLU | $\max(\alpha z, z)$ | $1$ if $z>0$ else $\alpha$ | $\mathbb{R}$ |
#
# Note that $\tanh z = 2\sigma(2z) - 1$: tanh is a rescaled, zero-centred sigmoid.

# %%
z = np.linspace(-5, 5, 400)
sig = 1 / (1 + np.exp(-z))
acts = {
    "sigmoid":    (sig,                    sig * (1 - sig)),
    "tanh":       (np.tanh(z),             1 - np.tanh(z) ** 2),
    "ReLU":       (np.maximum(0, z),       (z > 0).astype(float)),
    "leaky ReLU": (np.where(z > 0, z, 0.1 * z), np.where(z > 0, 1.0, 0.1)),
}
fig, axes = plt.subplots(1, 4, figsize=(15, 3.2), sharex=True)
for ax, (name, (f, df)) in zip(axes, acts.items()):
    ax.plot(z, f, label=r"$\phi(z)$")
    ax.plot(z, df, "--", label=r"$\phi'(z)$")
    ax.set_title(name); ax.set_ylim(-1.5, 2.5); ax.set_xlabel("$z$")
axes[0].legend(loc="upper left")
plt.tight_layout(); plt.show()

# %% [markdown]
# Things to notice:
#
# * **Saturation.** For $|z|\gtrsim 4$ the sigmoid and tanh are flat: $\phi'(z)\approx 0$.
#   Gradients flowing backwards are multiplied by $\phi'$ at every layer, so saturated
#   units pass almost no learning signal. The sigmoid's derivative never exceeds
#   $1/4$, so even unsaturated sigmoid networks shrink gradients by $\ge 4\times$ per layer.
# * **Zero-centring.** Sigmoid outputs are always positive, which makes the gradients
#   of all weights entering the next unit share a sign; tanh does not have this problem.
# * **ReLU** does not saturate for $z>0$, is extremely cheap, and its derivative is
#   exactly $1$ on the active side. Its downside is that a unit whose pre-activation is
#   negative for every input receives no gradient at all (a "dead" unit) — leaky ReLU
#   and its relatives (ELU, GELU, SiLU) soften this. ReLU-type activations are the
#   default in modern networks (Glorot, Bordes & Bengio, 2011; Nair & Hinton, 2010).
#
# ## 3. Universal approximation
#
# **Theorem (Cybenko, 1989; Hornik, 1991; Leshno et al., 1993).** Let $\phi$ be any
# continuous non-polynomial function. For every continuous $f$ on a compact set
# $K\subset\mathbb{R}^d$ and every $\varepsilon>0$ there is a one-hidden-layer network
# $g(\mathbf{x}) = \sum_{j=1}^m c_j\,\phi(\mathbf{w}_j^\top\mathbf{x} + b_j)$ with
# $\sup_{\mathbf x\in K}|f(\mathbf{x}) - g(\mathbf{x})| < \varepsilon$.
#
# In one dimension the idea is easy to see with ReLUs. A sum
# $g(x) = c_0 + \sum_j c_j\,\mathrm{ReLU}(x - t_j)$ is a **piecewise-linear** function with
# kinks at the "knots" $t_j$; the coefficient $c_j$ is the *change of slope* at $t_j$.
# Any continuous function can be approximated by a piecewise-linear interpolant with
# enough knots, so it can be approximated by a wide enough ReLU network. We can even
# write the weights down directly — no training needed:

# %%
def target(x):
    return np.sin(3 * x) + 0.3 * x ** 2

def relu_interpolant(f, knots):
    """One-hidden-layer ReLU net that linearly interpolates f at the knots."""
    vals = f(knots)
    slopes = np.diff(vals) / np.diff(knots)            # slope on each segment
    c = np.r_[slopes[0], np.diff(slopes)]              # slope changes = output weights
    t = knots[:-1]                                     # hidden-unit thresholds (biases = -t)
    return lambda x: vals[0] + np.maximum(0, x[:, None] - t[None, :]) @ c

xs = np.linspace(-2, 2, 500)
fig, axes = plt.subplots(1, 4, figsize=(15, 3.2), sharey=True)
for ax, m in zip(axes, [3, 6, 12, 48]):
    g = relu_interpolant(target, np.linspace(-2, 2, m + 1))
    err = np.max(np.abs(g(xs) - target(xs)))
    ax.plot(xs, target(xs), "k", lw=3, alpha=0.3, label="$f$")
    ax.plot(xs, g(xs), "C3", label="ReLU net")
    ax.set_title(f"{m} hidden units, max error {err:.3f}")
axes[0].legend(); plt.tight_layout(); plt.show()

# %% [markdown]
# The error of a piecewise-linear interpolant with spacing $h$ is $O(h^2\,\max|f''|)$,
# so here it shrinks by roughly $4\times$ each time we double the width.
#
# Two important caveats:
#
# * The theorem is about **existence**. It says nothing about whether gradient
#   descent will *find* those weights, or how many examples are needed to do so.
# * The width required by the constructive argument grows **exponentially** with the
#   input dimension (to tile $[0,1]^d$ with cells of side $h$ we need $h^{-d}$ cells).
#   Depth helps: there are functions that a deep network represents with polynomially
#   many units but a shallow one needs exponentially many (Telgarsky, 2016; Eldan &
#   Shamir, 2016). This is one theoretical argument for *deep* learning.
#
# We will come back to this example and *learn* the weights once we have backpropagation.
#
# ## 4. Computational graphs and the chain rule
#
# To train an MLP with gradient descent we need $\partial\mathcal L/\partial\theta$ for
# every parameter. Writing out these derivatives by hand for every architecture would
# be hopeless. Instead we break the computation into a **graph** of elementary
# operations, each of which knows its own local derivative, and combine them with
# the chain rule.
#
# Consider a single sigmoid neuron with squared error, $\mathcal L = (\sigma(wx+b) - y)^2$:
#
# ```
#  x ──┐
#      (×)── u ──(+)── z ──(σ)── a ──(−y)²── L
#  w ──┘          │
#            b ───┘
# ```
#
# **Forward pass:** evaluate the nodes left to right and *store* the intermediate values.
# **Backward pass:** walk right to left, carrying the "upstream gradient"
# $\bar v := \partial\mathcal L/\partial v$ for every node $v$ (often called the
# *adjoint* of $v$). At each node multiply by the local derivative:
#
# $$
# \bar L = 1,\quad
# \bar a = 2(a-y),\quad
# \bar z = \bar a\,\sigma(z)(1-\sigma(z)),\quad
# \bar u = \bar z,\quad \bar b = \bar z,\quad
# \bar w = \bar u\, x,\quad \bar x = \bar u\, w .
# $$

# %%
x, w, b, y_t = 1.5, -0.8, 0.4, 1.0
# forward
u = w * x
z_ = u + b
a = 1 / (1 + np.exp(-z_))
L = (a - y_t) ** 2
# backward
dL_da = 2 * (a - y_t)
dL_dz = dL_da * a * (1 - a)
dL_du, dL_db = dL_dz, dL_dz
dL_dw, dL_dx = dL_du * x, dL_du * w
print(f"forward : u={u:.3f}  z={z_:.3f}  a={a:.4f}  L={L:.4f}")
print(f"backward: dL/dw={dL_dw:.5f}  dL/db={dL_db:.5f}  dL/dx={dL_dx:.5f}")

# Check against central finite differences
f_L = lambda w, b: (1 / (1 + np.exp(-(w * x + b))) - y_t) ** 2
h = 1e-6
print(f"numeric : dL/dw={(f_L(w + h, b) - f_L(w - h, b)) / (2 * h):.5f}  "
      f"dL/db={(f_L(w, b + h) - f_L(w, b - h)) / (2 * h):.5f}")

# %% [markdown]
# Two rules cover every graph:
#
# * **Chain rule along a path:** if $v$ feeds into $u$, then $\bar v \mathrel{+}= \bar u\,\frac{\partial u}{\partial v}$.
# * **Sum over paths:** if $v$ feeds into several nodes, the contributions are *added*
#   (the multivariate chain rule). This is why gradients are accumulated with `+=`.
#
# **Why "reverse mode"?** We could also propagate derivatives forwards (forward-mode
# differentiation), but that costs one pass *per input*. A loss is a single scalar
# depending on millions of parameters, so the reverse sweep — one backward pass for
# *all* parameters at a cost of a small constant times the forward pass — is the
# right choice. This is **backpropagation** (Linnainmaa, 1970; Werbos, 1974;
# Rumelhart, Hinton & Williams, 1986): reverse-mode automatic differentiation applied
# to neural networks. The price is memory: all forward intermediates must be stored
# until the backward pass uses them.
#
# ### 4.1 Backpropagation through layers (matrix form)
#
# For vector-valued nodes, the local derivative is a Jacobian, but we never form it
# explicitly — we only need *vector–Jacobian products*. With upstream gradient
# $\bar Y = \partial\mathcal L/\partial Y$ of the same shape as $Y$:
#
# **Affine layer** $Y = XW + \mathbf b$, with $X\in\mathbb R^{N\times d_\text{in}}$.
# Since $Y_{nk} = \sum_i X_{ni}W_{ik} + b_k$,
#
# $$
# \bar W_{ik} = \sum_n \bar Y_{nk} X_{ni}
# \;\Rightarrow\; \bar W = X^\top \bar Y, \qquad
# \bar{\mathbf b} = \sum_n \bar Y_{n,:}, \qquad
# \bar X = \bar Y\, W^\top .
# $$
#
# A useful sanity check: each gradient has the **same shape** as the quantity it
# belongs to, and there is essentially only one way to multiply the matrices to get that shape.
#
# **Elementwise activation** $H = \phi(Z)$: the Jacobian is diagonal, so
# $\bar Z = \bar H \odot \phi'(Z)$.
#
# **Softmax + cross-entropy** $\mathcal L = -\frac1N\sum_n \log P_{n,y_n}$, $P = \operatorname{softmax}(Z)$.
# As in Lecture 1 the combined gradient is $\bar Z = \frac1N(P - Y)$ with $Y$ one-hot.
# Implementing the two as *one* layer is both simpler and numerically more stable.
#
# Putting it together for an MLP, with $\delta^{(l)} := \bar Z^{(l)}$:
#
# $$
# \delta^{(L)} = \tfrac1N(P-Y),\qquad
# \bar W^{(l)} = H^{(l-1)\top}\delta^{(l)},\qquad
# \bar{\mathbf b}^{(l)} = \mathbf 1^\top \delta^{(l)},\qquad
# \delta^{(l-1)} = \big(\delta^{(l)} W^{(l)\top}\big)\odot\phi'\big(Z^{(l-1)}\big).
# $$
#
# The last formula shows the danger already mentioned: the error signal is multiplied
# by $W^{(l)\top}$ and by $\phi'$ at every layer, so it can shrink or grow
# exponentially with depth. Lecture 3 deals with this.
#
# ## 5. A modular NumPy implementation
#
# Every layer gets two methods:
#
# * `forward(X)` computes the output and caches whatever the backward pass will need;
# * `backward(dY)` receives $\bar Y$, stores gradients of its own parameters, and
#   returns $\bar X$ for the layer below.
#
# A `Sequential` container chains them. This is the same design as `torch.nn`.

# %%
class Module:
    """Base class: a layer with (possibly empty) lists of parameters and gradients."""
    def params(self):
        return []
    def grads(self):
        return []
    def __call__(self, X):
        return self.forward(X)


class Linear(Module):
    def __init__(self, n_in, n_out, rng):
        # Scaled Gaussian initialisation; Lecture 3 explains the sqrt(2 / n_in).
        self.W = rng.normal(0.0, np.sqrt(2.0 / n_in), size=(n_in, n_out))
        self.b = np.zeros(n_out)
        self.dW, self.db = np.zeros_like(self.W), np.zeros_like(self.b)

    def forward(self, X):
        self.X = X
        return X @ self.W + self.b

    def backward(self, dY):
        self.dW[...] = self.X.T @ dY
        self.db[...] = dY.sum(axis=0)
        return dY @ self.W.T

    def params(self):
        return [self.W, self.b]
    def grads(self):
        return [self.dW, self.db]


class ReLU(Module):
    def forward(self, X):
        self.mask = X > 0
        return X * self.mask
    def backward(self, dY):
        return dY * self.mask


class Tanh(Module):
    def forward(self, X):
        self.H = np.tanh(X)
        return self.H
    def backward(self, dY):
        return dY * (1 - self.H ** 2)


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = list(layers)
    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X
    def backward(self, dY):
        for layer in reversed(self.layers):
            dY = layer.backward(dY)
        return dY
    def params(self):
        return [p for l in self.layers for p in l.params()]
    def grads(self):
        return [g for l in self.layers for g in l.grads()]

# %% [markdown]
# Losses are layers too, except that `forward` takes the targets and returns a
# scalar, and `backward` takes no argument (the upstream gradient of the loss is $1$).

# %%
class SoftmaxCrossEntropy:
    def forward(self, Z, y):
        Z = Z - Z.max(axis=1, keepdims=True)                 # stability shift
        logP = Z - np.log(np.exp(Z).sum(axis=1, keepdims=True))
        self.P, self.y = np.exp(logP), y
        return -np.mean(logP[np.arange(len(y)), y])
    def backward(self):
        G = self.P.copy()
        G[np.arange(len(self.y)), self.y] -= 1
        return G / len(self.y)


class MSE:
    """L = 1/(2N) * sum ||pred - target||^2  (the 1/2 makes the gradient clean)."""
    def forward(self, pred, target):
        self.diff = pred - target
        return 0.5 * np.mean(np.sum(self.diff ** 2, axis=1))
    def backward(self):
        return self.diff / len(self.diff)


def sgd_step(model, lr):
    for p, g in zip(model.params(), model.grads()):
        p -= lr * g        # in-place, so the layers see the update


def train(model, loss_fn, X, y, lr=0.1, epochs=100, batch_size=32, seed=0):
    """Mini-batch SGD. Returns the training loss after every epoch."""
    r = np.random.default_rng(seed)
    history = []
    for _ in range(epochs):
        idx = r.permutation(len(X))
        for s in range(0, len(X), batch_size):
            j = idx[s:s + batch_size]
            loss_fn.forward(model.forward(X[j]), y[j])
            model.backward(loss_fn.backward())
            sgd_step(model, lr)
        history.append(loss_fn.forward(model.forward(X), y))
    return history


def accuracy(model, X, y):
    return np.mean(model.forward(X).argmax(axis=1) == y)

# %% [markdown]
# ## 6. Gradient checking the whole network
#
# A bug in a backward pass rarely crashes anything — the network just trains badly.
# So before trusting our implementation we compare **every** analytical gradient with
# a central finite difference
#
# $$\frac{\partial\mathcal L}{\partial\theta_j}\approx\frac{\mathcal L(\theta+h\mathbf e_j)-\mathcal L(\theta-h\mathbf e_j)}{2h},$$
#
# whose error is $O(h^2)$. We report the **relative error**
# $\lVert g_\text{num}-g_\text{ana}\rVert / \lVert g_\text{num}+g_\text{ana}\rVert$; in double
# precision anything below $\sim10^{-7}$ means the gradient is correct.
#
# *Caveat:* ReLU is not differentiable at $0$. If a perturbation of size $h$ moves a
# pre-activation across the kink, the numerical estimate is wrong even though the code
# is right. With random real-valued data this is rare; using tanh for the check avoids it entirely.

# %%
def grad_check(model, loss_fn, X, y, h=1e-6):
    loss_fn.forward(model.forward(X), y)
    model.backward(loss_fn.backward())
    analytic = np.concatenate([g.ravel() for g in model.grads()])
    numeric = []
    for p in model.params():
        for i in np.ndindex(p.shape):
            old = p[i]
            p[i] = old + h; Lp = loss_fn.forward(model.forward(X), y)
            p[i] = old - h; Lm = loss_fn.forward(model.forward(X), y)
            p[i] = old
            numeric.append((Lp - Lm) / (2 * h))
    numeric = np.array(numeric)
    return np.linalg.norm(numeric - analytic) / np.linalg.norm(numeric + analytic)

r = np.random.default_rng(1)
Xg, yg = r.normal(size=(8, 5)), r.integers(0, 3, size=8)
for act in (Tanh, ReLU):
    net = Sequential(Linear(5, 7, r), act(), Linear(7, 6, r), act(), Linear(6, 3, r))
    n_params = sum(p.size for p in net.params())
    print(f"{act.__name__:5s} net, {n_params} parameters: relative error "
          f"{grad_check(net, SoftmaxCrossEntropy(), Xg, yg):.2e}")

# %% [markdown]
# ## 7. Solving XOR and two moons
#
# Let us return to the noisy XOR problem from Lecture 1, where logistic regression
# was stuck at chance level, and to the two-moons dataset. A helper plots the predicted
# probability of class 1 over the input plane.

# %%
def plot_decision(model, X, y, ax=None, title=None):
    ax = ax or plt.gca()
    pad = 0.5
    g1, g2 = np.meshgrid(np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 200),
                         np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 200))
    Z = model.forward(np.c_[g1.ravel(), g2.ravel()])
    Z = Z - Z.max(axis=1, keepdims=True)
    P = np.exp(Z) / np.exp(Z).sum(axis=1, keepdims=True)
    ax.contourf(g1, g2, P[:, 1].reshape(g1.shape), levels=20, cmap="RdBu_r", alpha=0.6, vmin=0, vmax=1)
    ax.contour(g1, g2, P[:, 1].reshape(g1.shape), levels=[0.5], colors="k")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", s=10, edgecolor="k", linewidth=0.3)
    if title:
        ax.set_title(title)

Xx = rng.uniform(-1, 1, size=(400, 2))
yx = ((Xx[:, 0] > 0) ^ (Xx[:, 1] > 0)).astype(int)
Xm, ym = make_moons(n_samples=400, noise=0.15, random_state=0)

r = np.random.default_rng(0)
linear_xor = Sequential(Linear(2, 2, r))                       # softmax regression
mlp_xor = Sequential(Linear(2, 16, r), ReLU(), Linear(16, 16, r), ReLU(), Linear(16, 2, r))
linear_moons = Sequential(Linear(2, 2, r))
mlp_moons = Sequential(Linear(2, 32, r), ReLU(), Linear(32, 32, r), ReLU(), Linear(32, 2, r))

hists = {}
for name, model, (X_, y_) in [("linear / XOR", linear_xor, (Xx, yx)), ("MLP / XOR", mlp_xor, (Xx, yx)),
                              ("linear / moons", linear_moons, (Xm, ym)), ("MLP / moons", mlp_moons, (Xm, ym))]:
    hists[name] = train(model, SoftmaxCrossEntropy(), X_, y_, lr=0.1, epochs=150, batch_size=32)
    print(f"{name:15s} training accuracy {accuracy(model, X_, y_):.3f}")

fig, axes = plt.subplots(1, 5, figsize=(19, 3.6))
for ax, (name, model, (X_, y_)) in zip(axes, [("linear / XOR", linear_xor, (Xx, yx)), ("MLP / XOR", mlp_xor, (Xx, yx)),
                                              ("linear / moons", linear_moons, (Xm, ym)), ("MLP / moons", mlp_moons, (Xm, ym))]):
    plot_decision(model, X_, y_, ax, name)
for name, h_ in hists.items():
    axes[4].plot(h_, label=name)
axes[4].set_xlabel("epoch"); axes[4].set_ylabel("training loss"); axes[4].legend(fontsize=8)
plt.tight_layout(); plt.show()

# %% [markdown]
# The two-hidden-layer MLP carves out the four XOR quadrants and follows the curved
# moons, while the linear model can only draw a straight line. Notice that the MLP's
# boundary is made of (slightly rounded) straight segments: a ReLU network computes a
# **piecewise-linear** function of its input, and the boundary is a union of pieces of
# hyperplanes, one for each "activation pattern" of the hidden units.
#
# ### 7.1 Learning the approximation from Section 3
#
# Now that we can train networks, let us fit $f(x) = \sin 3x + 0.3x^2$ from 200 noisy
# samples with one hidden layer of increasing width, using the squared-error loss.

# %%
r = np.random.default_rng(0)
x1 = np.sort(r.uniform(-2, 2, size=200))[:, None]
y1 = target(x1) + 0.05 * r.normal(size=x1.shape)     # noise variance 0.0025

fig, axes = plt.subplots(1, 4, figsize=(15, 3.2), sharey=True)
for ax, m in zip(axes, [1, 3, 10, 30]):
    r = np.random.default_rng(m)
    net = Sequential(Linear(1, m, r), Tanh(), Linear(m, 1, r))
    h_ = train(net, MSE(), x1, y1, lr=0.05, epochs=3000, batch_size=20, seed=m)
    ax.scatter(x1, y1, s=5, c="gray")
    ax.plot(xs, net.forward(xs[:, None]), "C3", lw=2)
    ax.set_title(f"width {m}: MSE {2 * h_[-1]:.4f}")
plt.tight_layout(); plt.show()

# %% [markdown]
# With one tanh unit the network can only produce a single "step"; with three it
# captures the gross shape; with ten or more it comes close to the noise level
# ($\sigma^2 = 0.0025$). Wider networks are not automatically better *trained*,
# though: they need more steps and a suitable learning rate (Lecture 3).
# Unlike the hand-built interpolant, the trained network places its units where they
# are needed.
#
# ## 8. Handwritten digits
#
# On the $8\times8$ digits, softmax regression reached roughly 96 % test accuracy in
# Lecture 1. A one-hidden-layer MLP does better.

# %%
digits = load_digits()
Xd = digits.data / 16.0
yd = digits.target
Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(Xd, yd, test_size=0.25, random_state=0, stratify=yd)

r = np.random.default_rng(0)
mlp_digits = Sequential(Linear(64, 128, r), ReLU(), Linear(128, 10, r))
loss_fn = SoftmaxCrossEntropy()
tr_loss, te_acc = [], []
for epoch in range(40):
    tr_loss += train(mlp_digits, loss_fn, Xd_tr, yd_tr, lr=0.1, epochs=1, batch_size=32, seed=epoch)
    te_acc.append(accuracy(mlp_digits, Xd_te, yd_te))
print(f"train accuracy {accuracy(mlp_digits, Xd_tr, yd_tr):.3f} | test accuracy {te_acc[-1]:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].plot(tr_loss); axes[0].set_xlabel("epoch"); axes[0].set_ylabel("training loss")
axes[1].plot(te_acc); axes[1].set_xlabel("epoch"); axes[1].set_ylabel("test accuracy")
plt.tight_layout(); plt.show()

# %% [markdown]
# Let us look at the mistakes the network still makes.

# %%
pred = mlp_digits.forward(Xd_te).argmax(1)
wrong = np.where(pred != yd_te)[0][:10]
fig, axes = plt.subplots(1, len(wrong), figsize=(1.1 * len(wrong), 1.6))
for ax, i in zip(np.atleast_1d(axes), wrong):
    ax.imshow(Xd_te[i].reshape(8, 8), cmap="gray_r")
    ax.set_title(f"{yd_te[i]}→{pred[i]}", fontsize=9); ax.axis("off")
plt.suptitle("Misclassified test digits (true → predicted)", y=1.15); plt.show()

# %% [markdown]
# ## 9. What do the hidden units learn?
#
# ### 9.1 Hidden units as feature detectors in the input plane
#
# A first-layer unit computes $\phi(\mathbf w_j^\top\mathbf x + b_j)$: it responds to one
# side of a line in the input plane. The output layer then combines these "soft half-planes".
# We train a small tanh network with 8 hidden units on the moons and plot each unit's activation.

# %%
r = np.random.default_rng(3)
small = Sequential(Linear(2, 8, r), Tanh(), Linear(8, 2, r))
train(small, SoftmaxCrossEntropy(), Xm, ym, lr=0.2, epochs=300, batch_size=32)
print(f"8-unit tanh network, training accuracy {accuracy(small, Xm, ym):.3f}")

g1, g2 = np.meshgrid(np.linspace(-2, 3, 150), np.linspace(-1.5, 2, 150))
G = np.c_[g1.ravel(), g2.ravel()]
H = np.tanh(small.layers[0].forward(G))
w_out = small.layers[2].W[:, 1] - small.layers[2].W[:, 0]   # contribution to the class-1 logit margin
fig, axes = plt.subplots(2, 5, figsize=(16, 6))
for j, ax in enumerate(axes.ravel()[:8]):
    ax.contourf(g1, g2, H[:, j].reshape(g1.shape), levels=20, cmap="PuOr", vmin=-1, vmax=1)
    ax.scatter(Xm[:, 0], Xm[:, 1], c=ym, cmap="RdBu_r", s=4)
    ax.set_title(f"unit {j}  (out weight {w_out[j]:+.1f})", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
plot_decision(small, Xm, ym, axes[1, 3], "network output")
axes[1, 4].bar(range(8), w_out); axes[1, 4].set_title("output weights (class 1 − class 0)", fontsize=9)
plt.tight_layout(); plt.show()

# %% [markdown]
# ### 9.2 Hidden layers *transform the space*
#
# Another view: the last hidden layer is a **learned feature space** in which the
# classes have become linearly separable — the output layer is just logistic regression.
# We force the last hidden layer to have only two units so that we can plot it.

# %%
r = np.random.default_rng(0)
bottleneck = Sequential(Linear(2, 16, r), Tanh(), Linear(16, 16, r), Tanh(),
                        Linear(16, 2, r), Tanh(), Linear(2, 2, r))
train(bottleneck, SoftmaxCrossEntropy(), Xm, ym, lr=0.2, epochs=300, batch_size=32)
print(f"bottleneck network, training accuracy {accuracy(bottleneck, Xm, ym):.3f}")

hidden = Sequential(*bottleneck.layers[:-1]).forward(Xm)   # output of the 2-unit layer
Wo, bo = bottleneck.layers[-1].W, bottleneck.layers[-1].b
w_eff, b_eff = Wo[:, 1] - Wo[:, 0], bo[1] - bo[0]          # decision line: w_eff · h + b_eff = 0

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].scatter(Xm[:, 0], Xm[:, 1], c=ym, cmap="RdBu_r", s=10); axes[0].set_title("input space")
axes[1].scatter(hidden[:, 0], hidden[:, 1], c=ym, cmap="RdBu_r", s=10)
hs = np.linspace(-1.05, 1.05, 50)
axes[1].plot(hs, -(w_eff[0] * hs + b_eff) / w_eff[1], "k")
axes[1].set_xlim(-1.1, 1.1); axes[1].set_ylim(-1.1, 1.1)
axes[1].set_title("last hidden layer: linearly separable")
plt.tight_layout(); plt.show()

# %% [markdown]
# ### 9.3 Hidden units on digits
#
# Each first-layer unit of the digits network has a weight vector in $\mathbb R^{64}$
# that we can display as an $8\times 8$ image. Unlike the ten smooth class templates of
# softmax regression in Lecture 1, these look noisy: with about 1 350 training images and
# $64\times128$ weights, much of the random initialisation survives, and each unit
# only needs to be *useful in combination* with the others (a **distributed
# representation**). A more readable picture is obtained by asking which inputs a unit
# responds to: below, the second row shows the average of the 30 training digits that
# activate each unit most strongly, and the label shows the most common class among them.

# %%
W1 = mlp_digits.layers[0].W
H_tr = np.maximum(0, Xd_tr @ W1 + mlp_digits.layers[0].b)      # hidden activations, (N, 128)
units = np.argsort(-H_tr.var(axis=0))[:10]                      # the 10 most variable units
fig, axes = plt.subplots(2, 10, figsize=(12, 3))
for col, j in enumerate(units):
    lim = np.abs(W1[:, j]).max()
    axes[0, col].imshow(W1[:, j].reshape(8, 8), cmap="RdBu_r", vmin=-lim, vmax=lim)
    top = np.argsort(-H_tr[:, j])[:30]
    axes[1, col].imshow(Xd_tr[top].mean(0).reshape(8, 8), cmap="gray_r")
    counts = np.bincount(yd_tr[top], minlength=10)
    axes[1, col].set_title(f"{counts.argmax()} ({counts.max()}/30)", fontsize=8)
    axes[0, col].set_title(f"unit {j}", fontsize=8)
for ax in axes.ravel():
    ax.axis("off")
axes[0, 0].text(-4, 3.5, "weights", rotation=90, va="center", fontsize=9)
axes[1, 0].text(-4, 3.5, "top-30 avg", rotation=90, va="center", fontsize=9)
plt.tight_layout(); plt.show()

# %% [markdown]
# Some units are nearly class-specific (all of their top inputs are one digit),
# others respond to a part shared by several classes — for example a vertical stroke
# common to 1, 4 and 7. The output layer combines these features linearly. Deeper
# networks and convolutional layers (later in the course) make such features far more structured.

# %% [markdown]
# ## Summary
#
# | Concept | Key formula / idea |
# |---|---|
# | MLP layer | $H^{(l)} = \phi(H^{(l-1)}W^{(l)} + \mathbf b^{(l)})$ |
# | Why nonlinearity | composition of affine maps is affine |
# | Activations | sigmoid/tanh saturate; ReLU: $\phi'=\mathbb 1[z>0]$, cheap, default choice |
# | Universal approximation | one wide hidden layer suffices *in principle*; depth can be exponentially more efficient |
# | Backpropagation | reverse-mode autodiff: $\bar v = \sum_{u\in\text{children}(v)} \bar u\,\partial u/\partial v$ |
# | Affine backward | $\bar W = X^\top\bar Y,\ \bar{\mathbf b} = \mathbf 1^\top\bar Y,\ \bar X = \bar Y W^\top$ |
# | Activation backward | $\bar Z = \bar H\odot\phi'(Z)$ |
# | Softmax + CE backward | $\bar Z = (P - Y)/N$ |
# | Gradient check | central differences, relative error $<10^{-7}$ |
# | Hidden layers | learn features that make the problem linearly separable for the output layer |
#
# **Further reading.**
# Goodfellow, Bengio & Courville, *Deep Learning*, ch. 6 (especially §6.5 on back-propagation);
# Rumelhart, Hinton & Williams (1986), "Learning representations by back-propagating errors", *Nature*;
# Cybenko (1989), "Approximation by superpositions of a sigmoidal function";
# Telgarsky (2016), "Benefits of depth in neural networks";
# Baydin et al. (2018), "Automatic differentiation in machine learning: a survey", *JMLR*;
# Karpathy's *micrograd* and the CS231n notes on backpropagation for a hands-on view.
