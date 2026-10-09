# %% [markdown]
# # Lecture 1 — Linear Models and the Perceptron
#
# **Neural Networks · Master's course**
#
# Every neural network is built from one simple ingredient: a weighted sum of
# inputs followed by a nonlinearity. In this lecture we study that ingredient
# on its own. We will
#
# 1. frame supervised learning as minimising a loss over parameters,
# 2. implement Rosenblatt's **perceptron** and its learning rule,
# 3. derive and implement **logistic regression** with gradient descent,
# 4. extend it to **softmax regression** for multi-class problems, and
# 5. see where linear models break down — the motivation for hidden layers.
#
# Everything is written in plain NumPy so that nothing is hidden from you.
# From Lecture 4 onwards we will let PyTorch do the bookkeeping.

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, load_digits
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(0)
plt.rcParams["figure.figsize"] = (5, 4)
plt.rcParams["axes.grid"] = True

# %% [markdown]
# ## 1. Supervised learning in one paragraph
#
# We are given a dataset $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$ with
# inputs $\mathbf{x}_i \in \mathbb{R}^d$ and targets $y_i$. We choose a
# parametric model $f_\theta$ and a loss $\ell$ that measures how wrong a
# prediction is, and we look for
#
# $$\theta^\star = \arg\min_\theta \; \frac{1}{N}\sum_{i=1}^N \ell\big(f_\theta(\mathbf{x}_i), y_i\big).$$
#
# A **linear model** uses $f_\theta(\mathbf{x}) = \mathbf{w}^\top \mathbf{x} + b$
# with $\theta = (\mathbf{w}, b)$. The decision boundary
# $\{\mathbf{x} : \mathbf{w}^\top\mathbf{x} + b = 0\}$ is a hyperplane: $\mathbf{w}$
# is its normal vector and $-b/\lVert\mathbf{w}\rVert$ its signed distance from the origin.
#
# Let us create a small two-class problem to play with.

# %%
X, y = make_blobs(n_samples=200, centers=[(-2.5, -1.5), (2.5, 2)], cluster_std=1.0, random_state=1)
y_pm = 2 * y - 1  # labels in {-1, +1} for the perceptron

def plot_data(X, y, ax=None):
    ax = ax or plt.gca()
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=15, label="class 0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=15, label="class 1")
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    return ax

def plot_boundary(w, b, X, ax=None, **kw):
    ax = ax or plt.gca()
    xs = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
    # w1 x1 + w2 x2 + b = 0  ->  x2 = -(w1 x1 + b) / w2
    ax.plot(xs, -(w[0] * xs + b) / w[1], **kw)
    ax.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)

plot_data(X, y); plt.legend(); plt.title("A linearly separable dataset"); plt.show()

# %% [markdown]
# ## 2. The perceptron (Rosenblatt, 1958)
#
# The perceptron predicts $\hat y = \operatorname{sign}(\mathbf{w}^\top\mathbf{x} + b)$
# with labels $y \in \{-1, +1\}$. It visits the examples one at a time and,
# **only when it makes a mistake**, nudges the hyperplane towards the misclassified point:
#
# $$\text{if } y_i(\mathbf{w}^\top\mathbf{x}_i + b) \le 0: \qquad
# \mathbf{w} \leftarrow \mathbf{w} + \eta\, y_i \mathbf{x}_i, \qquad b \leftarrow b + \eta\, y_i .$$
#
# **Convergence theorem (Novikoff, 1962).** If the data are linearly separable
# with margin $\gamma$ and $\lVert\mathbf{x}_i\rVert \le R$, the perceptron makes
# at most $(R/\gamma)^2$ mistakes — independently of the number of examples.

# %%
def train_perceptron(X, y, lr=1.0, epochs=20, seed=0):
    """Perceptron learning rule. Labels must be in {-1, +1}."""
    r = np.random.default_rng(seed)
    w, b = np.zeros(X.shape[1]), 0.0
    history = []
    for epoch in range(epochs):
        mistakes = 0
        for i in r.permutation(len(X)):
            if y[i] * (X[i] @ w + b) <= 0:
                w += lr * y[i] * X[i]
                b += lr * y[i]
                mistakes += 1
        history.append(mistakes)
        if mistakes == 0:
            break
    return w, b, history

w_p, b_p, hist = train_perceptron(X, y_pm)
print("mistakes per epoch:", hist)
acc = np.mean(np.sign(X @ w_p + b_p) == y_pm)
print(f"training accuracy: {acc:.3f}")

plot_data(X, y); plot_boundary(w_p, b_p, X, color="k", label="perceptron")
plt.legend(); plt.title("Perceptron decision boundary"); plt.show()

# %% [markdown]
# The perceptron stops as soon as it finds *any* separating hyperplane. That
# hyperplane is often close to some of the points — the algorithm has no notion
# of a "good" boundary, and if the data are **not** separable it never converges.
# Both problems are fixed by replacing the hard mistake-counting rule with a
# smooth, probabilistic loss.

# %% [markdown]
# ## 3. Logistic regression
#
# Model the probability of class 1 with the **sigmoid** (logistic) function
#
# $$p(y=1\mid\mathbf{x}) = \sigma(z) = \frac{1}{1+e^{-z}}, \qquad z = \mathbf{w}^\top\mathbf{x} + b .$$
#
# Maximising the likelihood of the labels $y_i\in\{0,1\}$ is equivalent to minimising the
# **binary cross-entropy**
#
# $$\mathcal{L}(\mathbf{w}, b) = -\frac{1}{N}\sum_i \Big[y_i \log \hat p_i + (1-y_i)\log(1-\hat p_i)\Big].$$
#
# Using $\sigma'(z) = \sigma(z)(1-\sigma(z))$ the gradient collapses to a remarkably simple form:
#
# $$\frac{\partial \mathcal{L}}{\partial z_i} = \frac{1}{N}(\hat p_i - y_i), \qquad
# \nabla_{\mathbf{w}}\mathcal{L} = \frac{1}{N} X^\top(\hat{\mathbf p} - \mathbf{y}), \qquad
# \frac{\partial\mathcal{L}}{\partial b} = \frac{1}{N}\sum_i(\hat p_i - y_i).$$
#
# "Prediction minus target" will show up again and again — it is also the
# gradient of softmax + cross-entropy and of the squared error for a linear output.

# %%
def sigmoid(z):
    # Numerically stable: never exponentiate a large positive number.
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out

def bce(p, y, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def train_logreg(X, y, lr=0.1, steps=500):
    w, b = np.zeros(X.shape[1]), 0.0
    losses = []
    for _ in range(steps):
        p = sigmoid(X @ w + b)
        losses.append(bce(p, y))
        grad_z = (p - y) / len(y)
        w -= lr * X.T @ grad_z
        b -= lr * grad_z.sum()
    return w, b, losses

w_l, b_l, losses = train_logreg(X, y)
print(f"final loss {losses[-1]:.4f}, accuracy {np.mean((sigmoid(X @ w_l + b_l) > 0.5) == y):.3f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(losses); axes[0].set_xlabel("step"); axes[0].set_ylabel("BCE"); axes[0].set_title("Training loss")
plot_data(X, y, axes[1])
plot_boundary(w_p, b_p, X, axes[1], color="gray", ls="--", label="perceptron")
plot_boundary(w_l, b_l, X, axes[1], color="k", label="logistic regression")
axes[1].legend(); axes[1].set_title("Decision boundaries")
plt.tight_layout(); plt.show()

# %% [markdown]
# ### Visualising the probabilities
#
# Logistic regression gives more than a boundary: it gives a calibrated-looking
# probability that changes smoothly as we move away from the hyperplane. The
# larger $\lVert\mathbf{w}\rVert$, the steeper the transition.

# %%
def plot_proba(predict_proba, X, y, ax=None, levels=20):
    ax = ax or plt.gca()
    x1, x2 = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200),
                         np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200))
    grid = np.c_[x1.ravel(), x2.ravel()]
    p = predict_proba(grid).reshape(x1.shape)
    cs = ax.contourf(x1, x2, p, levels=levels, cmap="RdBu_r", alpha=0.6, vmin=0, vmax=1)
    ax.contour(x1, x2, p, levels=[0.5], colors="k")
    plot_data(X, y, ax)
    return cs

cs = plot_proba(lambda G: sigmoid(G @ w_l + b_l), X, y)
plt.colorbar(cs, label="$p(y=1\\mid x)$"); plt.title("Predicted probability"); plt.show()

# %% [markdown]
# ## 4. Softmax regression: many classes
#
# For $K$ classes we keep one weight vector per class, stacked in
# $W \in \mathbb{R}^{d\times K}$, and turn the scores $\mathbf{z} = W^\top\mathbf{x} + \mathbf{b}$
# into probabilities with the **softmax**
#
# $$p(y=k\mid\mathbf{x}) = \frac{e^{z_k}}{\sum_j e^{z_j}} .$$
#
# With one-hot targets $Y$ and cross-entropy loss, the gradient is again "prediction minus target":
# $\nabla_W \mathcal{L} = \frac{1}{N} X^\top (P - Y)$.
#
# We use the scikit-learn `digits` dataset: 1 797 grayscale images of size
# $8\times 8$ (so $d = 64$), ten classes. It ships with scikit-learn, so no download is needed.

# %%
digits = load_digits()
Xd = digits.data / 16.0          # pixel values in [0, 1]
yd = digits.target
Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(Xd, yd, test_size=0.25, random_state=0, stratify=yd)
print("train", Xd_tr.shape, "test", Xd_te.shape)

fig, axes = plt.subplots(1, 10, figsize=(10, 1.4))
for k, ax in enumerate(axes):
    ax.imshow(Xd_tr[yd_tr == k][0].reshape(8, 8), cmap="gray_r"); ax.set_title(k); ax.axis("off")
plt.show()

# %%
def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)   # shift for numerical stability
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)

def one_hot(y, K):
    Y = np.zeros((len(y), K)); Y[np.arange(len(y)), y] = 1; return Y

def train_softmax(X, y, K, lr=0.5, epochs=100, batch_size=64, seed=0):
    r = np.random.default_rng(seed)
    W, b = np.zeros((X.shape[1], K)), np.zeros(K)
    Y = one_hot(y, K)
    losses = []
    for _ in range(epochs):
        idx = r.permutation(len(X))
        for start in range(0, len(X), batch_size):         # mini-batch gradient descent
            j = idx[start:start + batch_size]
            P = softmax(X[j] @ W + b)
            G = (P - Y[j]) / len(j)
            W -= lr * X[j].T @ G
            b -= lr * G.sum(axis=0)
        P = softmax(X @ W + b)
        losses.append(-np.mean(np.log(P[np.arange(len(y)), y] + 1e-12)))
    return W, b, losses

W, b, losses = train_softmax(Xd_tr, yd_tr, K=10)
acc_tr = np.mean(np.argmax(Xd_tr @ W + b, 1) == yd_tr)
acc_te = np.mean(np.argmax(Xd_te @ W + b, 1) == yd_te)
print(f"train accuracy {acc_tr:.3f} | test accuracy {acc_te:.3f}")

# %% [markdown]
# Each column of $W$ is a "template" the model matches against the input.
# Visualising them shows what a linear classifier can and cannot express: one
# blurry prototype per class, with no way to model variations in stroke position.

# %%
fig, axes = plt.subplots(1, 10, figsize=(10, 1.4))
lim = np.abs(W).max()
for k, ax in enumerate(axes):
    ax.imshow(W[:, k].reshape(8, 8), cmap="RdBu_r", vmin=-lim, vmax=lim); ax.set_title(k); ax.axis("off")
plt.suptitle("Learned weight templates (red = positive evidence)", y=1.15); plt.show()

# %% [markdown]
# ## 5. Where linear models fail: XOR
#
# Minsky and Papert (1969) pointed out that a single-layer perceptron cannot
# represent XOR: no straight line separates $\{(0,0),(1,1)\}$ from $\{(0,1),(1,0)\}$.
# Watch logistic regression try anyway on a noisy XOR dataset.

# %%
Xx = rng.uniform(-1, 1, size=(400, 2))
yx = ((Xx[:, 0] > 0) ^ (Xx[:, 1] > 0)).astype(int)
w_x, b_x, losses_x = train_logreg(Xx, yx, lr=0.5, steps=2000)
acc = np.mean((sigmoid(Xx @ w_x + b_x) > 0.5) == yx)
print(f"accuracy on XOR: {acc:.3f}  (chance = 0.5)")
plot_proba(lambda G: sigmoid(G @ w_x + b_x), Xx, yx); plt.title("Logistic regression on XOR"); plt.show()

# %% [markdown]
# Two ways out:
#
# * **Hand-craft features**: add $x_1 x_2$ as an extra input and XOR becomes linearly
#   separable. This works, but requires us to know the right features in advance
#   (you will try it in the exercises).
# * **Learn the features**: stack linear models with nonlinearities in between, so
#   that the first layers *compute* useful features and the last layer is a linear
#   classifier on top of them. That is a multilayer perceptron — the topic of Lecture 2.
#
# ## Summary
#
# | Model | Output | Loss | Update |
# |---|---|---|---|
# | Perceptron | $\operatorname{sign}(\mathbf{w}^\top\mathbf{x}+b)$ | mistakes | $\mathbf{w} \mathrel{+}= \eta\,y\,\mathbf{x}$ on errors |
# | Logistic regression | $\sigma(\mathbf{w}^\top\mathbf{x}+b)$ | binary cross-entropy | $\mathbf{w} \mathrel{-}= \eta\,X^\top(\hat{\mathbf p}-\mathbf y)/N$ |
# | Softmax regression | $\operatorname{softmax}(W^\top\mathbf{x}+\mathbf b)$ | cross-entropy | $W \mathrel{-}= \eta\,X^\top(P-Y)/N$ |
#
# **Further reading.** Goodfellow, Bengio & Courville, *Deep Learning*, ch. 5–6;
# Bishop, *Pattern Recognition and Machine Learning*, §4.1–4.3.
