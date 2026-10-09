# %% [markdown]
# # Exercises 1 — Linear Models
#
# Fill in every cell marked `# YOUR CODE HERE` and answer the written questions.
# The `assert` cells check your work: if a cell runs without an error, you are
# probably on the right track.

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, make_circles
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(0)

# %% [markdown]
# ## Exercise 1.1 — A numerically stable sigmoid
#
# The naive formula `1 / (1 + np.exp(-z))` overflows for large negative `z`
# (try `np.exp(1000)`). Implement `sigmoid(z)` so that it works for any real input
# without warnings. *Hint:* for $z<0$ use the equivalent form $e^{z}/(1+e^{z})$.

# %%
def sigmoid(z):
    z = np.asarray(z, dtype=float)
    ### BEGIN SOLUTION
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out
    ### END SOLUTION

# %%
with np.errstate(over="raise"):          # turn overflow warnings into errors
    s = sigmoid(np.array([-1000.0, -5.0, 0.0, 5.0, 1000.0]))
assert np.allclose(s, [0.0, 0.00669285, 0.5, 0.99330715, 1.0])
assert np.allclose(sigmoid(np.linspace(-3, 3, 7)) + sigmoid(-np.linspace(-3, 3, 7)), 1.0)
print("sigmoid OK")

# %% [markdown]
# ## Exercise 1.2 — Loss, gradient and a gradient check
#
# Implement `loss_and_grad(w, b, X, y)` for logistic regression. It should return
# the mean binary cross-entropy and its gradients `(dw, db)`.
#
# Then implement `numerical_grad`, which estimates the gradient with **central
# finite differences**, $\frac{\partial \mathcal L}{\partial \theta_j} \approx
# \frac{\mathcal L(\theta + h e_j) - \mathcal L(\theta - h e_j)}{2h}$. Gradient checking
# is the single most useful debugging tool when you write backpropagation by hand.

# %%
def loss_and_grad(w, b, X, y):
    ### BEGIN SOLUTION
    p = sigmoid(X @ w + b)
    eps = 1e-12
    loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    g = (p - y) / len(y)
    return loss, X.T @ g, g.sum()
    ### END SOLUTION


def numerical_grad(f, theta, h=1e-5):
    """Central-difference gradient of scalar function f at the 1-D array theta."""
    grad = np.zeros_like(theta)
    ### BEGIN SOLUTION
    for j in range(len(theta)):
        e = np.zeros_like(theta); e[j] = h
        grad[j] = (f(theta + e) - f(theta - e)) / (2 * h)
    ### END SOLUTION
    return grad

# %%
Xs = rng.normal(size=(30, 4)); ys = (rng.random(30) > 0.5).astype(float)
w0, b0 = rng.normal(size=4), 0.3
_, dw, db = loss_and_grad(w0, b0, Xs, ys)
f = lambda th: loss_and_grad(th[:4], th[4], Xs, ys)[0]
num = numerical_grad(f, np.r_[w0, b0])
rel_err = np.linalg.norm(num - np.r_[dw, db]) / np.linalg.norm(num + np.r_[dw, db])
print(f"relative error: {rel_err:.2e}")
assert rel_err < 1e-7
print("gradient OK")

# %% [markdown]
# ## Exercise 1.3 — Breast cancer classification
#
# Train logistic regression with full-batch gradient descent on the Wisconsin
# breast cancer dataset (30 features, 569 patients).
#
# 1. **Standardise** the features using the mean and standard deviation of the
#    *training* set only (why not the full dataset? — answer below).
# 2. Train for a few hundred steps using your `loss_and_grad`.
# 3. Report test accuracy. You should reach at least **95 %**.

# %%
data = load_breast_cancer()
X_tr, X_te, y_tr, y_te = train_test_split(data.data, data.target, test_size=0.3,
                                          random_state=0, stratify=data.target)

### BEGIN SOLUTION
mu, sd = X_tr.mean(0), X_tr.std(0)
X_tr_s, X_te_s = (X_tr - mu) / sd, (X_te - mu) / sd
w, b = np.zeros(X_tr.shape[1]), 0.0
history = []
for step in range(500):
    loss, dw, db = loss_and_grad(w, b, X_tr_s, y_tr)
    w -= 0.5 * dw; b -= 0.5 * db
    history.append(loss)
test_acc = np.mean((sigmoid(X_te_s @ w + b) > 0.5) == y_te)
plt.plot(history); plt.xlabel("step"); plt.ylabel("training loss"); plt.show()
### END SOLUTION
print(f"test accuracy: {test_acc:.3f}")

# %%
assert test_acc >= 0.95

# %% [markdown]
# **Question.** Why must the standardisation statistics be computed on the
# training set only? What would go wrong otherwise?
#
# <!-- BEGIN SOLUTION -->
# Using test-set statistics leaks information about the test data into training
# (data leakage). The test set is supposed to simulate unseen future data, for
# which we will not know the mean and variance in advance; using them gives an
# optimistically biased estimate of generalisation performance.
# <!-- END SOLUTION -->

# %% [markdown]
# ## Exercise 1.4 — Feature engineering for nonlinear data
#
# The `circles` dataset below is not linearly separable. Write a feature map
# `phi(X)` that appends **quadratic features** ($x_1^2$, $x_2^2$, $x_1 x_2$) to the
# inputs, and show that logistic regression on $\phi(\mathbf{x})$ reaches at
# least 95 % training accuracy.

# %%
Xc, yc = make_circles(n_samples=400, noise=0.08, factor=0.5, random_state=0)

def phi(X):
    ### BEGIN SOLUTION
    return np.c_[X, X[:, 0] ** 2, X[:, 1] ** 2, X[:, 0] * X[:, 1]]
    ### END SOLUTION

Phi = phi(Xc)
w, b = np.zeros(Phi.shape[1]), 0.0
for _ in range(3000):
    _, dw, db = loss_and_grad(w, b, Phi, yc)
    w -= 1.0 * dw; b -= 1.0 * db
circ_acc = np.mean((sigmoid(Phi @ w + b) > 0.5) == yc)
print(f"training accuracy with quadratic features: {circ_acc:.3f}")

# Decision boundary in the original input space
g1, g2 = np.meshgrid(np.linspace(-1.5, 1.5, 200), np.linspace(-1.5, 1.5, 200))
P = sigmoid(phi(np.c_[g1.ravel(), g2.ravel()]) @ w + b).reshape(g1.shape)
plt.contourf(g1, g2, P, levels=20, cmap="RdBu_r", alpha=0.6)
plt.scatter(Xc[:, 0], Xc[:, 1], c=yc, cmap="RdBu_r", s=10, edgecolor="k", linewidth=0.3)
plt.gca().set_aspect("equal"); plt.title("Linear in $\\phi(x)$, nonlinear in $x$"); plt.show()

# %%
assert Phi.shape == (400, 5)
assert circ_acc >= 0.95

# %% [markdown]
# **Question.** For inputs in $\mathbb{R}^d$, how many features does a full
# polynomial expansion of degree $p$ produce (order of magnitude)? Why does this
# motivate *learning* features instead of designing them?
#
# <!-- BEGIN SOLUTION -->
# The number of monomials of degree at most $p$ in $d$ variables is
# $\binom{d+p}{p} = O(d^p)$. For a $28\times28$ image ($d=784$) and $p=3$ this is
# already about $8\times10^7$ features. Hand-crafted expansions grow combinatorially
# and still need us to guess the right form; a neural network instead learns a
# compact, task-specific feature map from the data.
# <!-- END SOLUTION -->

# %% [markdown]
# ## Exercise 1.5 — The perceptron's margin (pen and paper)
#
# Novikoff's theorem bounds the number of perceptron mistakes by $(R/\gamma)^2$.
# Suppose you multiply every input by a constant $c>0$. What happens to $R$, to
# $\gamma$, and to the bound? What does this tell you about what the bound measures?
#
# <!-- BEGIN SOLUTION -->
# Both $R$ and $\gamma$ scale by $c$, so $(R/\gamma)^2$ is unchanged. The bound depends
# only on the *geometry* of the problem — how wide the margin is relative to the spread
# of the data — not on its absolute scale. (With $\mathbf{w}$ initialised at zero the
# perceptron's sequence of mistakes is indeed scale invariant.)
# <!-- END SOLUTION -->
