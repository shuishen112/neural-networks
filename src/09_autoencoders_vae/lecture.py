# %% [markdown]
# # Lecture 9 — Autoencoders and Variational Autoencoders
#
# **Neural Networks · Master's course**
#
# So far every network we trained was told what to output: a class label, a next
# token, a target value. In this lecture the target is the input itself. That
# turns out to be a surprisingly rich idea. It gives us
#
# 1. **autoencoders**, networks that compress data through a bottleneck and in
#    doing so learn a *representation* of it,
# 2. a precise connection to a classical method: a *linear* autoencoder trained
#    with squared error recovers **principal component analysis (PCA)**,
# 3. **denoising autoencoders**, which learn the structure of the data by
#    undoing corruption (an idea we meet again in Lecture 10),
# 4. and finally the **variational autoencoder (VAE)**, a proper probabilistic
#    *generative model* that we can sample new data from. We derive its training
#    objective, the **evidence lower bound (ELBO)**, and the **reparameterisation
#    trick** that makes it trainable with backpropagation.
#
# We close with a map of the generative-model landscape (GANs, normalising flows,
# autoregressive models and diffusion models) to set the stage for the last lecture.
#
# Throughout we use the $8\times8$ scikit-learn `digits` from Lecture 1: small
# enough that every model here trains in seconds on a laptop CPU.

# %%
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

torch.manual_seed(0)
rng = np.random.default_rng(0)
torch.set_num_threads(2)
plt.rcParams["figure.figsize"] = (5, 4)
plt.rcParams["axes.grid"] = True

digits = load_digits()
X_all = (digits.data / 16.0).astype(np.float32)          # pixel intensities in [0, 1]
y_all = digits.target
X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.25,
                                          random_state=0, stratify=y_all)
Xtr_t, Xte_t = torch.from_numpy(X_tr), torch.from_numpy(X_te)
print("train", Xtr_t.shape, Xtr_t.dtype, "| test", Xte_t.shape)

def show_images(rows, titles=None, row_labels=None, vmin=0.0, vmax=1.0, size=1.0):
    """Plot a list of rows, each an array of shape (n, 64), as a grid of 8x8 images."""
    rows = [np.asarray(r) for r in rows]
    n_r, n_c = len(rows), max(len(r) for r in rows)
    fig, axes = plt.subplots(n_r, n_c, figsize=(size * n_c, size * n_r + 0.3), squeeze=False)
    for i, r in enumerate(rows):
        for j in range(n_c):
            ax = axes[i, j]; ax.axis("off")
            if j < len(r):
                ax.imshow(r[j].reshape(8, 8), cmap="gray_r", vmin=vmin, vmax=vmax)
            if titles is not None and i == 0 and j < len(titles):
                ax.set_title(titles[j], fontsize=8)
        if row_labels is not None:
            axes[i, 0].text(-1.0, 3.5, row_labels[i], ha="right", va="center", fontsize=9)
    plt.show()

# %% [markdown]
# ## 1. Autoencoders
#
# An **autoencoder** consists of two networks:
#
# * an **encoder** $f_\phi:\mathbb{R}^d \to \mathbb{R}^k$ that maps an input
#   $\mathbf{x}$ to a **code** (or latent representation) $\mathbf{z} = f_\phi(\mathbf{x})$,
# * a **decoder** $g_\theta:\mathbb{R}^k \to \mathbb{R}^d$ that maps the code back to
#   a reconstruction $\hat{\mathbf{x}} = g_\theta(\mathbf{z})$.
#
# Both are trained together to minimise the reconstruction error, e.g.
#
# $$\mathcal{L}(\phi,\theta) = \frac{1}{N}\sum_{i=1}^N \big\lVert \mathbf{x}_i - g_\theta(f_\phi(\mathbf{x}_i))\big\rVert^2 .$$
#
# Without constraints the problem is trivial: the identity map reconstructs
# perfectly and learns nothing. Something must force the network to *summarise*.
# The classic choice is an **undercomplete** autoencoder, $k \ll d$: the code is a
# bottleneck, so the network must keep the directions of variation that matter
# and discard the rest. (Other choices: sparsity penalties on $\mathbf{z}$,
# corrupted inputs (§3), or a prior on $\mathbf{z}$ (§5).)
#
# ### 1.1 The linear autoencoder is PCA
#
# Take the simplest case: centred data ($\sum_i \mathbf{x}_i = 0$), linear encoder
# $\mathbf{z} = E\mathbf{x}$ with $E \in \mathbb{R}^{k\times d}$, linear decoder
# $\hat{\mathbf{x}} = D\mathbf{z}$ with $D\in\mathbb{R}^{d\times k}$, squared error.
# Stack the data as rows of $X\in\mathbb{R}^{N\times d}$; the loss is
#
# $$\mathcal{L}(E, D) = \tfrac1N\lVert X - X E^\top D^\top\rVert_F^2 .$$
#
# The reconstruction $X E^\top D^\top$ has rank at most $k$. By the
# **Eckart–Young theorem**, the best rank-$k$ approximation of $X$ in Frobenius
# norm is its truncated SVD, $X_k = X V_k V_k^\top$, where the columns of
# $V_k\in\mathbb{R}^{d\times k}$ are the top-$k$ right singular vectors of $X$, i.e.
# the top-$k$ **principal components** (eigenvectors of the covariance $X^\top X/N$).
# Choosing $E = V_k^\top$, $D = V_k$ attains this optimum, so
#
# $$\min_{E,D}\mathcal{L}(E,D) = \text{PCA reconstruction error with } k \text{ components.}$$
#
# But the optimum is **not unique**: for any invertible $A\in\mathbb{R}^{k\times k}$,
# $E' = A E$ and $D' = D A^{-1}$ give the same reconstruction. So a linear
# autoencoder recovers the PCA **subspace** $\operatorname{span}(V_k)$, but not
# the individual principal directions or their ordering (Baldi & Hornik, 1989,
# showed moreover that all other critical points are saddles, so gradient descent
# reliably finds this subspace).
#
# Let us verify this numerically. We train a bias-free linear autoencoder on the
# centred digits with full-batch Adam and compare it with `sklearn.decomposition.PCA`.

# %%
k = 8
x_mean = Xtr_t.mean(0, keepdim=True)
Xc_tr, Xc_te = Xtr_t - x_mean, Xte_t - x_mean          # centred data

torch.manual_seed(0)
lin_enc = nn.Linear(64, k, bias=False)
lin_dec = nn.Linear(k, 64, bias=False)
opt = torch.optim.Adam(list(lin_enc.parameters()) + list(lin_dec.parameters()), lr=1e-2)
lin_losses = []
for step in range(1500):
    loss = F.mse_loss(lin_dec(lin_enc(Xc_tr)), Xc_tr)
    opt.zero_grad(); loss.backward(); opt.step()
    lin_losses.append(loss.item())

pca = PCA(n_components=k).fit(X_tr)
with torch.no_grad():
    lin_rec_te = (lin_dec(lin_enc(Xc_te)) + x_mean).numpy()
pca_rec_te = pca.inverse_transform(pca.transform(X_te))
print(f"test MSE  PCA (k={k}): {np.mean((pca_rec_te - X_te) ** 2):.5f}")
print(f"test MSE  linear AE  : {np.mean((lin_rec_te - X_te) ** 2):.5f}")

plt.semilogy(lin_losses, label="linear AE (train)")
plt.axhline(np.mean((pca.inverse_transform(pca.transform(X_tr)) - X_tr) ** 2), color="k", ls="--", label="PCA (train)")
plt.xlabel("step"); plt.ylabel("MSE"); plt.legend(); plt.title("Linear AE converges to the PCA error"); plt.show()

# %% [markdown]
# The errors match. To check that the *subspaces* match, we compute the
# **principal angles** between $\operatorname{span}(D)$ (the column space of the
# decoder, i.e. all possible reconstructions) and the PCA subspace. If $Q$ is an
# orthonormal basis of $\operatorname{span}(D)$ (from a QR decomposition) and
# $V_k$ the PCA basis, the singular values of $V_k^\top Q$ are the cosines of the
# principal angles: all equal to $1$ means the subspaces coincide.

# %%
D_w = lin_dec.weight.detach().numpy().astype(np.float64)    # shape (64, k): columns span the reconstructions
Q, _ = np.linalg.qr(D_w)
cosines = np.linalg.svd(pca.components_ @ Q, compute_uv=False)
print("cosines of principal angles:", np.round(cosines, 4))

# The individual decoder columns are NOT the principal components (they are an arbitrary basis of the same subspace)
fig, axes = plt.subplots(2, k, figsize=(k, 2.3))
for j in range(k):
    for ax, v, name in [(axes[0, j], pca.components_[j], "PCA"), (axes[1, j], D_w[:, j], "AE")]:
        lim = np.abs(v).max()
        ax.imshow(v.reshape(8, 8), cmap="RdBu_r", vmin=-lim, vmax=lim); ax.axis("off")
        ax.set_title(f"{name} {j}", fontsize=8)
plt.suptitle("Same subspace, different basis", y=1.02); plt.show()

# %% [markdown]
# ### 1.2 A nonlinear autoencoder
#
# Replacing the linear maps by MLPs lets the autoencoder represent data that lie
# near a *curved* low-dimensional manifold, which PCA can only approximate with a
# flat subspace. We use a sigmoid output (pixels lie in $[0,1]$) and mini-batch
# training. A small helper trains any `encoder`/`decoder` pair; it takes an
# optional corruption function that we will need for denoising in §3.

# %%
def mlp(sizes, out_act=None):
    layers = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(sizes) - 2:
            layers.append(nn.SiLU())
    if out_act is not None:
        layers.append(out_act)
    return nn.Sequential(*layers)

def train_autoencoder(encoder, decoder, X, epochs=100, lr=1e-3, batch_size=64, corrupt=None, seed=0):
    torch.manual_seed(seed)
    params = list(encoder.parameters()) + list(decoder.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    losses = []
    for epoch in range(epochs):
        perm = torch.randperm(X.shape[0])
        total = 0.0
        for start in range(0, X.shape[0], batch_size):
            xb = X[perm[start:start + batch_size]]
            x_in = corrupt(xb) if corrupt is not None else xb
            loss = F.mse_loss(decoder(encoder(x_in)), xb)    # target is always the CLEAN input
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * xb.shape[0]
        losses.append(total / X.shape[0])
    return losses

torch.manual_seed(0)
enc8 = mlp([64, 128, k])
dec8 = mlp([k, 128, 64], out_act=nn.Sigmoid())
ae_losses = train_autoencoder(enc8, dec8, Xtr_t, epochs=100)

with torch.no_grad():
    ae_rec_te = dec8(enc8(Xte_t)).numpy()
print(f"test MSE  PCA (k={k})       : {np.mean((pca_rec_te - X_te) ** 2):.5f}")
print(f"test MSE  nonlinear AE (k={k}): {np.mean((ae_rec_te - X_te) ** 2):.5f}")

show_images([X_te[:10], pca_rec_te[:10], lin_rec_te[:10], ae_rec_te[:10]],
            row_labels=["input", "PCA", "linear AE", "MLP AE"])

# %% [markdown]
# PCA and the linear autoencoder produce (as promised) identical reconstructions.
# The MLP autoencoder, with the same $k=8$ numbers per image, typically
# reconstructs better: the extra flexibility is spent on modelling the curvature of
# the data manifold. Keep in mind the honest caveats: it has many more parameters,
# it needs iterative training, and its code dimensions have no natural ordering or
# orthogonality.

# %% [markdown]
# ## 2. Visualising a 2-D latent space
#
# With $k=2$ we can look at the code space directly. Each test image becomes a point
# $\mathbf{z}\in\mathbb{R}^2$; we colour the points by their (unused during
# training!) digit label. Classes that the network finds similar end up close.

# %%
torch.manual_seed(0)
enc2 = mlp([64, 256, 128, 2])
dec2 = mlp([2, 128, 256, 64], out_act=nn.Sigmoid())
ae2_losses = train_autoencoder(enc2, dec2, Xtr_t, epochs=150)
with torch.no_grad():
    Z2_tr = enc2(Xtr_t).numpy()
    Z2_te = enc2(Xte_t).numpy()
pca2 = PCA(n_components=2).fit(X_tr)
P2_te = pca2.transform(X_te)

def plot_latent(Z, y, ax, title):
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", s=8, vmin=-0.5, vmax=9.5)
    for c in range(10):                                # write each class label at its centroid
        m = Z[y == c].mean(0)
        ax.text(m[0], m[1], str(c), fontsize=12, weight="bold",
                bbox=dict(facecolor="white", alpha=0.7, lw=0, pad=1))
    ax.set_title(title); ax.set_xlabel("$z_1$"); ax.set_ylabel("$z_2$")
    return sc

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
plot_latent(P2_te, y_te, axes[0], "PCA (2 components)")
sc = plot_latent(Z2_te, y_te, axes[1], "Autoencoder, 2-D code")
plt.colorbar(sc, ax=axes[1], ticks=range(10)); plt.tight_layout(); plt.show()

# %% [markdown]
# The autoencoder's code space is far better organised than the PCA projection:
# most digit classes form their own clusters, although the network never saw a
# label. This is **representation learning**: the code is a useful feature vector
# for downstream tasks (e.g. a classifier trained on very few labelled examples).
#
# Notice also the shape of the code space: the clusters have arbitrary scales,
# elongated shapes and large empty regions between them. Remember this picture;
# it is the reason a plain autoencoder is *not* a generative model (§4).

# %% [markdown]
# ## 3. Denoising autoencoders
#
# A **denoising autoencoder** (DAE; Vincent et al., 2008) receives a corrupted
# input $\tilde{\mathbf{x}} \sim C(\tilde{\mathbf{x}}\mid\mathbf{x})$, for example
# $\tilde{\mathbf{x}} = \mathbf{x} + \sigma\boldsymbol{\epsilon}$ with
# $\boldsymbol{\epsilon}\sim\mathcal{N}(0, I)$, and is trained to output the
# **clean** $\mathbf{x}$:
#
# $$\mathcal{L}_{\text{DAE}} = \mathbb{E}_{\mathbf{x}}\,\mathbb{E}_{\tilde{\mathbf{x}}\sim C(\cdot\mid\mathbf{x})}\big\lVert \mathbf{x} - r_\theta(\tilde{\mathbf{x}})\big\rVert^2 .$$
#
# Now the identity map is useless, so the bottleneck is no longer essential: the
# network must learn *where the data live* in order to project a noisy point back
# onto the data manifold. A beautiful result makes this precise (Vincent, 2011;
# Alain & Bengio, 2014): for small Gaussian noise the optimal denoiser satisfies
#
# $$r^\star(\tilde{\mathbf{x}}) - \tilde{\mathbf{x}} \;\approx\; \sigma^2\,\nabla_{\tilde{\mathbf{x}}}\log p(\tilde{\mathbf{x}}),$$
#
# i.e. the **denoising direction is the score** (gradient of the log-density) of
# the data distribution. Learning to denoise is learning the gradient of
# $\log p$, and by following that gradient one can *generate* data. This is
# exactly the idea behind diffusion models in Lecture 10.

# %%
sigma_noise = 0.3

def add_noise(x):
    return (x + sigma_noise * torch.randn_like(x)).clamp(0.0, 1.0)

torch.manual_seed(0)
dae_enc = mlp([64, 256, 32])
dae_dec = mlp([32, 256, 64], out_act=nn.Sigmoid())
dae_losses = train_autoencoder(dae_enc, dae_dec, Xtr_t, epochs=100, corrupt=add_noise)

torch.manual_seed(1)
X_noisy = add_noise(Xte_t)
with torch.no_grad():
    X_denoised = dae_dec(dae_enc(X_noisy))
    X_plain_ae = dec8(enc8(X_noisy))                   # the ordinary AE from §1.2, fed noisy inputs
print(f"MSE(noisy,    clean): {F.mse_loss(X_noisy, Xte_t).item():.4f}")
print(f"MSE(plain AE, clean): {F.mse_loss(X_plain_ae, Xte_t).item():.4f}")
print(f"MSE(DAE,      clean): {F.mse_loss(X_denoised, Xte_t).item():.4f}")
show_images([X_noisy[:10].numpy(), X_denoised[:10].numpy(), X_te[:10]],
            row_labels=["noisy", "denoised", "clean"])

# %% [markdown]
# ## 4. From autoencoder to generative model
#
# A **generative model** lets us draw *new* samples $\mathbf{x}\sim p_\theta(\mathbf{x})$
# that look like the training data. An autoencoder has a decoder that maps codes to
# images, so it is tempting to sample a random code and decode it. Which
# distribution should we sample codes from, though? The training objective says
# nothing about the distribution of $\mathbf{z}$, so:
#
# * the codes can have any scale and location (sampling $\mathcal{N}(0, I)$ may
#   land far from all training codes);
# * they occupy an irregular region with **holes**; the decoder was never trained
#   on codes in the holes, so its output there is unconstrained;
# * nothing makes nearby codes decode to similar images (no smoothness pressure).
#
# Let us look at what the 2-D autoencoder decodes on a regular grid spanning its codes.

# %%
def decode_grid(decoder, z1, z2):
    """Decode every point of the grid z1 x z2 and tile the images (top row = largest z2)."""
    G = np.array([[a, b] for b in z2[::-1] for a in z1], dtype=np.float32)
    with torch.no_grad():
        imgs = decoder(torch.from_numpy(G)).numpy()
    n1, n2 = len(z1), len(z2)
    return imgs.reshape(n2, n1, 8, 8).transpose(0, 2, 1, 3).reshape(n2 * 8, n1 * 8), G

lo, hi = Z2_tr.min(0), Z2_tr.max(0)
n_grid = 15
z1 = np.linspace(lo[0], hi[0], n_grid); z2 = np.linspace(lo[1], hi[1], n_grid)
canvas, G = decode_grid(dec2, z1, z2)

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
plot_latent(Z2_tr, y_tr, axes[0], "AE codes and the decoding grid")
axes[0].scatter(G[:, 0], G[:, 1], marker="+", c="k", s=12, lw=0.6)
axes[1].imshow(canvas, cmap="gray_r"); axes[1].axis("off"); axes[1].set_title("decoder output on the grid")
plt.tight_layout(); plt.show()

# %% [markdown]
# Grid points inside a cluster decode to recognisable digits, but many points fall
# in empty space and decode to smudges or abrupt mixtures. Now try the obvious
# "prior" on the 8-dimensional codes of the autoencoder from §1.2: a standard
# normal, and a Gaussian fitted to the training codes.

# %%
with torch.no_grad():
    Z8 = enc8(Xtr_t)
    print("AE code statistics: mean norm %.2f, per-dimension std:" % Z8.mean(0).norm().item(),
          np.round(Z8.std(0).numpy(), 2))
    torch.manual_seed(0)
    z_std = torch.randn(10, k)
    m, C = Z8.mean(0), torch.cov(Z8.T)
    fitted = torch.distributions.MultivariateNormal(m, covariance_matrix=C + 1e-4 * torch.eye(k))
    z_fit = fitted.sample((10,))
    show_images([dec8(z_std).numpy(), dec8(z_fit).numpy()],
                row_labels=["z ~ N(0, I)", "z ~ fitted N"])

# %% [markdown]
# Neither is satisfactory: the standard normal ignores the actual scale of the codes, and
# even a fitted Gaussian places mass on regions the decoder never saw. The fix is to
# **make the distribution of the codes part of the model** and train the encoder
# to respect it. That is the variational autoencoder.

# %% [markdown]
# ## 5. Latent-variable models and the variational autoencoder
#
# ### 5.1 The model
#
# A **latent-variable model** explains each observation $\mathbf{x}$ through an
# unobserved cause $\mathbf{z}$:
#
# $$\mathbf{z}\sim p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, I), \qquad \mathbf{x}\mid\mathbf{z} \sim p_\theta(\mathbf{x}\mid\mathbf{z}),$$
#
# where the likelihood $p_\theta(\mathbf{x}\mid\mathbf{z})$ is given by a neural
# network, the **decoder**. For our digits we use independent Bernoulli pixels whose
# probabilities are the decoder outputs, $p_\theta(\mathbf{x}\mid\mathbf{z}) =
# \prod_j \hat x_j^{x_j}(1-\hat x_j)^{1-x_j}$ with $\hat{\mathbf{x}} = \sigma(g_\theta(\mathbf{z}))$.
# (Our pixels are grey levels in $[0,1]$ rather than binary; using the Bernoulli
# cross-entropy anyway is common practice. Strictly it is not a normalised density
# for continuous $x$, see the *continuous Bernoulli* of Loaiza-Ganem & Cunningham, 2019.)
#
# Sampling is trivial (draw $\mathbf{z}$, decode), and there are no holes by
# construction because we choose the prior. The difficulty is **training**: maximum
# likelihood needs the marginal
#
# $$p_\theta(\mathbf{x}) = \int p_\theta(\mathbf{x}\mid\mathbf{z})\,p(\mathbf{z})\,d\mathbf{z},$$
#
# an integral over all latent configurations that has no closed form when the
# decoder is a neural network. Monte Carlo with samples from the prior is hopeless:
# almost every $\mathbf{z}\sim p(\mathbf{z})$ explains a given $\mathbf{x}$ very
# badly, so the estimate has enormous variance. We would like to sample $\mathbf{z}$
# from the **posterior** $p_\theta(\mathbf{z}\mid\mathbf{x})$, the codes that
# actually explain $\mathbf{x}$, but that requires $p_\theta(\mathbf{x})$ again.
#
# ### 5.2 The evidence lower bound (ELBO)
#
# **Variational inference** introduces an approximate posterior
# $q_\phi(\mathbf{z}\mid\mathbf{x})$, here a Gaussian whose mean and variance are
# produced by another network, the **encoder** (or inference network):
#
# $$q_\phi(\mathbf{z}\mid\mathbf{x}) = \mathcal{N}\big(\mathbf{z};\ \boldsymbol\mu_\phi(\mathbf{x}),\ \operatorname{diag}\boldsymbol\sigma^2_\phi(\mathbf{x})\big).$$
#
# Using one shared network for all data points instead of separate variational
# parameters per point is called **amortised inference**. Now multiply and divide by
# $q_\phi$ inside the integral and apply Jensen's inequality ($\log$ is concave):
#
# $$\log p_\theta(\mathbf{x}) = \log \mathbb{E}_{q_\phi(\mathbf{z}\mid\mathbf{x})}\!\left[\frac{p_\theta(\mathbf{x}\mid\mathbf{z})\,p(\mathbf{z})}{q_\phi(\mathbf{z}\mid\mathbf{x})}\right]
# \;\ge\; \mathbb{E}_{q_\phi}\!\left[\log\frac{p_\theta(\mathbf{x}\mid\mathbf{z})\,p(\mathbf{z})}{q_\phi(\mathbf{z}\mid\mathbf{x})}\right] =: \mathcal{L}_{\theta,\phi}(\mathbf{x}).$$
#
# Splitting the logarithm gives the form we implement:
#
# $$\boxed{\;\mathcal{L}_{\theta,\phi}(\mathbf{x}) = \underbrace{\mathbb{E}_{q_\phi(\mathbf{z}\mid\mathbf{x})}\big[\log p_\theta(\mathbf{x}\mid\mathbf{z})\big]}_{\text{reconstruction}} \;-\; \underbrace{\mathrm{KL}\big(q_\phi(\mathbf{z}\mid\mathbf{x})\,\big\Vert\,p(\mathbf{z})\big)}_{\text{regulariser}}\;}$$
#
# The first term is an autoencoder: encode $\mathbf{x}$ to a (random) code, decode,
# and score the reconstruction. The second term pulls each posterior towards the prior,
# which removes the holes and fixes the scale of the code space.
#
# **How tight is the bound?** A second derivation, without Jensen, shows the gap exactly.
# Since $p_\theta(\mathbf{x}\mid\mathbf{z})p(\mathbf{z}) = p_\theta(\mathbf{z}\mid\mathbf{x})p_\theta(\mathbf{x})$,
#
# $$\mathcal{L}_{\theta,\phi}(\mathbf{x}) = \mathbb{E}_{q_\phi}\!\left[\log\frac{p_\theta(\mathbf{z}\mid\mathbf{x})\,p_\theta(\mathbf{x})}{q_\phi(\mathbf{z}\mid\mathbf{x})}\right]
# = \log p_\theta(\mathbf{x}) - \mathrm{KL}\big(q_\phi(\mathbf{z}\mid\mathbf{x})\,\Vert\,p_\theta(\mathbf{z}\mid\mathbf{x})\big).$$
#
# Because KL divergences are non-negative, $\mathcal{L}\le\log p_\theta(\mathbf{x})$ with
# equality iff $q_\phi$ equals the true posterior. Maximising the ELBO over
# $\theta$ and $\phi$ therefore does two things at once: it increases the likelihood
# of the data, and it makes the encoder a better approximation of the posterior.
# We train by **minimising the negative ELBO**, averaged over the data.
#
# ### 5.3 The KL term in closed form
#
# For $q = \mathcal{N}(\boldsymbol\mu, \operatorname{diag}\boldsymbol\sigma^2)$ and
# $p = \mathcal{N}(\mathbf{0}, I)$ the KL factorises over dimensions. For one
# dimension, using $\log q(z) = -\tfrac12\log(2\pi\sigma^2) - \tfrac{(z-\mu)^2}{2\sigma^2}$ and
# $\log p(z) = -\tfrac12\log(2\pi) - \tfrac{z^2}{2}$:
#
# $$\mathrm{KL}(q\Vert p) = \mathbb{E}_q[\log q - \log p]
# = -\tfrac12\log\sigma^2 - \tfrac12\underbrace{\mathbb{E}_q\!\left[\tfrac{(z-\mu)^2}{\sigma^2}\right]}_{=1} + \tfrac12\underbrace{\mathbb{E}_q[z^2]}_{=\mu^2+\sigma^2}$$
#
# so, summing over the $k$ latent dimensions,
#
# $$\mathrm{KL}\big(\mathcal{N}(\boldsymbol\mu,\operatorname{diag}\boldsymbol\sigma^2)\,\Vert\,\mathcal{N}(\mathbf 0,I)\big) = \frac12\sum_{j=1}^k\Big(\mu_j^2 + \sigma_j^2 - 1 - \log\sigma_j^2\Big).$$
#
# It is zero exactly when $\mu_j=0,\ \sigma_j=1$. In code the encoder outputs
# $\log\sigma^2$ rather than $\sigma$: the log-variance is unconstrained, so no
# positivity constraint is needed and the optimisation is better conditioned.
# (You will derive the general case $\mathrm{KL}(\mathcal N(\mu_1,\sigma_1^2)\Vert\mathcal N(\mu_2,\sigma_2^2))$ in the exercises.)
#
# ### 5.4 The reparameterisation trick
#
# The reconstruction term is an expectation over $q_\phi$, which itself depends on
# $\phi$. We estimate it with a single sample $\mathbf{z}\sim q_\phi$, but how do we
# backpropagate *through a sampling operation*? The trick (Kingma & Welling, 2014;
# Rezende et al., 2014) is to move the randomness into a parameter-free variable:
#
# $$\mathbf{z} = \boldsymbol\mu_\phi(\mathbf{x}) + \boldsymbol\sigma_\phi(\mathbf{x})\odot\boldsymbol\epsilon, \qquad \boldsymbol\epsilon\sim\mathcal{N}(\mathbf 0, I).$$
#
# Then $\mathbb{E}_{q_\phi}[h(\mathbf z)] = \mathbb{E}_{\boldsymbol\epsilon}[h(\boldsymbol\mu_\phi + \boldsymbol\sigma_\phi\odot\boldsymbol\epsilon)]$, the
# distribution we average over no longer depends on $\phi$, and we can push the
# gradient inside the expectation:
#
# $$\nabla_\phi\,\mathbb{E}_{q_\phi}[h(\mathbf z)] = \mathbb{E}_{\boldsymbol\epsilon}\big[\nabla_\phi h(\boldsymbol\mu_\phi + \boldsymbol\sigma_\phi\odot\boldsymbol\epsilon)\big],$$
#
# which ordinary backpropagation computes for one sample of $\boldsymbol\epsilon$.
#
# **Why not the alternative?** The general-purpose **score-function** (REINFORCE)
# estimator uses $\nabla_\phi\mathbb{E}_{q_\phi}[h] = \mathbb{E}_{q_\phi}[h(\mathbf z)\,\nabla_\phi\log q_\phi(\mathbf z)]$.
# It works even for discrete $\mathbf z$ or non-differentiable $h$, but it only
# uses the *value* of $h$, not its gradient, and typically has a much larger
# variance. A tiny experiment: estimate $\frac{d}{d\mu}\mathbb{E}_{z\sim\mathcal N(\mu,1)}[z^2] = 2\mu$ with single-sample estimators.

# %%
mu0, n_rep = 1.0, 100_000
eps = rng.standard_normal(n_rep)
z = mu0 + eps
g_score = z ** 2 * (z - mu0)          # h(z) * d/dmu log N(z; mu, 1)
g_reparam = 2 * (mu0 + eps)           # d/dmu h(mu + eps)
print(f"true gradient       : {2 * mu0:.3f}")
print(f"score function      : mean {g_score.mean():.3f}, variance {g_score.var():.2f}")
print(f"reparameterisation  : mean {g_reparam.mean():.3f}, variance {g_reparam.var():.2f}")

# %% [markdown]
# Both estimators are unbiased, but the score-function estimator is several times
# noisier even in one dimension, and the gap grows with the dimension of
# $\mathbf z$ (variance reduction with baselines/control variates helps, but rarely
# closes it). In practice, use reparameterisation whenever the latent variable is
# continuous and the model is differentiable.
#
# ### 5.5 Implementation
#
# The per-example negative ELBO with a Bernoulli decoder is
#
# $$-\mathcal L(\mathbf x) \approx \underbrace{\sum_{j=1}^{64}\mathrm{BCE}\big(\text{logit}_j, x_j\big)}_{-\log p_\theta(\mathbf x\mid\mathbf z),\ \mathbf z = \boldsymbol\mu+\boldsymbol\sigma\odot\boldsymbol\epsilon}
# + \beta\cdot\frac12\sum_{j=1}^k\big(\mu_j^2+\sigma_j^2-1-\log\sigma_j^2\big),$$
#
# with $\beta = 1$ for the true ELBO. Note the **sums** over pixels and latent
# dimensions: using a per-pixel *mean* for the reconstruction (as `F.mse_loss` or
# `F.binary_cross_entropy` do by default) silently up-weights the KL by a factor of 64,
# a very common bug that leads to blurry, collapsed models.
#
# We use `binary_cross_entropy_with_logits` on the decoder's logits for numerical
# stability (the same reason we wrote a stable sigmoid in Lecture 1).

# %%
class VAE(nn.Module):
    def __init__(self, d=64, k=8, hidden=(256, 128)):
        super().__init__()
        h1, h2 = hidden
        self.k = k
        self.encoder = mlp([d, h1, h2, 2 * k])        # outputs [mu, logvar]
        self.decoder = mlp([k, h2, h1, d])            # outputs Bernoulli logits

    def encode(self, x):
        mu, logvar = self.encoder(x).chunk(2, dim=-1)
        return mu, logvar

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return self.decoder(z), mu, logvar

def neg_elbo(x, logits, mu, logvar, beta=1.0):
    rec = F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(dim=1)
    kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=1)
    return (rec + beta * kl).mean(), rec.mean(), kl.mean()

def train_vae(model, X, epochs=100, lr=1e-3, batch_size=128, beta=1.0, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = {"loss": [], "rec": [], "kl": []}
    for epoch in range(epochs):
        perm = torch.randperm(X.shape[0])
        sums = np.zeros(3)
        for start in range(0, X.shape[0], batch_size):
            xb = X[perm[start:start + batch_size]]
            loss, rec, kl = neg_elbo(xb, *model(xb), beta=beta)
            opt.zero_grad(); loss.backward(); opt.step()
            sums += np.array([loss.item(), rec.item(), kl.item()]) * xb.shape[0]
        for key, v in zip(hist, sums / X.shape[0]):
            hist[key].append(v)
    return hist

@torch.no_grad()
def evaluate_vae(model, X, beta=1.0):
    torch.manual_seed(123)
    loss, rec, kl = neg_elbo(X, *model(X), beta=beta)
    return loss.item(), rec.item(), kl.item()

torch.manual_seed(0)
vae = VAE(k=8)
hist = train_vae(vae, Xtr_t, epochs=100)
loss_te, rec_te, kl_te = evaluate_vae(vae, Xte_t)
print(f"test: -ELBO {loss_te:.2f} nats = reconstruction {rec_te:.2f} + KL {kl_te:.2f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].plot(hist["loss"], label="$-$ELBO"); axes[0].plot(hist["rec"], label="reconstruction")
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("nats per image"); axes[0].legend()
axes[1].plot(hist["kl"], color="C2"); axes[1].set_xlabel("epoch"); axes[1].set_title("KL term")
plt.tight_layout(); plt.show()

# %% [markdown]
# The KL term starts near zero (the freshly initialised encoder outputs
# $\boldsymbol\mu\approx 0$, $\log\boldsymbol\sigma^2\approx0$, i.e. the prior)
# and then grows: the model *pays* a few nats of KL to encode information about
# $\mathbf x$ in $\mathbf z$ because this buys a much larger reduction in
# reconstruction error. The KL term is literally the number of nats the code
# transmits about the input, on average (the "rate" in the rate–distortion view).

# %% [markdown]
# ### 5.6 What the VAE can do: sampling, interpolation, a latent manifold
#
# **Sampling from the prior.** Draw $\mathbf z\sim\mathcal N(\mathbf 0,I)$ and decode.
# We display the Bernoulli *means* $\sigma(g_\theta(\mathbf z))$ rather than binary samples.

# %%
@torch.no_grad()
def decode_mean(model, z):
    return torch.sigmoid(model.decoder(z)).numpy()

torch.manual_seed(1)
z_prior = torch.randn(30, vae.k)
samples = decode_mean(vae, z_prior)
show_images([samples[:10], samples[10:20], samples[20:30]])

# %% [markdown]
# Compare with the plain autoencoder samples in §4: decoding random prior codes now
# gives (mostly) plausible digits. They are also noticeably **blurry**; we return to
# why in the summary and the exercises.
#
# **Interpolation.** Encode two test images to their posterior means and decode
# points along the straight line between them. A smooth latent space gives a gradual
# morph rather than an abrupt switch.

# %%
@torch.no_grad()
def interpolate_images(model, xa, xb, n=10):
    mu_a, _ = model.encode(xa[None]); mu_b, _ = model.encode(xb[None])
    lam = torch.linspace(0, 1, n)[:, None]
    return decode_mean(model, (1 - lam) * mu_a + lam * mu_b)

pairs = [(0, 1), (3, 8), (4, 9), (7, 1)]
rows = []
for a, b in pairs:
    ia = int(np.where(y_te == a)[0][0]); ib = int(np.where(y_te == b)[0][0])
    rows.append(interpolate_images(vae, Xte_t[ia], Xte_t[ib]))
show_images(rows, row_labels=[f"{a}→{b}" for a, b in pairs])

# %% [markdown]
# **A 2-D VAE and its latent manifold.** With $k=2$ we can again look at the
# latent space, and decode a regular grid. Because the prior is $\mathcal N(0,I)$,
# we place the grid at equally spaced *quantiles* of the standard normal: each cell
# of the grid then covers equal prior probability mass.

# %%
torch.manual_seed(0)
vae2 = VAE(k=2)
hist2 = train_vae(vae2, Xtr_t, epochs=150)
print("2-D VAE test (-ELBO, rec, KL):", np.round(evaluate_vae(vae2, Xte_t), 2))

with torch.no_grad():
    mu2_te, _ = vae2.encode(Xte_t)
mu2_te = mu2_te.numpy()

quantiles = torch.linspace(0.03, 0.97, n_grid)
zq = torch.distributions.Normal(0.0, 1.0).icdf(quantiles).numpy()
canvas_vae, Gv = decode_grid(lambda z: torch.sigmoid(vae2.decoder(z)), zq, zq)

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
plot_latent(mu2_te, y_te, axes[0], "VAE posterior means (test set)")
axes[0].scatter(Gv[:, 0], Gv[:, 1], marker="+", c="k", s=12, lw=0.6)
axes[1].imshow(canvas_vae, cmap="gray_r"); axes[1].axis("off"); axes[1].set_title("decoded quantile grid")
plt.tight_layout(); plt.show()

# %% [markdown]
# Compare with the plain AE in §4: the codes now fill a roughly Gaussian blob
# centred at the origin, classes border on each other without large gaps, and
# *every* grid point decodes to something digit-like, with smooth transitions
# between neighbouring classes. The price is that some classes overlap in two
# dimensions and the reconstructions are blurrier than those of a plain AE.

# %% [markdown]
# ### 5.7 Active units, posterior collapse and $\beta$-VAE
#
# The KL term in the ELBO is a sum over latent dimensions, so we can ask how much
# information each dimension carries. A dimension whose posterior equals the prior
# for every input ($\mu_j(\mathbf x)\approx0,\ \sigma_j(\mathbf x)\approx1$) has KL $\approx0$
# and is ignored by the decoder: it is **inactive**. A common diagnostic (Burda et al.,
# 2016) counts dimensions with $\operatorname{Var}_{\mathbf x}\big[\mu_j(\mathbf x)\big] > 0.01$.

# %%
with torch.no_grad():
    mu_te8, logvar_te8 = vae.encode(Xte_t)
    kl_per_dim = (0.5 * (mu_te8.pow(2) + logvar_te8.exp() - 1 - logvar_te8)).mean(0).numpy()
    act = mu_te8.var(0).numpy()
print("Var_x[mu_j]   :", np.round(act, 3))
print("active units  :", int((act > 0.01).sum()), "of", vae.k)
plt.bar(range(vae.k), kl_per_dim); plt.xlabel("latent dimension $j$"); plt.ylabel("average KL (nats)")
plt.title("Information per latent dimension"); plt.show()

# %% [markdown]
# Typically only some of the 8 dimensions are used: the ELBO prefers to switch off
# dimensions whose information is not worth their KL cost. In the extreme, a very
# powerful decoder (for instance an autoregressive one) can model the data without
# using $\mathbf z$ at all and the whole posterior collapses to the prior, a failure
# mode called **posterior collapse**. Remedies include KL annealing (start with
# $\beta=0$ and increase it), "free bits" (do not penalise the first $\lambda$ nats
# per dimension) and weaker decoders.
#
# The **$\beta$-VAE** (Higgins et al., 2017) deliberately uses $\beta \neq 1$ in
# $\mathbb{E}_q[\log p_\theta(\mathbf x\mid\mathbf z)] - \beta\,\mathrm{KL}(q\Vert p)$.
# With $\beta>1$ the code is squeezed harder, which was observed to encourage
# **disentangled** dimensions (each capturing one factor of variation) at the
# price of worse reconstructions; $\beta<1$ gives sharper reconstructions but a
# latent space that matches the prior less well, so samples get worse. For
# $\beta\ne1$ the objective is no longer a bound on $\log p_\theta(\mathbf x)$.
# The effect on our 2-D VAE: a quick comparison of the code spaces.

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
for ax, beta in zip(axes, [0.1, 1.0, 4.0]):
    if beta == 1.0:
        model_b = vae2
    else:
        torch.manual_seed(0); model_b = VAE(k=2)
        train_vae(model_b, Xtr_t, epochs=150, beta=beta)
    l, r, kl = evaluate_vae(model_b, Xte_t, beta=beta)
    with torch.no_grad():
        mu_b, _ = model_b.encode(Xte_t)
    plot_latent(mu_b.numpy(), y_te, ax, f"$\\beta$={beta}: rec {r:.1f}, KL {kl:.1f}")
plt.tight_layout(); plt.show()

# %% [markdown]
# Small $\beta$: the codes spread out far beyond the prior (sampling from
# $\mathcal N(0,I)$ would miss much of it), the KL is large and reconstructions are good.
# Large $\beta$: the code is compressed towards the prior, classes merge and the
# reconstruction error grows.

# %% [markdown]
# ## 6. The landscape of deep generative models
#
# The VAE is one answer to the question *how do we fit a flexible density
# $p_\theta(\mathbf x)$ with a neural network?* Four other families dominate:
#
# **Generative adversarial networks (GANs; Goodfellow et al., 2014).** A generator
# $G$ maps noise $\mathbf z\sim p(\mathbf z)$ to samples; a discriminator $D$ tries to
# tell real from generated data. They play the minimax game
#
# $$\min_G\max_D\; \mathbb{E}_{\mathbf x\sim p_{\text{data}}}\big[\log D(\mathbf x)\big] + \mathbb{E}_{\mathbf z\sim p(\mathbf z)}\big[\log\big(1 - D(G(\mathbf z))\big)\big].$$
#
# For the optimal discriminator $D^\star = p_{\text{data}}/(p_{\text{data}}+p_G)$ the
# generator minimises $2\,\mathrm{JS}(p_{\text{data}}\Vert p_G) - \log 4$, a
# Jensen–Shannon divergence. GANs need no likelihood and produce famously sharp
# images, but training is a saddle-point problem (unstable, sensitive to
# hyperparameters) and they tend to **mode-collapse** (cover only part of the data).
# They give no likelihood and no encoder.
#
# **Normalising flows** (Dinh et al., 2017; Papamakarios et al., 2021). Use an
# *invertible* network $\mathbf x = f_\theta(\mathbf z)$ and the change-of-variables formula
#
# $$\log p_\theta(\mathbf x) = \log p(\mathbf z) - \log\left|\det \frac{\partial f_\theta}{\partial\mathbf z}\right|, \qquad \mathbf z = f_\theta^{-1}(\mathbf x).$$
#
# This gives *exact* likelihoods and exact inference, but every layer must be
# invertible with a cheap Jacobian determinant (coupling layers, triangular
# Jacobians), and $\dim\mathbf z = \dim\mathbf x$, so there is no compression.
#
# **Autoregressive models.** Factorise by the chain rule,
# $p_\theta(\mathbf x) = \prod_{j} p_\theta(x_j\mid x_{<j})$, and model each conditional with
# a masked network (PixelCNN, WaveNet) or a Transformer (Lecture 8). Exact likelihood,
# excellent quality; this is how large language models generate text. Sampling is
# sequential, one dimension (token, pixel) at a time, which is slow for large outputs.
#
# **Diffusion models** (Sohl-Dickstein et al., 2015; Ho et al., 2020). Fix a forward
# process that gradually adds Gaussian noise until only noise remains, and learn to
# reverse it step by step with a denoising network. One can view a diffusion model as a
# hierarchical VAE with a fixed encoder and $T$ latent layers, or as learning the score
# $\nabla\log p$ at many noise levels, the denoising-autoencoder connection of §3.
# State of the art for images, audio and video. **This is Lecture 10.**
#
# | Family | Exact likelihood | Fast sampling | Latent code | Typical weakness |
# |---|---|---|---|---|
# | VAE | lower bound | yes (1 pass) | yes | blurry samples |
# | GAN | no | yes (1 pass) | no encoder | unstable training, mode collapse |
# | Normalising flow | yes | yes (1 pass) | yes (same dim.) | restricted architectures |
# | Autoregressive | yes | no (sequential) | no | slow sampling |
# | Diffusion | lower bound | no (many steps) | (noise) | slow sampling, much compute |

# %% [markdown]
# ## Summary
#
# * An **autoencoder** learns a code $\mathbf z = f_\phi(\mathbf x)$ by reconstructing its
#   input through a bottleneck. A **linear** autoencoder with squared error spans exactly
#   the **PCA** subspace (but not the individual components); nonlinear autoencoders capture
#   curved manifolds and give useful features without labels.
# * A **denoising autoencoder** maps corrupted inputs back to clean ones; its denoising
#   direction approximates the **score** $\nabla\log p(\mathbf x)$.
# * A plain autoencoder is not a generative model: its codes have no known distribution and
#   the decoder is untrained in the gaps.
# * The **VAE** is a latent-variable model $p(\mathbf z)p_\theta(\mathbf x\mid\mathbf z)$ trained by
#   maximising the **ELBO**
#   $\mathbb{E}_{q_\phi}[\log p_\theta(\mathbf x\mid\mathbf z)] - \mathrm{KL}(q_\phi(\mathbf z\mid\mathbf x)\Vert p(\mathbf z))
#   = \log p_\theta(\mathbf x) - \mathrm{KL}(q_\phi\Vert p_\theta(\mathbf z\mid\mathbf x))$,
#   with the Gaussian KL $\tfrac12\sum_j(\mu_j^2+\sigma_j^2-1-\log\sigma_j^2)$ and the
#   **reparameterisation** $\mathbf z = \boldsymbol\mu+\boldsymbol\sigma\odot\boldsymbol\epsilon$,
#   a low-variance gradient estimator.
# * VAE samples are **blurry** because a factorised Gaussian/Bernoulli decoder must
#   average over everything consistent with an imprecise code, and because the encoder's
#   posterior overlaps neighbouring codes. Avoid by better decoders, hierarchical latents, or
#   different model families.
#
# **Further reading.**
# Goodfellow, Bengio & Courville, *Deep Learning*, ch. 14 (autoencoders) and ch. 20 (generative models);
# Kingma & Welling, "Auto-Encoding Variational Bayes", ICLR 2014, and their review
# "An Introduction to Variational Autoencoders" (2019);
# Baldi & Hornik, "Neural networks and principal component analysis", *Neural Networks* 2 (1989);
# Vincent et al., "Extracting and composing robust features with denoising autoencoders", ICML 2008;
# Higgins et al., "β-VAE", ICLR 2017;
# Bishop & Bishop, *Deep Learning: Foundations and Concepts* (2024), ch. 16–17 and 19;
# Murphy, *Probabilistic Machine Learning: Advanced Topics* (2023), ch. 21–26.
