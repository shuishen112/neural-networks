# %% [markdown]
# # Exercises 10 — Denoising Diffusion Models
#
# Fill in every cell marked `# YOUR CODE HERE` and answer the written questions.
# The `assert` cells check your work: if a cell runs without an error, you are
# probably on the right track. At the end, you put all your pieces together and
# train a small diffusion model on 2-D data.
#
# **Conventions** (as in the lecture): schedule arrays have length $T+1$ and are
# indexed directly by $t\in\{0,\dots,T\}$, with $\beta_0 = 0$ and $\bar\alpha_0 = 1$.
# A batch of time steps `t` is a `torch.long` tensor of shape `(B,)`.

# %%
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_swiss_roll
from sklearn.neighbors import NearestNeighbors

torch.manual_seed(0)
torch.set_num_threads(2)

# %% [markdown]
# ## Exercise 10.1 — Noise schedules
#
# Implement
#
# * `linear_alpha_bar(T, beta_1, beta_T)`: $\beta_t$ linearly spaced from $\beta_1$ to $\beta_T$ for $t=1..T$,
#   and $\bar\alpha_t = \prod_{s=1}^t(1-\beta_s)$;
# * `cosine_alpha_bar(T, s)`: $\bar\alpha_t = f(t)/f(0)$ with $f(t) = \cos^2\!\big(\frac{t/T+s}{1+s}\cdot\frac{\pi}{2}\big)$.
#
# Both must return a float64 NumPy array of length $T+1$ **including** $\bar\alpha_0 = 1$.
# Then `betas_from_alpha_bar(alpha_bar, max_beta)` must recover $\beta_t = 1-\bar\alpha_t/\bar\alpha_{t-1}$
# for $t = 1..T$, clipped to at most `max_beta`, and return an array of length $T+1$ with $\beta_0 = 0$.

# %%
def linear_alpha_bar(T, beta_1=1e-4, beta_T=0.02):
    ### BEGIN SOLUTION
    betas = np.linspace(beta_1, beta_T, T)
    return np.concatenate([[1.0], np.cumprod(1.0 - betas)])
    ### END SOLUTION

def cosine_alpha_bar(T, s=0.008):
    ### BEGIN SOLUTION
    t = np.arange(T + 1, dtype=np.float64)
    f = np.cos((t / T + s) / (1 + s) * np.pi / 2) ** 2
    return f / f[0]
    ### END SOLUTION

def betas_from_alpha_bar(alpha_bar, max_beta=0.999):
    ### BEGIN SOLUTION
    betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return np.concatenate([[0.0], np.clip(betas, 0.0, max_beta)])
    ### END SOLUTION

# %%
T = 1000
ab_lin, ab_cos = linear_alpha_bar(T), cosine_alpha_bar(T)
for ab in (ab_lin, ab_cos):
    assert ab.shape == (T + 1,)
    assert np.isclose(ab[0], 1.0)
    assert np.all(np.diff(ab) < 0), "alpha_bar must be strictly decreasing"
    assert np.all(ab > 0) and ab[-1] < 1e-3
assert np.isclose(ab_lin[1], 1 - 1e-4) and np.isclose(ab_lin[2], (1 - 1e-4) * (1 - 1e-4 - (0.02 - 1e-4) / 999))
assert np.isclose(ab_cos[T // 2], np.cos((0.5 + 0.008) / 1.008 * np.pi / 2) ** 2 / np.cos(0.008 / 1.008 * np.pi / 2) ** 2)
b = betas_from_alpha_bar(ab_lin)
assert b.shape == (T + 1,) and b[0] == 0.0
assert np.allclose(b[1:], np.linspace(1e-4, 0.02, T))
assert betas_from_alpha_bar(ab_cos).max() <= 0.999
print("schedules OK")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
for ab, name in [(ab_lin, "linear"), (ab_cos, "cosine")]:
    axes[0].plot(ab, label=name)
    axes[1].semilogy(ab[1:] / (1 - ab[1:]), label=name)
axes[0].set_title(r"$\bar\alpha_t$"); axes[1].set_title("SNR"); axes[0].legend(); axes[1].legend()
for ax in axes:
    ax.set_xlabel("t")
plt.show()

# %% [markdown]
# **Question.** Using the plot, at roughly which fraction of the chain does the SNR fall below
# $10^{-2}$ for each schedule? Why is it wasteful if a large part of the chain has SNR $\approx 0$?
#
# <!-- BEGIN SOLUTION -->
# For the linear schedule the SNR drops below $10^{-2}$ after roughly two thirds of the chain
# ($t\approx 670$ of 1000); for the cosine schedule only after roughly 94 %. In the
# region where SNR $\approx 0$, $\mathbf x_t$ is essentially pure noise for *every* data point,
# so neighbouring steps are nearly identical tasks: predicting $\boldsymbol\epsilon$ from
# $\mathbf x_t\approx\boldsymbol\epsilon$ is trivial and teaches the model little. Training
# samples (uniform in $t$) and sampling steps spent there are wasted; the cosine schedule
# spends them at intermediate noise levels where the denoising problem is informative.
# <!-- END SOLUTION -->

# %% [markdown]
# ## Exercise 10.2 — Forward noising `q_sample`
#
# Implement `q_sample(x0, t, noise, alpha_bar)`, which returns
# $\mathbf x_t = \sqrt{\bar\alpha_t}\,\mathbf x_0 + \sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon$ for a batch.
# Shapes: `x0`, `noise`: `(B, d)`; `t`: `(B,)` long; `alpha_bar`: float32 tensor `(T+1,)`.
# Mind the broadcasting: `alpha_bar[t]` has shape `(B,)`.

# %%
def q_sample(x0, t, noise, alpha_bar):
    ### BEGIN SOLUTION
    ab = alpha_bar[t].reshape(-1, 1)
    return ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise
    ### END SOLUTION

# %% [markdown]
# We check the statistics: for a fixed $\mathbf x_0$ and $t$, the samples must have mean
# $\sqrt{\bar\alpha_t}\mathbf x_0$ and variance $1-\bar\alpha_t$; for standardised data
# ($\operatorname{Var}=1$) the marginal variance must stay $1$ at all $t$ ("variance preserving");
# and different examples in a batch must use their own $t$.

# %%
alpha_bar_t = torch.tensor(ab_lin, dtype=torch.float32)
torch.manual_seed(0)
N = 100_000
x0 = torch.tensor([[2.0, -1.0, 0.5]]).repeat(N, 1)
t = torch.full((N,), 300, dtype=torch.long)
xt = q_sample(x0, t, torch.randn_like(x0), alpha_bar_t)
a = alpha_bar_t[300]
assert xt.shape == (N, 3)
assert torch.allclose(xt.mean(0), a.sqrt() * x0[0], atol=0.02)
assert torch.allclose(xt.var(0), (1 - a).expand(3), rtol=0.03)

x0_std = torch.randn(N, 2)                        # unit-variance "data"
for tt in (1, 100, 500, 1000):
    xt = q_sample(x0_std, torch.full((N,), tt, dtype=torch.long), torch.randn_like(x0_std), alpha_bar_t)
    assert torch.allclose(xt.var(0), torch.ones(2), atol=0.03), tt

x0_b = torch.ones(2, 4); t_b = torch.tensor([0, 1000])
xt_b = q_sample(x0_b, t_b, torch.zeros(2, 4), alpha_bar_t)
assert torch.allclose(xt_b[0], torch.ones(4)) and xt_b[1].abs().max() < 0.01
print("q_sample OK")

# %% [markdown]
# ## Exercise 10.3 — Sinusoidal time embedding
#
# Implement `timestep_embedding(t, dim)` returning a float32 tensor of shape `(B, dim)`:
#
# $$\operatorname{emb}(t) = \big[\sin(\omega_0 t),\dots,\sin(\omega_{h-1} t),\ \cos(\omega_0 t),\dots,\cos(\omega_{h-1} t)\big],\qquad
# \omega_i = \exp\!\Big(-\ln(10000)\cdot\frac{i}{h}\Big),\quad h = \text{dim}/2 .$$
#
# (`dim` is even.) Remember to convert the long tensor `t` to float.

# %%
def timestep_embedding(t, dim, max_period=10000.0):
    ### BEGIN SOLUTION
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    ### END SOLUTION

# %%
e = timestep_embedding(torch.tensor([0, 1, 50]), 16)
assert e.shape == (3, 16) and e.dtype == torch.float32
assert torch.allclose(e[0, :8], torch.zeros(8)) and torch.allclose(e[0, 8:], torch.ones(8))
assert torch.allclose(e[1, 0], torch.sin(torch.tensor(1.0))) and torch.allclose(e[1, 8], torch.cos(torch.tensor(1.0)))
assert torch.allclose(e[2, 7], torch.sin(torch.tensor(50.0 * math.exp(-math.log(10000.0) * 7 / 8))))
assert torch.allclose(e.pow(2)[:, :8] + e.pow(2)[:, 8:], torch.ones(3, 8), atol=1e-6)
print("embedding OK")

# %% [markdown]
# **Question.** Why not feed the raw integer $t$ (or $t/T$) to the network as a single scalar input?
# What do the high- and low-frequency components of the embedding provide?
#
# <!-- BEGIN SOLUTION -->
# A single scalar input is a weak signal: an MLP with smooth activations must learn sharply
# different behaviour for different noise levels from one coordinate, which takes many layers
# and is badly conditioned when $t$ ranges over $[0, 1000]$ (and almost indistinguishable values if
# scaled to $t/T$). The sinusoidal embedding maps $t$ to a high-dimensional vector in which
# different time steps are nearly orthogonal (like Fourier features), so even a linear layer can
# respond very differently to them. The high-frequency components distinguish neighbouring steps;
# the low-frequency components vary slowly and encode the coarse position in the chain, so nearby
# steps have similar embeddings and the network can share what it learns across them.
# <!-- END SOLUTION -->

# %% [markdown]
# ## Exercise 10.4 — The training loss
#
# Implement `diffusion_loss(model, x0, alpha_bar)` (the DDPM "simple" loss) for a batch:
#
# 1. sample $t\sim\mathcal U\{1,\dots,T\}$ independently for every example ($T$ = `len(alpha_bar) - 1`; use `torch.randint`),
# 2. sample $\boldsymbol\epsilon\sim\mathcal N(\mathbf 0, I)$,
# 3. form $\mathbf x_t$ with your `q_sample`,
# 4. return the mean squared error between `model(x_t, t)` and $\boldsymbol\epsilon$.

# %%
def diffusion_loss(model, x0, alpha_bar):
    ### BEGIN SOLUTION
    T = alpha_bar.shape[0] - 1
    t = torch.randint(1, T + 1, (x0.shape[0],))
    noise = torch.randn_like(x0)
    xt = q_sample(x0, t, noise, alpha_bar)
    return F.mse_loss(model(xt, t), noise)
    ### END SOLUTION

# %%
# Two "oracle" models let us check the loss without training anything.
# For data that are exactly zero, x_t = sqrt(1 - abar_t) * eps, so eps = x_t / sqrt(1 - abar_t) is a perfect prediction.
oracle = lambda xt, t: xt / (1 - alpha_bar_t[t]).sqrt().reshape(-1, 1)
zero_model = lambda xt, t: torch.zeros_like(xt)
torch.manual_seed(0)
x_zero = torch.zeros(4096, 3)
assert diffusion_loss(oracle, x_zero, alpha_bar_t).item() < 1e-6
l0 = diffusion_loss(zero_model, x_zero, alpha_bar_t)
assert l0.dim() == 0 and abs(l0.item() - 1.0) < 0.05          # E[eps^2] = 1

seen = []
spy = lambda xt, t: (seen.append(t.clone()), torch.zeros_like(xt))[1]
diffusion_loss(spy, torch.zeros(20_000, 1), alpha_bar_t)
ts = seen[0]
assert ts.dtype == torch.long and ts.min() >= 1 and ts.max() <= T and ts.float().std() > 250
print("loss OK")

# %% [markdown]
# ## Exercise 10.5 — One reverse sampling step
#
# Implement `p_sample_step(model, x, t, betas, alpha_bar)`, one step of DDPM ancestral sampling for a
# Python integer `t >= 1`:
#
# $$\mathbf x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\Big(\mathbf x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\boldsymbol\epsilon_\theta(\mathbf x_t, t)\Big) + \sigma_t\mathbf z,
# \qquad \sigma_t = \sqrt{\beta_t},\quad \mathbf z\sim\mathcal N(\mathbf 0, I) \text{ if } t>1, \text{ else } \mathbf z = \mathbf 0 .$$
#
# The model expects a batch of time steps, so build `torch.full((B,), t, dtype=torch.long)`.
# Use `torch.no_grad()`.

# %%
@torch.no_grad()
def p_sample_step(model, x, t, betas, alpha_bar):
    ### BEGIN SOLUTION
    tt = torch.full((x.shape[0],), t, dtype=torch.long)
    eps = model(x, tt)
    beta, ab = betas[t], alpha_bar[t]
    mean = (x - beta / (1.0 - ab).sqrt() * eps) / (1.0 - beta).sqrt()
    if t > 1:
        return mean + beta.sqrt() * torch.randn_like(x)
    return mean
    ### END SOLUTION

@torch.no_grad()
def sample(model, betas, alpha_bar, n, d=2, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(n, d)
    for t in range(len(betas) - 1, 0, -1):
        x = p_sample_step(model, x, t, betas, alpha_bar)
    return x

# %% [markdown]
# **The test.** If the data distribution is itself Gaussian, $\mathcal N(\mathbf m, \operatorname{diag}\mathbf s^2)$,
# then $q_t(\mathbf x_t)$ is Gaussian too and the optimal noise predictor is known in closed form:
#
# $$\boldsymbol\epsilon^\star(\mathbf x_t, t) = \sqrt{1-\bar\alpha_t}\;\frac{\mathbf x_t - \sqrt{\bar\alpha_t}\,\mathbf m}{\bar\alpha_t\mathbf s^2 + 1-\bar\alpha_t}
# \qquad\Big(= -\sqrt{1-\bar\alpha_t}\,\nabla\log q_t(\mathbf x_t)\Big).$$
#
# Running your sampler with this "oracle" must reproduce the data distribution.

# %%
T_s = 100
betas_s = torch.tensor(betas_from_alpha_bar(linear_alpha_bar(T_s, 1e-4, 0.1)), dtype=torch.float32)
ab_s = torch.tensor(linear_alpha_bar(T_s, 1e-4, 0.1), dtype=torch.float32)
m_true, s_true = torch.tensor([1.5, -0.5]), torch.tensor([0.5, 2.0])

def gaussian_oracle(xt, t):
    ab = ab_s[t].reshape(-1, 1)
    return (1 - ab).sqrt() * (xt - ab.sqrt() * m_true) / (ab * s_true ** 2 + 1 - ab)

x_one = p_sample_step(gaussian_oracle, torch.zeros(5, 2), 1, betas_s, ab_s)
assert x_one.shape == (5, 2)
assert torch.allclose(x_one, p_sample_step(gaussian_oracle, torch.zeros(5, 2), 1, betas_s, ab_s)), "no noise at t = 1"
xs = sample(gaussian_oracle, betas_s, ab_s, n=20_000)
print("sample mean", xs.mean(0).numpy().round(3), " target", m_true.numpy())
print("sample std ", xs.std(0).numpy().round(3), " target", s_true.numpy())
assert torch.allclose(xs.mean(0), m_true, atol=0.1)
assert torch.allclose(xs.std(0), s_true, rtol=0.1)
print("reverse step OK")

# %% [markdown]
# ## Exercise 10.6 — Putting it together: a diffusion model for a Swiss roll
#
# Train a denoiser on a 2-D Swiss roll using **your** schedule, `q_sample`, `diffusion_loss`,
# `timestep_embedding` and `p_sample_step`. Complete the `forward` method of the
# MLP denoiser (concatenate $\mathbf x_t$ with the time embedding and apply `self.net`), then
# write the training loop: 3000 Adam steps with learning rate $2\cdot10^{-3}$ (decayed with
# `CosineAnnealingLR`), batch size 256, $T = 100$, linear schedule $\beta_1 = 10^{-4}$, $\beta_T = 0.1$.
#
# The final check is deliberately generous: at least **60 %** of generated points must lie within
# distance $0.25$ of a training point, and the sample mean and standard deviation must be close to
# those of the data.

# %%
X_roll, _ = make_swiss_roll(n_samples=8000, noise=0.3, random_state=0)
X_roll = X_roll[:, [0, 2]]
X_roll = ((X_roll - X_roll.mean(0)) / X_roll.std(0)).astype(np.float32)
Xr_t = torch.from_numpy(X_roll)
plt.scatter(X_roll[:, 0], X_roll[:, 1], s=1); plt.gca().set_aspect("equal"); plt.title("Swiss roll"); plt.show()

T_r = 100
ab_r = torch.tensor(linear_alpha_bar(T_r, 1e-4, 0.1), dtype=torch.float32)
betas_r = torch.tensor(betas_from_alpha_bar(linear_alpha_bar(T_r, 1e-4, 0.1)), dtype=torch.float32)

class Denoiser(nn.Module):
    def __init__(self, d=2, hidden=128, t_dim=32):
        super().__init__()
        self.t_dim = t_dim
        self.net = nn.Sequential(nn.Linear(d + t_dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, d))

    def forward(self, x, t):
        ### BEGIN SOLUTION
        return self.net(torch.cat([x, timestep_embedding(t, self.t_dim)], dim=-1))
        ### END SOLUTION

torch.manual_seed(0)
model = Denoiser()
losses = []
### BEGIN SOLUTION
opt = torch.optim.Adam(model.parameters(), lr=2e-3)
steps = 3000
lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
for step in range(steps):
    idx = torch.randint(0, Xr_t.shape[0], (256,))
    loss = diffusion_loss(model, Xr_t[idx], ab_r)
    opt.zero_grad(); loss.backward(); opt.step(); lr_sched.step()
    losses.append(loss.item())
### END SOLUTION
plt.plot(np.convolve(losses, np.ones(50) / 50, mode="valid")); plt.xlabel("step"); plt.ylabel("loss"); plt.show()

# %%
x_gen = sample(model, betas_r, ab_r, n=2000, seed=0).numpy()
dist, _ = NearestNeighbors(n_neighbors=1).fit(X_roll).kneighbors(x_gen)
frac_near = float((dist[:, 0] < 0.25).mean())
print(f"fraction within 0.25 of the data: {frac_near:.3f}")
print("mean", x_gen.mean(0).round(3), "std", x_gen.std(0).round(3))

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
axes[0].scatter(X_roll[:2000, 0], X_roll[:2000, 1], s=1); axes[0].set_title("data")
axes[1].scatter(x_gen[:, 0], x_gen[:, 1], s=1, c="C1"); axes[1].set_title("samples")
for ax in axes:
    ax.set_aspect("equal"); ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
plt.show()

# %%
assert frac_near > 0.6
assert np.all(np.abs(x_gen.mean(0)) < 0.25)
assert np.all(np.abs(x_gen.std(0) - 1.0) < 0.25)
print("diffusion model OK")

# %% [markdown]
# ## Exercise 10.7 — Written questions
#
# **(a) Why predict $\boldsymbol\epsilon$?** The network could instead predict $\mathbf x_0$ directly
# (and the two are related by $\mathbf x_0 = (\mathbf x_t - \sqrt{1-\bar\alpha_t}\boldsymbol\epsilon)/\sqrt{\bar\alpha_t}$).
# Give two reasons why the $\boldsymbol\epsilon$-parameterisation works well in practice.
#
# <!-- BEGIN SOLUTION -->
# (1) **Well-scaled targets at every $t$.** $\boldsymbol\epsilon$ always has unit variance, so the
# regression target has the same scale for all noise levels and the plain unweighted MSE is a
# sensible loss (it corresponds to a reweighting of the ELBO that Ho et al. found to give better
# samples). An $\mathbf x_0$-prediction loss implicitly weights steps very differently: at small
# $t$ the target is almost the input, at large $t$ it is nearly unpredictable from $\mathbf x_t$.
# (2) **Easy at low noise, residual form.** At small $t$, $\mathbf x_t\approx\mathbf x_0$ and predicting
# $\mathbf x_0$ amounts to learning the identity with tiny corrections, while the noise is a small
# residual that the network can isolate directly (like a ResNet predicting a residual). (3) The
# $\boldsymbol\epsilon$-prediction is, up to a known scale, the **score** $\nabla\log q_t$,
# connecting the model to denoising score matching and Langevin dynamics, and it makes the
# sampler formula in the lecture exact. (Note: dividing by $\sqrt{\bar\alpha_t}$ when converting
# $\boldsymbol\epsilon$ to $\mathbf x_0$ amplifies errors at large $t$, so in practice both
# parameterisations, and the "$v$-prediction" mixing them, are in use.)
# <!-- END SOLUTION -->
#
# **(b) The number of steps $T$.** What goes wrong if $T$ is very small (say $T = 5$) with
# $\bar\alpha_T$ still close to $0$? And what is the cost of a very large $T$?
#
# <!-- BEGIN SOLUTION -->
# The reverse step $p_\theta(\mathbf x_{t-1}\mid\mathbf x_t)$ is modelled as a Gaussian. This is only
# accurate when each forward step adds little noise ($\beta_t$ small): then the true reverse conditional
# $q(\mathbf x_{t-1}\mid\mathbf x_t)$ is close to Gaussian. With $T=5$ and $\bar\alpha_T\approx0$, each
# $\beta_t$ is huge and the true reverse conditional is multimodal (a noisy point can come from many
# quite different less-noisy points), so a Gaussian with the predicted mean averages the modes and the
# samples are blurry/off-manifold, like a one-step VAE decoder. A very large $T$ makes each step
# accurate, but sampling needs $T$ sequential network evaluations, which is slow; training is
# unaffected in cost per step (we sample one $t$ per example) but each individual $t$ is seen less often.
# DDIM and other ODE solvers decouple the number of *sampling* steps from the $T$ used in training.
# <!-- END SOLUTION -->
#
# **(c) Diffusion as a VAE.** In what sense is a diffusion model a (hierarchical) VAE? Name one way in
# which it differs from the VAE of Lecture 9.
#
# <!-- BEGIN SOLUTION -->
# Both are latent-variable models trained by maximising (a weighted) ELBO: the latents are
# $\mathbf x_1,\dots,\mathbf x_T$, the "encoder" is $q(\mathbf x_{1:T}\mid\mathbf x_0)$ and the
# "decoder" is the chain $p(\mathbf x_T)\prod_t p_\theta(\mathbf x_{t-1}\mid\mathbf x_t)$, and the ELBO splits into
# Gaussian KL terms exactly like the VAE's KL. Differences: the encoder is **fixed** (no parameters,
# no amortised inference network to learn, hence no posterior collapse); the latents have the **same
# dimension** as the data (no compression); there are $T$ layers of latents with a shared network;
# and the prior-matching term is satisfied by construction ($q(\mathbf x_T\mid\mathbf x_0)\approx\mathcal N(\mathbf 0,I)$).
# <!-- END SOLUTION -->
