# %% [markdown]
# # Exercises 7 — Recurrent Neural Networks
#
# Fill in every cell marked `# YOUR CODE HERE` and answer the written questions.
# The `assert` cells check your work: if a cell runs without an error, you are
# probably on the right track. Everything runs on a CPU in a couple of minutes.

# %%
import math

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_num_threads(2)

# %% [markdown]
# ## Exercise 7.1 — An LSTM cell from the equations
#
# Implement one step of an LSTM:
#
# $$
# \begin{aligned}
# [\mathbf{a}_i, \mathbf{a}_f, \mathbf{a}_g, \mathbf{a}_o] &= W_{ih}\mathbf{x}_t + \mathbf{b}_{ih} + W_{hh}\mathbf{h}_{t-1} + \mathbf{b}_{hh}\\
# \mathbf{i}_t = \sigma(\mathbf{a}_i),\quad \mathbf{f}_t &= \sigma(\mathbf{a}_f),\quad \mathbf{g}_t = \tanh(\mathbf{a}_g),\quad \mathbf{o}_t = \sigma(\mathbf{a}_o)\\
# \mathbf{c}_t &= \mathbf{f}_t\odot\mathbf{c}_{t-1} + \mathbf{i}_t\odot\mathbf{g}_t\\
# \mathbf{h}_t &= \mathbf{o}_t\odot\tanh(\mathbf{c}_t)
# \end{aligned}
# $$
#
# The parameters are already created with the **same names and shapes as in `nn.LSTMCell`**
# (`weight_ih`: $(4H, d)$, `weight_hh`: $(4H, H)$, `bias_ih`, `bias_hh`: $(4H,)$), and the four
# blocks are stacked in PyTorch's order **i, f, g, o**. This means we can copy the weights of a
# reference `nn.LSTMCell` into your module with `load_state_dict` and compare outputs.
#
# Inputs: `x` of shape `(B, d)` and an optional `state = (h, c)` with both of shape `(B, H)`
# (zeros if `None`). Return the new `(h, c)`. *Hint:* `torch.chunk(a, 4, dim=-1)`.

# %%
class MyLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size, self.hidden_size = input_size, hidden_size
        k = 1.0 / math.sqrt(hidden_size)
        self.weight_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size).uniform_(-k, k))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size).uniform_(-k, k))
        self.bias_ih = nn.Parameter(torch.empty(4 * hidden_size).uniform_(-k, k))
        self.bias_hh = nn.Parameter(torch.empty(4 * hidden_size).uniform_(-k, k))

    def forward(self, x, state=None):
        B = x.shape[0]
        if state is None:
            zeros = x.new_zeros(B, self.hidden_size)
            state = (zeros, zeros)
        h, c = state
        ### BEGIN SOLUTION
        a = x @ self.weight_ih.T + self.bias_ih + h @ self.weight_hh.T + self.bias_hh
        a_i, a_f, a_g, a_o = torch.chunk(a, 4, dim=-1)
        i, f, g, o = torch.sigmoid(a_i), torch.sigmoid(a_f), torch.tanh(a_g), torch.sigmoid(a_o)
        c = f * c + i * g
        h = o * torch.tanh(c)
        ### END SOLUTION
        return h, c

# %%
torch.manual_seed(1)
d, H, B, T = 5, 7, 3, 6
ref = nn.LSTMCell(d, H)
mine = MyLSTMCell(d, H)
mine.load_state_dict(ref.state_dict())          # copy the weights

xs = torch.randn(T, B, d)
state_ref = state_mine = None
for t in range(T):                              # unroll over a short sequence
    state_ref = ref(xs[t], state_ref)
    state_mine = mine(xs[t], state_mine)
    assert state_mine[0].shape == (B, H) and state_mine[1].shape == (B, H)
assert torch.allclose(state_mine[0], state_ref[0], atol=1e-6), "hidden states differ"
assert torch.allclose(state_mine[1], state_ref[1], atol=1e-6), "cell states differ"

# Gradients must match too.
(state_ref[0].sum() + state_ref[1].pow(2).sum()).backward()
(state_mine[0].sum() + state_mine[1].pow(2).sum()).backward()
for name, p in mine.named_parameters():
    assert torch.allclose(p.grad, dict(ref.named_parameters())[name].grad, atol=1e-5), name
print("LSTM cell OK")

# %% [markdown]
# ## Exercise 7.2 — Gradient clipping by norm
#
# Implement `clip_grad_norm(parameters, max_norm)`:
#
# 1. compute the **global** norm $\lVert\mathbf{g}\rVert = \sqrt{\sum_p \lVert \nabla_p \rVert^2}$
#    over all parameters that have a gradient (skip those whose `.grad` is `None`);
# 2. if it exceeds `max_norm`, multiply every gradient **in place** by
#    $\text{max\_norm} / (\lVert\mathbf{g}\rVert + 10^{-6})$ (the small constant matches PyTorch);
# 3. return the norm **before** clipping as a tensor.
#
# Modify gradients without recording them in the autograd graph (use `torch.no_grad()` and
# `p.grad.mul_(...)`).

# %%
def clip_grad_norm(parameters, max_norm):
    grads = [p.grad for p in parameters if p.grad is not None]
    ### BEGIN SOLUTION
    total_norm = torch.sqrt(sum((g.detach() ** 2).sum() for g in grads))
    coef = max_norm / (total_norm + 1e-6)
    if coef < 1:
        with torch.no_grad():
            for g in grads:
                g.mul_(coef)
    return total_norm
    ### END SOLUTION

# %%
def two_identical_models_with_grads(scale):
    torch.manual_seed(0)
    m1 = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 3))
    m2 = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 3))
    m2.load_state_dict(m1.state_dict())
    x = torch.randn(16, 4)
    for m in (m1, m2):
        (scale * m(x).pow(2).sum()).backward()
    return m1, m2

for scale, max_norm in [(100.0, 1.0), (0.001, 1.0), (5.0, 0.5)]:
    m1, m2 = two_identical_models_with_grads(scale)
    n_mine = clip_grad_norm(list(m1.parameters()), max_norm)
    n_ref = torch.nn.utils.clip_grad_norm_(m2.parameters(), max_norm)
    assert torch.is_tensor(n_mine)
    assert torch.allclose(n_mine, n_ref, rtol=1e-5), (n_mine, n_ref)
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(p1.grad, p2.grad, rtol=1e-5, atol=1e-8)
    new_norm = torch.sqrt(sum((p.grad ** 2).sum() for p in m1.parameters()))
    print(f"scale {scale:>7}: norm before {n_mine.item():10.4f}, after {new_norm.item():.4f}")
    assert new_norm <= max_norm + 1e-4 or torch.allclose(new_norm, n_ref)

# A parameter without a gradient must simply be ignored.
m1, _ = two_identical_models_with_grads(1.0)
extra = nn.Parameter(torch.zeros(3))
clip_grad_norm(list(m1.parameters()) + [extra], 1.0)
assert extra.grad is None
print("gradient clipping OK")

# %% [markdown]
# **Question.** Clipping changes the *length* of the gradient but not its *direction*.
# Explain why this keeps a step that hits a "cliff" in the loss landscape safe, and why
# the same trick cannot cure vanishing gradients.
#
# <!-- BEGIN SOLUTION -->
# Near a cliff the gradient is huge, so a plain step $-\eta\mathbf{g}$ would jump very far,
# typically right off the region where the local linear approximation is valid, and destroy
# progress. Clipping keeps the descent direction (still a descent direction for small enough
# steps) but caps the step length at $\eta c$, so the update stays in the region where the gradient
# is informative. Vanishing gradients are the opposite problem: the component of the gradient
# that would carry information about distant time steps is (exponentially) close to zero and is
# swamped by the contributions from recent steps. Rescaling the whole vector cannot recover a
# component that is essentially absent, and *increasing* the norm would also amplify the
# short-range terms. The fix must be architectural (gates/additive paths) or in the
# initialisation, not in the optimiser.
# <!-- END SOLUTION -->

# %% [markdown]
# ## Exercise 7.3 — Which came first? (sequence classification)
#
# Each sequence has length $T=12$ over the alphabet $\{0,\dots,5\}$. Tokens 0–3 are random
# distractors; exactly one position holds the token **A = 4** and exactly one holds **B = 5**.
# The label is 1 if A occurs **before** B and 0 otherwise.
#
# Note that a model that only counts tokens ("bag of words") cannot do better than chance:
# every sequence contains one A and one B. The model must use the **order** — this is a
# small version of the "temporal order" problem of Hochreiter & Schmidhuber (1997).
#
# Build a many-to-one recurrent classifier (embedding or one-hot input, then `nn.GRU` or
# `nn.LSTM` with `batch_first=True`, then a linear layer on the last hidden state) and train it
# with Adam on freshly generated batches from `order_batch`. Reach at least **90 %** test accuracy.
# A few hundred steps with a hidden size of 32 is plenty. Store the model in `model`; it must
# map a `LongTensor` of shape `(B, T)` to logits of shape `(B, 2)`.

# %%
T_ORDER, VOCAB = 12, 6

def order_batch(B, gen):
    x = torch.randint(0, 4, (B, T_ORDER), generator=gen)
    pos = torch.stack([torch.randperm(T_ORDER, generator=gen)[:2] for _ in range(B)])  # (B, 2)
    rows = torch.arange(B)
    x[rows, pos[:, 0]] = 4          # A
    x[rows, pos[:, 1]] = 5          # B
    y = (pos[:, 0] < pos[:, 1]).long()
    return x, y

gen = torch.Generator().manual_seed(0)
xb, yb = order_batch(4, gen)
print(xb); print(yb)

# %%
torch.manual_seed(0)
gen = torch.Generator().manual_seed(1)
### BEGIN SOLUTION
class OrderClassifier(nn.Module):
    def __init__(self, vocab, E=16, H=32):
        super().__init__()
        self.emb = nn.Embedding(vocab, E)
        self.rnn = nn.GRU(E, H, batch_first=True)
        self.head = nn.Linear(H, 2)

    def forward(self, x):
        out, _ = self.rnn(self.emb(x))
        return self.head(out[:, -1])

model = OrderClassifier(VOCAB)
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
losses = []
for step in range(800):
    x, y = order_batch(64, gen)
    loss = F.cross_entropy(model(x), y)
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    losses.append(loss.item())
plt.plot(losses); plt.xlabel("step"); plt.ylabel("cross-entropy"); plt.show()
### END SOLUTION

# %%
test_gen = torch.Generator().manual_seed(123)
x_test, y_test = order_batch(2000, test_gen)
with torch.no_grad():
    logits = model(x_test)
assert logits.shape == (2000, 2)
order_acc = (logits.argmax(-1) == y_test).float().mean().item()
print(f"test accuracy: {order_acc:.3f}")
assert order_acc >= 0.90

# %% [markdown]
# ## Exercise 7.4 — Temperature sampling
#
# Implement
#
# * `temperature_probs(logits, temperature)`: the softmax of `logits / temperature` along the
#   last dimension, and
# * `sample_next(logits, temperature, generator)`: for logits of shape `(B, V)` return a
#   `LongTensor` of shape `(B,)` with one sampled index per row. If `temperature == 0`
#   return the **argmax** (greedy decoding) instead of sampling. Use `torch.multinomial`
#   and pass the generator on so that results are reproducible.

# %%
def temperature_probs(logits, temperature):
    ### BEGIN SOLUTION
    return F.softmax(logits / temperature, dim=-1)
    ### END SOLUTION


def sample_next(logits, temperature=1.0, generator=None):
    ### BEGIN SOLUTION
    if temperature == 0:
        return logits.argmax(dim=-1)
    probs = temperature_probs(logits, temperature)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
    ### END SOLUTION

# %%
z = torch.tensor([2.0, 1.0, 0.0, -1.0])
p1 = temperature_probs(z, 1.0)
assert torch.allclose(p1, F.softmax(z, dim=-1))
assert torch.allclose(temperature_probs(z, 1e-3), torch.tensor([1.0, 0.0, 0.0, 0.0]), atol=1e-6)
assert torch.allclose(temperature_probs(z, 1e4), torch.full((4,), 0.25), atol=1e-3)
# Entropy must increase with temperature.
ent = [-(temperature_probs(z, t) * temperature_probs(z, t).log()).sum().item() for t in [0.5, 1, 2, 4]]
assert all(a < b for a, b in zip(ent, ent[1:])), ent

g = torch.Generator().manual_seed(0)
Z = z.repeat(20000, 1)                                  # (20000, 4)
s = sample_next(Z, 1.0, g)
assert s.shape == (20000,) and s.dtype == torch.long
freq = torch.bincount(s, minlength=4).float() / len(s)
print("empirical:", freq.numpy().round(3), " expected:", p1.numpy().round(3))
assert torch.allclose(freq, p1, atol=0.02)
s2 = sample_next(Z, 0.5, g)
assert torch.allclose(torch.bincount(s2, minlength=4).float() / len(s2), temperature_probs(z, 0.5), atol=0.02)
Zr = torch.randn(5, 10)
assert torch.equal(sample_next(Zr, 0), Zr.argmax(-1))
# Same generator seed -> same samples.
a = sample_next(Z[:100], 1.0, torch.Generator().manual_seed(7))
b = sample_next(Z[:100], 1.0, torch.Generator().manual_seed(7))
assert torch.equal(a, b)
print("temperature sampling OK")

# %% [markdown]
# ## Exercise 7.5 — Vanishing gradients (pen and paper, then a check)
#
# Consider the *linear* RNN $\mathbf{h}_t = W\mathbf{h}_{t-1} + U\mathbf{x}_t$ with
# $W = \rho\,Q$, where $Q$ is an orthogonal matrix and $\rho > 0$.
#
# 1. Show that $\big\lVert \partial\mathbf{h}_T / \partial\mathbf{h}_0 \big\rVert_2 = \rho^T$.
# 2. For $\rho = 0.9$, by what factor is the gradient signal from $\mathbf{h}_0$ reduced after
#    $T = 50$ steps? After how many steps has it dropped below $10^{-3}$?
# 3. Now consider an LSTM whose forget gate is constant, $\mathbf{f}_t = f\,\mathbf{1}$. Along the
#    cell-state path, what is the corresponding factor? What value of the forget-gate
#    bias would you need for this path to keep at least 10 % of the signal over 50 steps?
#
# <!-- BEGIN SOLUTION -->
# 1. $\partial\mathbf{h}_T/\partial\mathbf{h}_0 = W^T = \rho^T Q^T$ (the matrix power, not the transpose).
#    A product of orthogonal matrices is orthogonal, so all its singular values are 1 and
#    $\lVert W^T\rVert_2 = \rho^T$.
# 2. $0.9^{50} \approx 5.2\times10^{-3}$, i.e. a reduction by a factor of almost 200.
#    $0.9^T < 10^{-3}$ when $T > \ln(10^{-3})/\ln(0.9) \approx 65.6$, i.e. after 66 steps.
# 3. Along the cell path $\partial\mathbf{c}_T/\partial\mathbf{c}_0 \approx \operatorname{diag}(f^T)$,
#    so the factor is $f^T$ — again exponential, but now $f$ is a *learned, input-dependent* gate
#    that can be pushed arbitrarily close to 1, with no weight matrix or $\tanh'$ in the product.
#    Requiring $f^{50}\ge 0.1$ gives $f \ge 0.1^{1/50} \approx 0.955$, i.e. a pre-activation of
#    $\sigma^{-1}(0.955) = \ln(0.955/0.045) \approx 3.06$. A forget bias of about 3 (with the
#    other contributions to the pre-activation near zero) does the job — which is why
#    initialising the forget bias to 1–3 is a common trick.
# <!-- END SOLUTION -->
#
# The cell below checks part 1 numerically with autograd (no code to write).

# %%
torch.manual_seed(0)
Hq, rho, T_check = 16, 0.9, 50
Q, _ = torch.linalg.qr(torch.randn(Hq, Hq, dtype=torch.float64))
W = rho * Q

def unroll(h0):
    h = h0
    for _ in range(T_check):
        h = W @ h
    return h

J = torch.autograd.functional.jacobian(unroll, torch.randn(Hq, dtype=torch.float64))  # (Hq, Hq)
print(f"||dh_T/dh_0||_2 = {torch.linalg.matrix_norm(J, ord=2).item():.6e},  rho^T = {rho ** T_check:.6e}")
assert abs(torch.linalg.matrix_norm(J, ord=2).item() - rho ** T_check) < 1e-10
