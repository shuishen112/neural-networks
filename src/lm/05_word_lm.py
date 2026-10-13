# %% [markdown]
# # LM 5 — Word-level language models, overfitting and its cures
#
# *Companion to Lecture 5 — Regularisation.*
#
# So far our models predicted characters. Here we move up to **words**: we write a
# tokeniser, build a vocabulary, meet Zipf's law and the unknown-word problem, and
# train an MLP that predicts the next word of Shakespeare from the previous five.
# Word-level models have many more parameters than training examples per
# parameter, so they **overfit fast** — the perfect testbed for Lecture 5's
# regularisers: dropout, weight decay, early stopping and a language-model
# speciality, **weight tying**, which every GPT still uses.
#
# > **Language-model track:** 1 Bigram · 2 MLP LM · 3 Training dynamics · 4 PyTorch · **5 Word-level & regularisation** · 6 WaveNet · 7 Char-RNN · 8 GPT · 9 Sequence VAE · 10 Diffusion LM
#
# **Plan.**
# 1. A word tokeniser
# 2. Zipf's law
# 3. A closed vocabulary with `<unk>`; out-of-vocabulary rate
# 4. A fixed-context MLP word model — and how it overfits
# 5. Regularisers on the same budget: dropout, weight decay, early stopping, weight tying
# 6. Word-level versus character-level perplexity (bits per character)
# 7. Sampling, and where word-level modelling breaks down

# %%
import copy
import math
import re
import time
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
plt.rcParams["figure.figsize"] = (5, 4)
plt.rcParams["axes.grid"] = True

device = "cuda" if torch.cuda.is_available() else "cpu"
print("PyTorch", torch.__version__, "| device:", device)

# %%
from pathlib import Path
import urllib.request

def load_text(name):
    """Load a corpus from ../data (repository checkout) or download it (e.g. on Colab)."""
    for folder in (Path("data"), Path("../data")):
        if (folder / name).exists():
            return (folder / name).read_text(encoding="utf-8")
    url = f"https://raw.githubusercontent.com/shuishen112/neural-networks/main/data/{name}"
    return urllib.request.urlopen(url).read().decode("utf-8")

# %%
text = load_text("tinyshakespeare.txt")
n_split = int(0.9 * len(text))
train_text, val_text = text[:n_split], text[n_split:]
print(f"{len(text):,} characters | train {len(train_text):,} | val {len(val_text):,}")
print(train_text[:300])

# %% [markdown]
# ## 1. A word tokeniser
#
# A **tokeniser** turns a string into a list of tokens. Ours is deliberately simple:
#
# * lower-case everything (so `The` and `the` are one token);
# * a *word* is a run of letters, optionally with internal apostrophes
#   (`i'll`, `o'er`, `answer'd` are single tokens);
# * every punctuation character is its own token, and so is the **newline** — line
#   breaks carry real structure in a play (speaker names, verse lines);
# * spaces are discarded: when we turn tokens back into text we re-insert them.
#
# One regular expression does it: `[a-z]+(?:'[a-z]+)*` for words, `\n` for line
# breaks, `[^\sa-z]` for any other non-space character.

# %%
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)*|\n|[^\sa-z]")

def tokenize(s):
    return TOKEN_RE.findall(s.lower())

def detokenize(tokens):
    s = " ".join(tokens)
    s = re.sub(r" ?\n ?", "\n", s)                  # no spaces around line breaks
    s = re.sub(r" ([.,;:!?])", r"\1", s)            # no space before punctuation
    return s

example = "First Citizen:\nWe are accounted poor citizens, the patricians good.\nI'll speak o'er it!"
print(tokenize(example))
print(detokenize(tokenize(example)))

train_tokens = tokenize(train_text)
val_tokens = tokenize(val_text)
print(f"train: {len(train_tokens):,} tokens | val: {len(val_tokens):,} tokens | "
      f"{len(val_text) / len(val_tokens):.2f} characters per token")

# %% [markdown]
# Note what the round trip loses: capital letters and the exact spacing.
# Tokenisers used in LLMs (LM 8) are **lossless** — `decode(encode(s)) == s` for
# every string — because a model that cannot write a capital letter or an
# indented line of code is of limited use.
#
# ## 2. Zipf's law
#
# Count the tokens and sort them by frequency. Zipf's law says that the frequency
# of the token of rank $r$ falls off like a power law,
#
# $$f(r) \approx \frac{C}{r^{\alpha}}, \qquad \alpha\approx 1,$$
#
# a straight line of slope $-\alpha$ on a log–log plot: a few tokens are extremely
# common, and there is a very long tail of rare ones.

# %%
counts = Counter(train_tokens)
print(f"distinct tokens in train: {len(counts):,}")
print("most common:", [(t if t != "\n" else "\\n", c) for t, c in counts.most_common(15)])

freqs = np.array(sorted(counts.values(), reverse=True), dtype=float)
ranks = np.arange(1, len(freqs) + 1)
fit_range = (ranks >= 10) & (ranks <= 2000)
slope, intercept = np.polyfit(np.log(ranks[fit_range]), np.log(freqs[fit_range]), 1)
singletons = (freqs == 1).sum()
print(f"fitted exponent alpha = {-slope:.2f} (ranks 10–2000)")
print(f"tokens seen exactly once: {singletons:,} ({singletons / len(freqs):.0%} of the distinct tokens)")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].loglog(ranks, freqs, ".", ms=2, label="Shakespeare (train)")
axes[0].loglog(ranks, np.exp(intercept) * ranks ** slope, "k--", lw=1, label=f"$C / r^{{{-slope:.2f}}}$")
axes[0].set_xlabel("rank r"); axes[0].set_ylabel("frequency f(r)"); axes[0].set_title("Zipf's law"); axes[0].legend()
coverage = np.cumsum(freqs) / freqs.sum()
axes[1].semilogx(ranks, coverage)
axes[1].axvline(2000, color="C3", ls=":", label="our vocabulary (2 000)")
axes[1].set_xlabel("vocabulary size (most frequent tokens)"); axes[1].set_ylabel("fraction of train tokens covered")
axes[1].set_title("Coverage"); axes[1].legend()
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 3. A closed vocabulary and `<unk>`
#
# Every token in the vocabulary needs a row in the embedding table *and* a row in
# the output layer, so we cannot afford all of them (and most rare words are seen
# too few times to learn anything about them anyway). We keep the 2 000 most
# frequent training tokens and map everything else to a special **`<unk>`**
# ("unknown") token at index 0. The **out-of-vocabulary (OOV) rate** — the fraction
# of validation tokens that become `<unk>` — measures what we lose.

# %%
VOCAB_SIZE = 2000
itos = ["<unk>"] + [t for t, _ in counts.most_common(VOCAB_SIZE)]
stoi = {t: i for i, t in enumerate(itos)}
V = len(itos)
UNK = stoi["<unk>"]

def encode(tokens):
    return torch.tensor([stoi.get(t, UNK) for t in tokens], dtype=torch.long)

train_ids, val_ids = encode(train_tokens), encode(val_tokens)
print(f"V = {V} (2 000 tokens + <unk>)")
print(f"train tokens mapped to <unk>: {(train_ids == UNK).float().mean().item():.1%}")
print(f"val OOV rate               : {(val_ids == UNK).float().mean().item():.1%}")

print(f"\n{'vocab size':>10s} {'val OOV rate':>13s}")
for k in (500, 1000, 2000, 5000, len(counts)):
    vocab_k = {t for t, _ in counts.most_common(k)}
    oov = sum(t not in vocab_k for t in val_tokens) / len(val_tokens)
    print(f"{k:10d} {oov:13.1%}")

oov_examples = Counter(t for t in val_tokens if t not in stoi).most_common(12)
print("\nfrequent OOV tokens in val:", oov_examples)

# %% [markdown]
# The most frequent unknown tokens are the **names of characters** who only appear
# in the plays at the end of the file (our validation text) — a vivid example of
# a distribution shift between training and test data. Even with *every*
# training token in the vocabulary, a few percent of validation tokens have never
# been seen: Zipf's long tail guarantees that new text brings new words. A closed
# vocabulary will always need `<unk>`.
#
# ## 4. A fixed-context MLP word model
#
# The model is the Bengio MLP of LM 2–4 with words instead of characters: embed
# the previous $n=5$ tokens, concatenate, one $\tanh$ hidden layer, logits over the
# $V=2\,001$ tokens. To make the regularisation experiment fast **and** clearly
# visible we train on a **subset of the first 30 000 training tokens** (about an
# eighth of the training text), and validate on the full validation text.

# %%
CONTEXT = 5
N_TRAIN_TOKENS = 30_000

def make_windows(ids, context=CONTEXT):
    """All (previous `context` tokens -> next token) pairs: X (N, context), Y (N,)."""
    n = len(ids) - context
    X = torch.stack([ids[i:i + n] for i in range(context)], dim=1)
    Y = ids[context:]
    return X, Y

Xtr, Ytr = make_windows(train_ids[:N_TRAIN_TOKENS])
Xva, Yva = make_windows(val_ids)
print("train", tuple(Xtr.shape), "| val", tuple(Xva.shape))
for k in range(3):
    print([itos[i] for i in Xtr[k].tolist()], "->", repr(itos[Ytr[k].item()]))
assert torch.equal(Xtr[1, :-1], Xtr[0, 1:]) and Xtr[1, -1] == Ytr[0]

# %% [markdown]
# The model has switches for every regulariser of this notebook. Dropout acts on
# the concatenated embeddings and on the hidden layer. Weight tying (section 5.4)
# needs a small projection from the hidden layer back to the embedding size.
# We initialise the embeddings with standard deviation 0.1 (rather than
# `nn.Embedding`'s default of 1) so that all variants — tied or not — start near
# the uniform distribution, with an initial loss of about $\ln V \approx 7.6$.

# %%
class WordMLP(nn.Module):
    def __init__(self, vocab_size, context=CONTEXT, d_emb=64, d_hidden=256,
                 p_emb=0.0, p_hidden=0.0, tie_weights=False):
        super().__init__()
        self.context = context
        self.tie_weights = tie_weights
        self.emb = nn.Embedding(vocab_size, d_emb)
        self.drop_emb = nn.Dropout(p_emb)
        self.hidden = nn.Linear(context * d_emb, d_hidden)
        self.drop_hidden = nn.Dropout(p_hidden)
        if tie_weights:
            self.to_emb = nn.Linear(d_hidden, d_emb)      # back to the embedding space
            self.out = nn.Linear(d_emb, vocab_size)
            self.out.weight = self.emb.weight             # ONE shared (V, d_emb) parameter
        else:
            self.out = nn.Linear(d_hidden, vocab_size)
        nn.init.normal_(self.emb.weight, std=0.1)

    def forward(self, idx):                               # idx: (B, context) long
        e = self.drop_emb(self.emb(idx)).flatten(1)       # (B, context * d_emb)
        h = self.drop_hidden(torch.tanh(self.hidden(e)))  # (B, d_hidden)
        if self.tie_weights:
            h = self.to_emb(h)                            # (B, d_emb)
        return self.out(h)                                # (B, V) logits


def n_params(model):
    return sum(p.numel() for p in model.parameters())     # shared parameters are counted once

print(WordMLP(V))
print("parameters:", n_params(WordMLP(V)))

# %% [markdown]
# **Training.** The same loop for every experiment: AdamW (with `weight_decay`
# applied only to weight matrices, as in Lecture 5), a fixed number of steps, and
# every `eval_every` steps an evaluation of the loss on a fixed training subset and
# on the validation set — both in `eval()` mode, so that runs with and without
# dropout are measured the same way. We also keep a copy of the parameters with the
# best validation loss seen so far (`copy.deepcopy` — `state_dict()` alone returns
# references to the live tensors). To save time, the curves use the first 5 000
# windows of each split; final numbers use the whole validation set.

# %%
def make_optimizer(model, lr, weight_decay):
    decay = [p for p in model.parameters() if p.ndim >= 2]        # embeddings and weight matrices
    no_decay = [p for p in model.parameters() if p.ndim < 2]      # biases
    return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}], lr=lr)


@torch.no_grad()
def evaluate(model, X, Y, batch_size=4096):
    """Mean NLL in nats per token."""
    was_training = model.training
    model.eval()
    total = 0.0
    for i in range(0, X.shape[0], batch_size):
        xb, yb = X[i:i + batch_size].to(device), Y[i:i + batch_size].to(device)
        total += F.cross_entropy(model(xb), yb, reduction="sum").item()
    model.train(was_training)
    return total / X.shape[0]


def train(model, steps=1500, batch_size=64, lr=2e-3, weight_decay=0.0, eval_every=100, seed=0):
    optimizer = make_optimizer(model, lr, weight_decay)
    g = torch.Generator().manual_seed(seed)
    hist = {"step": [], "train": [], "val": []}
    best = {"val": float("inf"), "step": 0, "state": None}
    model.train()
    for step in range(steps + 1):
        if step % eval_every == 0:
            tr = evaluate(model, Xtr[:5000], Ytr[:5000])
            va = evaluate(model, Xva[:5000], Yva[:5000])
            hist["step"].append(step); hist["train"].append(tr); hist["val"].append(va)
            if va < best["val"]:
                best = {"val": va, "step": step, "state": copy.deepcopy(model.state_dict())}
        if step == steps:
            break
        ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
        xb, yb = Xtr[ix].to(device), Ytr[ix].to(device)
        loss = F.cross_entropy(model(xb), yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return hist, best


results = {}

def run(name, steps=1500, weight_decay=0.0, **model_kwargs):
    torch.manual_seed(0)
    model = WordMLP(V, **model_kwargs).to(device)
    t0 = time.perf_counter()
    hist, best = train(model, steps=steps, weight_decay=weight_decay)
    final_val = evaluate(model, Xva, Yva)
    best_model = WordMLP(V, **model_kwargs).to(device)
    best_model.load_state_dict(best["state"])
    best_val = evaluate(best_model, Xva, Yva)
    results[name] = dict(model=model, best_model=best_model, hist=hist, best_step=best["step"],
                         final_val=final_val, best_val=best_val, params=n_params(model))
    print(f"{name:22s} | {n_params(model):9,d} params | final val NLL {final_val:.3f} (ppl {math.exp(final_val):6.1f}) "
          f"| best val NLL {best_val:.3f} (ppl {math.exp(best_val):6.1f}) at step {best['step']} "
          f"| {time.perf_counter() - t0:.1f} s")

# %% [markdown]
# ### 4.1 The unregularised baseline overfits
#
# The baseline has about 0.7 M parameters and sees 30 000 training windows. One
# epoch is about 470 steps of 64 examples, so 1 500 steps are a bit over three
# passes over the data.

# %%
run("baseline")
h = results["baseline"]["hist"]
plt.plot(h["step"], h["train"], label="train")
plt.plot(h["step"], h["val"], label="validation")
plt.axvline(results["baseline"]["best_step"], color="k", ls=":", label="best validation loss")
plt.xlabel("step"); plt.ylabel("NLL (nats / token)"); plt.ylim(2, 8)
plt.title("Unregularised word MLP"); plt.legend(); plt.show()

# %% [markdown]
# The two curves separate almost immediately. The training loss keeps falling —
# the model is memorising which word follows which 5-word context in *these*
# 30 000 tokens — while the validation loss reaches its minimum early (within the
# first epoch) and then **rises**. The model has far more parameters than training
# tokens — the output layer alone has $256\times2\,001\approx 0.5$ M weights,
# against 30 000 training tokens. This is the regime
# Lecture 5 warned about, and the one word-level models are always in: the
# vocabulary makes them wide, and most words are rare.
#
# ## 5. Regularisers on the same budget
#
# Every run below uses the same data, the same 1 500 steps, the same learning rate
# and the same seed. Only the regulariser changes.
#
# ### 5.1 Early stopping
#
# The simplest cure needs no new run: keep the parameters from the step with the
# lowest validation loss (we stored them during training). In practice one would
# also *stop* training a few evaluations after the minimum ("patience") and save
# the compute. Its result is the **best val** number printed above: the
# baseline's best checkpoint is much better than its final state.
#
# ### 5.2 Dropout
#
# Dropout (Lecture 5) on the input embeddings and on the hidden units. Zeroing
# random coordinates of the context-word embeddings forces the model not to rely
# on any single feature of any single context word. (A related trick for LMs, *word
# dropout*, replaces entire context words by `<unk>`.) Remember that dropout is
# only active in `train()` mode; `evaluate` switches to `eval()`.

# %%
run("dropout", p_emb=0.3, p_hidden=0.5)

# %% [markdown]
# ### 5.3 Weight decay (AdamW)
#
# Decoupled weight decay shrinks every weight matrix by a factor $(1-\eta\lambda)$
# per step. For a word model it has a special effect on the **embeddings of rare
# words**: they receive a gradient only in the few steps where they appear, so
# between appearances they decay towards zero. Rare words therefore end up close
# to each other — the model treats them almost like `<unk>` instead of fitting
# their few occurrences exactly.
#
# $\lambda = 1$ looks large, but AdamW multiplies it by the learning rate: each step
# shrinks the weights by a factor $1 - \eta\lambda = 0.998$.

# %%
run("weight decay", weight_decay=1.0)

# %% [markdown]
# ### 5.4 Weight tying
#
# Look at the two big matrices of a language model:
#
# * the **input embedding** $E\in\mathbb R^{V\times d}$: row $E_w$ represents word $w$
#   when it is *read*;
# * the **output layer** $U\in\mathbb R^{V\times d}$: the logit of word $w$ is
#   $z_w = U_w^\top \mathbf h + b_w$, so row $U_w$ represents word $w$ when it is
#   *predicted*.
#
# Both are "one vector per word", and words that are interchangeable as inputs
# (`king`/`duke`, `thou`/`you`) should be interchangeable as outputs too. **Weight
# tying** (Press & Wolf 2017; Inan, Khosravi & Socher 2017) simply sets $U = E$.
# This requires the hidden state to live in the embedding space, so we add a
# small projection $\mathbf h \mapsto P\mathbf h$ with $P\in\mathbb R^{d\times H}$:
#
# $$z_w = E_w^\top (P\mathbf h) + b_w .$$
#
# Why it regularises:
#
# * **Fewer parameters.** We drop $V\cdot H$ output weights and add $d\cdot H$
#   projection weights. Here: $2\,001\cdot256 \approx 512$ k replaced by
#   $64\cdot256 \approx 16$ k — and with $V = 50\,000$ (GPT-2) tying saves tens of
#   millions of parameters.
# * **Every row is trained at every step.** In an untied model the embedding of a
#   rare word is updated only when that word occurs in a context. Through the
#   softmax, every output row gets a gradient at every step (pushed down when the
#   word is not the target), and with tying that gradient reaches the embedding.
#
# In PyTorch tying is one line, `self.out.weight = self.emb.weight`: both modules
# now hold the *same* `nn.Parameter`, autograd adds the two gradient
# contributions, and `model.parameters()` lists it once.

# %%
tied = WordMLP(V, tie_weights=True)
print("same tensor:", tied.out.weight is tied.emb.weight,
      "| same storage:", tied.out.weight.data_ptr() == tied.emb.weight.data_ptr())
print(f"untied: {n_params(WordMLP(V)):,} parameters | tied: {n_params(tied):,} parameters")

run("weight tying", tie_weights=True)

# %% [markdown]
# ### 5.5 Combinations
#
# Regularisers combine. The strengths below were not tuned (try it!).

# %%
run("dropout + wd", p_emb=0.3, p_hidden=0.5, weight_decay=1.0)
run("tying + dropout + wd", tie_weights=True, p_emb=0.3, p_hidden=0.5, weight_decay=1.0)

# %% [markdown]
# ### 5.6 Summary
#
# Perplexity $e^{\text{NLL}}$ is the conventional unit for word models: the model is
# "as uncertain as if it chose uniformly among" that many words. The uniform model
# has perplexity $V=2\,001$. "Final" is the model after 1 500 steps; "best" is the
# best validation checkpoint, i.e. the run *with early stopping*.

# %%
print(f"{'setting':22s} {'params':>9s} {'final val ppl':>14s} {'best val ppl':>13s} {'best step':>10s}")
for name, r in results.items():
    print(f"{name:22s} {r['params']:9,d} {math.exp(r['final_val']):14.1f} "
          f"{math.exp(r['best_val']):13.1f} {r['best_step']:10d}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for i, (name, r) in enumerate(results.items()):
    h = r["hist"]
    axes[0].plot(h["step"], h["val"], color=f"C{i}", label=name)
    axes[0].plot(h["step"], h["train"], color=f"C{i}", ls="--", alpha=0.5)
axes[0].set_ylim(2.5, 7.0); axes[0].set_xlabel("step"); axes[0].set_ylabel("NLL (nats / token)")
axes[0].set_title("validation (solid) and train (dashed)"); axes[0].legend(fontsize=8)
names = list(results)
axes[1].barh(names, [math.exp(results[n]["best_val"]) for n in names], color=[f"C{i}" for i in range(len(names))])
axes[1].set_xlabel("best validation perplexity (lower is better)"); axes[1].invert_yaxis()
plt.tight_layout(); plt.show()

best_name = min(results, key=lambda n: results[n]["best_val"])
best_model = results[best_name]["best_model"]
best_nll = results[best_name]["best_val"]
print(f"best setting: {best_name} | val NLL (nats/token): {best_nll:.3f} | val perplexity: {math.exp(best_nll):.1f}")
assert best_nll < math.log(V)          # far better than uniform guessing

# %% [markdown]
# How to read the table (exact numbers vary a little with the random seed):
#
# * **Final** column. The unregularised model ends far worse than it was at its
#   best. Dropout and weight decay keep the validation loss from blowing up: their
#   final and best values are close, so *when* we stop matters much less. Weight
#   tying alone cuts the parameter count by a factor of three at a similar best
#   perplexity, but on its own it still overfits; it is a way to spend parameters
#   wisely rather than a substitute for the other regularisers.
# * **Best** column. Here the differences are much smaller. With only 30 000
#   training tokens the *data* is what limits the model; early stopping alone
#   captures most of what can be had, and dropout and weight decay add a modest
#   improvement on top. In our run weight decay alone did best; the combinations,
#   whose strengths were not re-tuned together, did not add up. Regularisers
#   interact, and their strengths have to be tuned jointly.
#
# This is typical. Regularisation pays off most when we train longer, or use a
# bigger model, than the data alone would support — dropout in particular slows
# learning down and needs more steps to show its full benefit (Try it yourself, 2).
#
# ## 6. Word-level versus character-level perplexity
#
# A perplexity of 100 per **word** and a perplexity of 5 per **character** cannot be
# compared directly: the units differ. Both models assign a probability to the
# same validation text, though, so we can convert the total log-likelihood to a
# common unit, **bits per character** (bpc):
#
# $$\text{bpc} = \frac{\text{total NLL in nats}}{\ln 2\cdot \#\text{characters}}
# = \frac{\text{NLL per token}}{\ln 2}\cdot\frac{\#\text{tokens}}{\#\text{characters}} .$$
#
# Our tokens average about 3.7 characters (counting the spaces we dropped), so one
# nat per token is worth only about $1/(3.7\ln 2)\approx0.39$ bits per character.

# %%
n_tokens_scored = Yva.shape[0]                          # predictions made on the val set
chars_per_token = len(val_text) / len(val_tokens)
total_nats = best_nll * n_tokens_scored
bpc = total_nats / (math.log(2) * n_tokens_scored * chars_per_token)
print(f"val NLL               : {best_nll:.3f} nats / token  (perplexity {math.exp(best_nll):.1f})")
print(f"characters per token  : {chars_per_token:.3f}")
print(f"bits per character    : {best_nll:.3f} / ({math.log(2):.3f} x {chars_per_token:.3f}) = {bpc:.3f}")
print(f"nats per character    : {bpc * math.log(2):.3f}   (compare: uniform over 65 characters = {math.log(65):.2f})")

# %% [markdown]
# Treat this number with suspicion — it is **flattering**, for two reasons:
#
# 1. *The task is easier.* We lower-cased the text and threw away whitespace, so our
#    model never pays for capital letters or spacing, which a character model must
#    predict.
# 2. *`<unk>` is free.* About one validation token in ten is `<unk>`. Predicting
#    "some rare word" is easy; a character model has to spell out *which* rare word.
#    A fair comparison would add the cost of spelling every `<unk>` (e.g. with a
#    character model), which is substantial.
#
# Comparisons across tokenisations must be made per character (or per byte) of
# the *original* text, and only for lossless tokenisers. LLM papers that report
# bits-per-byte on raw text do exactly this. Keep this calculation in mind for LM 7
# (character LSTM on the same text) and LM 8 (GPT with a BPE tokeniser).
#
# ## 7. Sampling
#
# Generation works as for characters: start from a 5-token prompt, sample the next
# token from the softmax at temperature $\tau$, slide the window, repeat. We forbid
# `<unk>` by setting its logit to $-\infty$ — otherwise the model would happily
# write "the <unk> of <unk>", one of the side effects of a closed vocabulary.

# %%
@torch.no_grad()
def generate(model, prompt, n_tokens=60, temperature=1.0, seed=0):
    model.eval()
    g = torch.Generator(device=device).manual_seed(seed)
    context = [stoi.get(t, UNK) for t in tokenize(prompt)][-model.context:]
    assert len(context) == model.context, "the prompt must contain at least `context` tokens"
    out = []
    for _ in range(n_tokens):
        logits = model(torch.tensor([context], device=device))[0] / temperature
        logits[UNK] = -float("inf")
        ix = torch.multinomial(F.softmax(logits, dim=-1), 1, generator=g).item()
        out.append(itos[ix])
        context = context[1:] + [ix]
    return detokenize(out)

prompt = "\nFirst Citizen:\nWe are"
for tau in (0.7, 1.0):
    print(f"--- temperature {tau} ---")
    print(prompt.strip() + " " + generate(best_model, prompt, temperature=tau, seed=1))

# %% [markdown]
# Locally plausible, globally meaningless: five words of context are enough for
# grammatical fragments and speaker-name formatting, nothing more.
#
# **Where word-level modelling breaks down.**
#
# * **Open vocabulary.** New text always contains unseen words (Zipf), names,
#   numbers, typos, code, other languages. A closed vocabulary maps them all to
#   `<unk>` — the model can neither read nor write them.
# * **No sharing between related words.** `speak`, `speaks`, `spoke`, `speaketh` are
#   four unrelated rows of $E$; each must be learnt from its own occurrences.
# * **Size.** Real word vocabularies run to $10^5$–$10^6$ entries, and both the
#   embedding and the output layer grow linearly with $V$ (the softmax cost too).
#
# Characters (LM 7) fix all three but make sequences ~4× longer. The compromise
# used by every modern LLM is **subword tokenisation**: byte-pair encoding (BPE,
# LM 8) starts from bytes and greedily merges frequent pairs, so common words become
# single tokens and rare words are spelled from a few pieces — no `<unk>` ever.
#
# ## From here to LLMs
#
# * **Weight tying** is standard in GPT-2, many BERT-style models and
#   small-to-medium LLMs (e.g. Gemma); with $V\approx 50\text{k}$–$250\text{k}$ and
#   $d$ in the thousands, the embedding matrix is a large share of a small model's
#   parameters. (Some very large models untie them again, because the saving
#   matters less at scale.)
# * **Dropout** was used in GPT-2-era pre-training but is usually **off** in modern
#   LLM pre-training: with trillions of tokens, each token is seen about once and
#   overfitting is not the problem. It returns for *fine-tuning* on small data
#   (e.g. LoRA dropout).
# * **AdamW weight decay** (typically $\lambda = 0.1$) is in essentially every
#   LLM recipe, usually not applied to biases and normalisation gains.
# * **Early stopping**, in its LLM form, is checkpoint selection on held-out loss,
#   and the reason fine-tuning runs are often only 1–3 epochs.
# * **Perplexity** and **bits per byte** are still the headline intrinsic metrics,
#   and the per-character normalisation above is how models with different
#   tokenisers are compared.
# * The **vocabulary** problem is solved by subwords (LM 8); Zipf's law still governs
#   which tokens get merged first.
#
# ## Try it yourself
#
# 1. Train the baseline on *all* 263 k training tokens for the same 1 500 steps. How
#    much of the overfitting disappears? (More data is the best regulariser.)
# 2. Tune the dropout rates and the weight decay $\lambda$ on validation data. Does
#    the best combination change if you train for 3 000 steps?
# 3. Implement **word dropout**: during training replace each context token by
#    `<unk>` with probability 0.1. Compare with embedding dropout.
# 4. Change `VOCAB_SIZE` to 500 and 5 000. How do the OOV rate, the perplexity and
#    the bits per character move? Why is perplexity alone misleading here?
# 5. After training the tied model, find the nearest neighbours (cosine similarity)
#    of `king`, `thou` and `love` in the embedding table. Do the same for the
#    untied model's input and output embeddings.
# 6. Keep the capital letters (remove `.lower()`). What happens to the vocabulary
#    size and the OOV rate?
#
# **Further reading.**
# Bengio et al., *A Neural Probabilistic Language Model*, JMLR 2003;
# Zipf, *Human Behavior and the Principle of Least Effort*, 1949 (and Piantadosi,
# *Zipf's word frequency law in natural language*, Psychon. Bull. Rev. 2014);
# Srivastava et al., *Dropout*, JMLR 2014; Gal & Ghahramani, *A Theoretically Grounded
# Application of Dropout in Recurrent Neural Networks*, NeurIPS 2016 (embedding/word dropout);
# Loshchilov & Hutter, *Decoupled Weight Decay Regularization*, ICLR 2019;
# Press & Wolf, *Using the Output Embedding to Improve Language Models*, EACL 2017;
# Inan, Khosravi & Socher, *Tying Word Vectors and Word Classifiers*, ICLR 2017;
# Merity, Keskar & Socher, *Regularizing and Optimizing LSTM Language Models* (AWD-LSTM), ICLR 2018;
# Jurafsky & Martin, *Speech and Language Processing*, ch. 2–3 (tokenisation, perplexity).
