# %% [markdown]
# # LM 8 — Byte-pair encoding and a GPT from scratch
#
# *Companion to Lecture 8 — Attention and Transformers.*
#
# This notebook builds the two halves of every modern large language model. First the
# **tokeniser**: we implement **byte-pair encoding** (BPE), the algorithm behind the GPT-2/3/4 and
# Llama vocabularies, train it on Shakespeare and study what it learns. Then the **model**: a
# decoder-only Transformer (**GPT**) written from scratch — explicit causal multi-head attention,
# pre-LN blocks, weight tying — trained with the recipe used for real LLMs (AdamW, warm-up and
# cosine learning-rate decay, gradient clipping), and decoded with temperature, top-$k$ and
# nucleus sampling and a **KV cache**. Apart from scale, this is the architecture of GPT-2.
#
# > **Language-model track:** 1 Bigram · 2 MLP LM · 3 Training dynamics · 4 PyTorch · 5 Word-level & regularisation · 6 WaveNet · 7 Char-RNN · **8 GPT** · 9 Sequence VAE · 10 Diffusion LM

# %%
import math
import re
import time
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)
torch.set_num_threads(2)
device = "cuda" if torch.cuda.is_available() else "cpu"
plt.rcParams["figure.figsize"] = (5, 4)
plt.rcParams["axes.grid"] = True
print("torch", torch.__version__, "| device:", device)

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
n_train = int(0.9 * len(text))
train_text, val_text = text[:n_train], text[n_train:]

chars = sorted(set(text))
V_char = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}

def encode_char(s):
    return [stoi[c] for c in s]

def decode_char(ids):
    return "".join(itos[i] for i in ids)

print(f"train {len(train_text):,} characters, val {len(val_text):,} characters, {V_char} distinct characters")

# %% [markdown]
# ## 1. Tokenisation: between characters and words
#
# The tokeniser decides what one "step" of the language model is. We have now seen both extremes:
#
# * **characters** (LM 7): tiny vocabulary, never an unknown symbol, but long sequences — the
#   model spends capacity and context length on spelling;
# * **words** (LM 5): short sequences, but a huge vocabulary with a long tail of rare words that
#   are seen too few times to learn, and an `<unk>` token for everything unseen.
#
# **Subword** tokenisers sit in between: frequent words become single tokens, rare words are split
# into frequent pieces, and every string can still be encoded. **Byte-pair encoding** (Gage 1994 as
# a compression algorithm; Sennrich et al. 2016 for neural machine translation) builds such a
# vocabulary greedily:
#
# 1. Start with a base vocabulary of single symbols. Following GPT-2 we use the **256 byte values**,
#    so that any UTF-8 text — any language, emoji, code — can be encoded.
# 2. Count all **adjacent pairs** of tokens in the training text.
# 3. Create a new token for the **most frequent pair**, and replace every occurrence of that pair
#    by the new token.
# 4. Repeat steps 2–3 for a chosen number of merges $M$. The vocabulary size is $256 + M$.
#
# The output of training is the ordered list of merges. Encoding a new string replays them.
#
# ### 1.1 The direct implementation
#
# The algorithm fits in a few lines. Each merge needs one pass over the whole sequence to count
# pairs and one to replace — $O(N)$ per merge, $O(MN)$ in total.

# %%
def get_pair_counts(ids, counts=None):
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, new_id):
    """Replace every occurrence of `pair` in the list `ids` by `new_id`."""
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out

def train_bpe_naive(text, n_merges):
    ids = list(text.encode("utf-8"))
    merges = {}                                       # (id, id) -> new id, in creation order
    for k in range(n_merges):
        counts = get_pair_counts(ids)
        pair = max(counts, key=counts.get)
        ids = merge(ids, pair, 256 + k)
        merges[pair] = 256 + k
    return merges

print(merge([5, 6, 6, 7, 5, 6], (5, 6), 99))      # -> [99, 6, 7, 99]

n_sub, n_m = 30_000, 30
t0 = time.perf_counter()
merges_naive = train_bpe_naive(train_text[:n_sub], n_m)
t_naive = time.perf_counter() - t0
est = t_naive / n_m * (len(train_text) / n_sub) * 400
print(f"naive BPE: {n_m} merges on {n_sub:,} characters took {t_naive:.2f} s")
print(f"-> extrapolated to 400 merges on the full training text: ~{est:.0f} s")
vocab_naive = {i: bytes([i]) for i in range(256)}
for (a, b), idx in merges_naive.items():
    vocab_naive[idx] = vocab_naive[a] + vocab_naive[b]
print("first merges:", [vocab_naive[idx].decode() for idx in list(merges_naive.values())[:12]])

# %% [markdown]
# Too slow for comfort in pure Python, and look at the first merges: tokens such as `'e '`,
# `'t '` or `', '` glue the end of one word to the space before the next, so the vocabulary
# fills up with fragments that straddle word boundaries. Real tokenisers fix both problems:
#
# * **Pre-tokenisation.** Split the text into word-like chunks with a regular expression first
#   (GPT-2 uses one that keeps a leading space attached to the word: `" the"`, `" king"`), and
#   never merge across chunk boundaries.
# * **Counting over unique chunks.** Shakespeare's 1 M characters contain ~270 k chunks but only
#   ~14 k *distinct* ones. Store each distinct chunk once with its frequency, and when a pair is
#   merged update the pair counts **incrementally** — only for the chunks that contain the pair
#   (Sennrich et al.'s original implementation does the same).
#
# ### 1.2 An efficient implementation

# %%
# GPT-2-style pre-tokeniser (simplified to ASCII letters, which is all Shakespeare needs):
# contractions | optional space + letters | optional space + digits | optional space + other symbols
# | whitespace not followed by a non-space (so " word" keeps its space) | remaining whitespace
PRETOK = re.compile(r"'(?:s|t|ll|ve|re|d|m)| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+")

chunks = PRETOK.findall(train_text)
assert "".join(chunks) == train_text                  # pre-tokenisation loses nothing
print(f"{len(chunks):,} chunks, {len(set(chunks)):,} distinct;", chunks[:12])

def train_bpe(text, n_merges):
    word_freq = Counter(PRETOK.findall(text))
    words = [list(w.encode("utf-8")) for w in word_freq]      # each distinct chunk as a list of ids
    freqs = list(word_freq.values())
    pair_counts = defaultdict(int)                             # pair -> weighted count
    where = defaultdict(set)                                   # pair -> indices of chunks containing it
    for wi, (w, f) in enumerate(zip(words, freqs)):
        for pair in zip(w, w[1:]):
            pair_counts[pair] += f
            where[pair].add(wi)
    merges = {}
    for k in range(n_merges):
        pair = max(pair_counts, key=pair_counts.get)
        if pair_counts[pair] < 2:
            break
        new_id = 256 + k
        merges[pair] = new_id
        for wi in where.pop(pair):                             # only the chunks that contain the pair
            w, f = words[wi], freqs[wi]
            for p in zip(w, w[1:]):                            # remove the old pairs of this chunk ...
                pair_counts[p] -= f
            w = merge(w, pair, new_id)
            words[wi] = w
            for p in zip(w, w[1:]):                            # ... and add the new ones
                pair_counts[p] += f
                where[p].add(wi)
        del pair_counts[pair]
    return merges

N_MERGES = 400
t0 = time.perf_counter()
merges = train_bpe(train_text, N_MERGES)
t_fast = time.perf_counter() - t0
print(f"efficient BPE: {len(merges)} merges on the full training text ({len(train_text):,} chars) in {t_fast:.2f} s")

# %% [markdown]
# A speed-up of roughly two orders of magnitude, from two ideas: work on distinct chunks
# instead of the raw stream, and update counts locally instead of recounting.
# (Production tokenisers such as `tiktoken` or Hugging Face `tokenizers` are written in Rust and use
# priority queues, but the algorithm is the same.)
#
# ### 1.3 Encoding and decoding
#
# The vocabulary maps every token id to its bytes: the 256 single bytes, and for each merge the
# concatenation of its two parts. **Decoding** concatenates bytes and decodes UTF-8.
# **Encoding** pre-tokenises, then inside each chunk repeatedly applies the *earliest learned*
# merge that is applicable — the same order in which merges were created during training. We cache
# the encoding of each distinct chunk, which makes encoding the whole corpus fast.

# %%
vocab = {i: bytes([i]) for i in range(256)}
for (a, b), idx in merges.items():                    # dicts keep insertion order = merge order
    vocab[idx] = vocab[a] + vocab[b]
V_bpe = 256 + len(merges)

def encode_chunk(b, merges):
    ids = list(b)
    while len(ids) >= 2:
        pair = min(set(zip(ids, ids[1:])), key=lambda p: merges.get(p, math.inf))
        if pair not in merges:
            break                                     # no learned merge applies any more
        ids = merge(ids, pair, merges[pair])
    return ids

def encode_bpe(s, merges=merges):
    cache, out = {}, []
    for chunk in PRETOK.findall(s):
        if chunk not in cache:
            cache[chunk] = encode_chunk(chunk.encode("utf-8"), merges)
        out.extend(cache[chunk])
    return out

def decode_bpe(ids):
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")

for s in ["Hello world!", "ROMEO:\nBut, soft! what light through yonder window breaks?",
          "Göteborg och Stockholm ✓ 🤖", val_text[:5000]]:
    assert decode_bpe(encode_bpe(s)) == s
t0 = time.perf_counter()
val_bpe_ids = encode_bpe(val_text)
assert decode_bpe(val_bpe_ids) == val_text
print(f"round trip decode(encode(s)) == s holds; encoding the validation text took {time.perf_counter() - t0:.2f} s")

def show(ids):
    return [vocab[i].decode("utf-8", errors="replace") for i in ids]

print("\nfirst 24 merges:")
for (a, b), idx in list(merges.items())[:24]:
    print(f"  {idx}: {vocab[a].decode()!r:8s} + {vocab[b].decode()!r:8s} -> {vocab[idx].decode()!r}")
print("\nlast 8 merges:", [vocab[idx].decode() for idx in list(merges.values())[-8:]])
longest = sorted(range(256, V_bpe), key=lambda i: -len(vocab[i]))[:12]
print("longest tokens:", [vocab[i].decode() for i in longest])

# %% [markdown]
# The first merges attach a space to frequent initial letters (`' t'`, `' a'`, `' s'`) and join
# the most common letter pairs of English (`'he'`, `'ou'`, `'in'`); after only a dozen merges the first
# whole word, `' the'`, appears. Later merges create frequent words with their leading space
# (`' should'`, `' father'`) and corpus-specific pieces such as the capitalised speaker names
# (`'ICHARD'` from `RICHARD:`). Now see what happens to words of different frequency:

# %%
for w in [" the", " king", " thou", " Romeo", " hath", " unfortunately", " backpropagation", " Gothenburg"]:
    ids = encode_bpe(w)
    print(f"{w!r:20s} -> {len(ids)} token(s): {show(ids)}")
print("\nA line of the validation set:")
line = "JULIET:\nO Romeo, Romeo! wherefore art thou Romeo?"
print(show(encode_bpe(line)))

# %% [markdown]
# Frequent words are a single token; a rare word is split into frequent pieces (often roughly
# morphological: prefixes, stems and endings); a word never seen in training falls back to small
# pieces and, in the worst case, to single bytes. **Nothing is ever unknown.**
#
# ### 1.4 How many merges? The vocabulary-size trade-off
#
# The **compression ratio** (characters per token) measures how much shorter the sequence gets.
# We re-encode the validation text using only the first $M$ merges:

# %%
Ms = [0, 25, 50, 100, 200, 300, len(merges)]
ratios = []
for M in Ms:
    merges_M = {p: i for p, i in merges.items() if i < 256 + M}
    ratios.append(len(val_text) / len(encode_bpe(val_text, merges_M)))
    print(f"M = {M:3d} merges (vocab {256 + M:3d}): {ratios[-1]:.2f} characters per token")

train_bpe_counts = Counter(encode_bpe(train_text))
plt.figure(figsize=(11, 3.8))
plt.subplot(1, 2, 1)
plt.plot([256 + M for M in Ms], ratios, "o-")
plt.xlabel("vocabulary size $256 + M$"); plt.ylabel("characters per token (val)")
plt.title("Compression")
plt.subplot(1, 2, 2)
plt.plot(range(256, V_bpe), [train_bpe_counts.get(i, 0) for i in range(256, V_bpe)], ".", ms=3)
plt.yscale("log"); plt.xlabel("token id (= merge order)"); plt.ylabel("occurrences in train")
plt.title("Later merges are rarer tokens")
plt.tight_layout(); plt.show()

# %% [markdown]
# The ratio grows quickly at first and then with diminishing returns, because each new merge
# covers rarer patterns. Choosing $M$ is a trade-off:
#
# * **+** shorter sequences: more text per context window, fewer steps to generate, less compute
#   per character;
# * **−** a larger embedding matrix and softmax ($V \times d$ parameters each, unless tied);
# * **−** each token is seen less often in training, so rare tokens are poorly learned;
# * **−** more "spelling" is hidden inside tokens (one reason LLMs are bad at counting letters).
#
# GPT-2 uses 50 257 tokens, Llama 2 32 000, GPT-4 ~100 k and Llama 3 ~128 k — large corpora can
# afford large vocabularies. For our 1 M characters, a few hundred merges is plenty.
#
# ## 2. A GPT from scratch
#
# ### 2.1 Choosing the tokens
#
# We train the GPT on **characters** by default, so its validation loss can be compared directly
# with the char-LSTM of LM 7 (similar parameter count, same batch shape and the same number of
# training steps). Set `TOKENIZER = "bpe"` to train the identical model on the BPE tokens from
# above. Losses per *token* are not comparable across tokenisers — one of our BPE tokens carries
# about 2 characters — so we always report **nats per character**:
#
# $$\text{NLL per char} = \frac{\sum_{\text{tokens}} -\ln p(\text{token}\mid\text{context})}{\text{number of characters these tokens decode to}} .$$

# %%
TOKENIZER = "char"          # "char" or "bpe"

if TOKENIZER == "char":
    vocab_size, tok_encode, tok_decode = V_char, encode_char, decode_char
else:
    vocab_size, tok_encode, tok_decode = V_bpe, encode_bpe, decode_bpe
train_tok = torch.tensor(tok_encode(train_text), dtype=torch.long)
val_tok = torch.tensor(tok_encode(val_text), dtype=torch.long)
print(f"tokenizer {TOKENIZER!r}: vocabulary {vocab_size}, train {len(train_tok):,} tokens, "
      f"val {len(val_tok):,} tokens ({len(val_text) / len(val_tok):.2f} chars/token)")

# %% [markdown]
# A Transformer has no state to carry, so — unlike the LSTM's parallel tracks — each training
# example is an independent random window of `block_size` tokens; inputs and targets are the window
# shifted by one, and the model predicts *all* positions at once.

# %%
def get_batch(data, B, T, gen):
    ix = torch.randint(0, len(data) - T - 1, (B,), generator=gen)       # window starts
    offs = torch.arange(T)
    x = data[ix[:, None] + offs]                                        # (B, T)
    y = data[ix[:, None] + offs + 1]                                    # (B, T), shifted by one
    return x.to(device), y.to(device)

xb, yb = get_batch(train_tok, 2, 16, torch.Generator().manual_seed(0))
print(repr(tok_decode(xb[0].tolist())), "->", repr(tok_decode(yb[0].tolist())))

# %% [markdown]
# ### 2.2 Causal multi-head self-attention
#
# For an input $X \in \mathbb{R}^{T\times d}$ (one sequence), each of the $h$ heads computes
# queries, keys and values $Q = XW_Q$, $K = XW_K$, $V = XW_V$ of width $d_h = d/h$, and
#
# $$\operatorname{head}(X) = \operatorname{softmax}\!\Big(\frac{QK^\top}{\sqrt{d_h}} + M\Big)V,
# \qquad M_{tj} = \begin{cases} 0 & j \le t \\ -\infty & j > t \end{cases}$$
#
# so position $t$ only attends to positions $\le t$ — the **causal mask** that makes a Transformer
# a language model (Lecture 8). The heads are concatenated and projected by $W_O$. In code, one
# linear layer produces $Q$, $K$, $V$ for all heads at once, and the heads become an extra batch
# dimension: shapes $(B, h, T, d_h)$.
#
# The layer also accepts a **cache** `past_kv` of keys and values from earlier positions (used
# for fast generation in Section 4). With a cache of length $P$, the $T$ new queries sit at absolute
# positions $P, \dots, P+T-1$ and the keys at $0,\dots,P+T-1$, so the causal mask is the slice
# `tril[P:P+T, :P+T]` of the full lower-triangular matrix.

# %%
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head, self.head_dim = n_head, n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)            # W_Q, W_K, W_V for all heads
        self.proj = nn.Linear(n_embd, n_embd)               # W_O
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)),
                             persistent=False)
        self.use_sdpa = False                               # explicit attention by default
        self.att = None                                     # last attention weights (for plots)

    def forward(self, x, past_kv=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)                                   # each (B, T, C)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)           # (B, h, T, d_h)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        if past_kv is not None:                                                 # prepend cached keys/values
            k = torch.cat([past_kv[0], k], dim=2)                               # (B, h, P+T, d_h)
            v = torch.cat([past_kv[1], v], dim=2)
        present = (k, v)
        Tk = k.size(2)
        if self.use_sdpa and past_kv is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)        # fused kernel
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)          # (B, h, T, Tk)
            att = att.masked_fill(~self.tril[Tk - T:Tk, :Tk], float("-inf"))
            att = F.softmax(att, dim=-1)
            self.att = att.detach()
            y = att @ v                                                         # (B, h, T, d_h)
        y = y.transpose(1, 2).contiguous().view(B, T, C)                        # concatenate heads
        return self.proj(y), present

# %% [markdown]
# ### 2.3 The GPT
#
# * **Embeddings:** token embedding plus a *learned* position embedding (as in GPT-1/2),
#   $\mathbf{x}_t = E[c_t] + P[t]$.
# * **Pre-LN blocks** (Lecture 8): $X \leftarrow X + \operatorname{MHA}(\operatorname{LN}(X))$, then
#   $X \leftarrow X + \operatorname{MLP}(\operatorname{LN}(X))$ with $\operatorname{MLP}(\mathbf z) =
#   W_2\operatorname{GELU}(W_1\mathbf z)$ of hidden width $4d$.
# * A **final LayerNorm** and a linear **head** to logits — whose weight matrix is **tied** to the
#   token embedding $E$ (Press & Wolf, 2017): the same vector represents a token at the input and
#   scores it at the output. This saves $V\times d$ parameters (a lot for a 50 k vocabulary) and
#   usually helps.
# * **Initialisation** as in GPT-2: weights $\mathcal N(0, 0.02^2)$, and the two projections that
#   write into the residual stream scaled down by $1/\sqrt{2L}$, so that the variance of the
#   residual stream does not grow with depth.

# %%
class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd))

    def forward(self, x, past_kv=None):
        a, present = self.attn(self.ln1(x), past_kv)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, present


class GPT(nn.Module):
    def __init__(self, vocab_size, block_size=64, n_layer=4, n_head=4, n_embd=64):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight                  # weight tying: (V, d) shared
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("attn.proj.weight") or name.endswith("mlp.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, past_kvs=None):
        """idx: (B, T) token ids. past_kvs: None or one (k, v) per block. Returns logits, new caches."""
        B, T = idx.shape
        P = 0 if past_kvs is None else past_kvs[0][0].size(2)          # number of cached positions
        assert P + T <= self.block_size, "sequence longer than the context window"
        pos = torch.arange(P, P + T, device=idx.device)                 # positions continue after the cache
        x = self.tok_emb(idx) + self.pos_emb(pos)                       # (B, T, d)
        presents = []
        for i, blk in enumerate(self.blocks):
            x, present = blk(x, None if past_kvs is None else past_kvs[i])
            presents.append(present)
        return self.head(self.ln_f(x)), presents                        # logits (B, T, V)


def set_sdpa(model, flag):
    for blk in model.blocks:
        blk.attn.use_sdpa = flag

# %% [markdown]
# Before using the fused `F.scaled_dot_product_attention` (which never materialises the $T\times T$
# matrix and is much faster), we check that it computes exactly what our explicit version
# computes, and that the causal mask works: changing a future token must not change earlier
# predictions.

# %%
torch.manual_seed(0)
m = GPT(vocab_size).to(device).eval()
x = torch.randint(0, vocab_size, (3, 64), device=device)
with torch.no_grad():
    set_sdpa(m, False); out_explicit, _ = m(x)
    set_sdpa(m, True); out_sdpa, _ = m(x)
    x2 = x.clone(); x2[:, 40] = (x2[:, 40] + 1) % vocab_size
    out2, _ = m(x2)
print("logits", tuple(out_explicit.shape))
print(f"explicit vs SDPA: max |diff| = {(out_explicit - out_sdpa).abs().max().item():.2e}")
print(f"changing token 40: max change at positions <40: {(out2 - out_sdpa)[:, :40].abs().max().item():.2e},"
      f" at >=40: {(out2 - out_sdpa)[:, 40:].abs().max().item():.2e}")
assert out_explicit.shape == (3, 64, vocab_size)
assert torch.allclose(out_explicit, out_sdpa, atol=1e-5)
assert (out2 - out_sdpa)[:, :40].abs().max().item() < 1e-5

# %% [markdown]
# ### 2.4 Counting parameters
#
# Per block: attention has $4d^2 + 4d$ (Q, K, V, O), the MLP $8d^2 + 5d$, the two LayerNorms $4d$
# — about $12d^2$ in total, the classic estimate $N \approx 12\,L\,d^2$ for the non-embedding
# parameters. `model.parameters()` lists the tied matrix only once.

# %%
BLOCK, N_LAYER, N_HEAD, N_EMBD = 64, 4, 4, 64
torch.manual_seed(0)
model = GPT(vocab_size, BLOCK, N_LAYER, N_HEAD, N_EMBD).to(device)
n_params = sum(p.numel() for p in model.parameters())
n_emb = model.tok_emb.weight.numel() + model.pos_emb.weight.numel()
print(f"token embedding (= tied head) : {model.tok_emb.weight.numel():8,d}")
print(f"position embedding            : {model.pos_emb.weight.numel():8,d}")
print(f"{N_LAYER} blocks                      : {sum(p.numel() for p in model.blocks.parameters()):8,d}"
      f"   (12 L d^2 = {12 * N_LAYER * N_EMBD ** 2:,d})")
print(f"final LayerNorm               : {sum(p.numel() for p in model.ln_f.parameters()):8,d}")
print(f"total                         : {n_params:8,d}")
assert model.head.weight.data_ptr() == model.tok_emb.weight.data_ptr()

# %% [markdown]
# ### 2.5 Training like an LLM
#
# The recipe used (with larger numbers) to train GPT-2/3 and Llama:
#
# * **AdamW** (Loshchilov & Hutter, 2019) with $\beta = (0.9, 0.99)$ and weight decay 0.1 applied
#   to weight *matrices* only — not to biases and LayerNorm gains.
# * **Learning-rate warm-up** for the first steps (Adam's second-moment estimates are unreliable at
#   the start, and large early updates can destabilise attention), then **cosine decay** to 10 %
#   of the peak:
#
# $$\eta_t = \begin{cases} \eta_{\max}\, \frac{t+1}{t_w} & t < t_w \\
# \eta_{\min} + \tfrac12(\eta_{\max} - \eta_{\min})\big(1 + \cos\big(\pi\frac{t - t_w}{t_{\text{total}} - t_w}\big)\big) & t \ge t_w \end{cases}$$
#
# * **Gradient clipping** at global norm 1.
# * **Periodic validation**: every 100 steps on the first ~16 k validation tokens (cheap estimate);
#   at the end on the full validation set.
#
# The budget matches LM 7: 1500 steps of 32 windows × 64 tokens.

# %%
STEPS, BATCH, MAX_LR, WARMUP = 1500, 32, 3e-3, 100

def lr_at(step, max_lr=MAX_LR, warmup=WARMUP, total=STEPS, min_ratio=0.1):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return max_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))

plt.figure(figsize=(5, 2.5))
plt.plot([lr_at(s) for s in range(STEPS)])
plt.xlabel("step"); plt.ylabel("learning rate"); plt.title("Warm-up + cosine schedule"); plt.show()

# %% [markdown]
# **Evaluation.** A GPT sees at most `block_size` tokens. If we cut the validation text into
# non-overlapping windows, the first tokens of every window are predicted with almost no context,
# which inflates the loss. The standard remedy is a **sliding window**: windows overlap by half,
# and in each window we only score the tokens that were not scored before — so every token
# (except the first few) is predicted with at least `block_size/2` tokens of context. Every
# validation token except the very first is scored exactly once.

# %%
@torch.no_grad()
def evaluate(model, ids, stride=None, batch=256):
    """Return (total NLL in nats, number of characters predicted) over the token sequence `ids`."""
    model.eval()
    bs = model.block_size
    stride = stride or bs // 2
    N = len(ids)
    starts = np.array(list(range(0, N - 1 - bs, stride)) + [N - 1 - bs])
    prev_end = np.concatenate([[0], starts[:-1] + bs])
    first = np.maximum(starts, prev_end) - starts            # first position of each window to score
    offs = torch.arange(bs)
    total, n_scored = 0.0, 0
    for i in range(0, len(starts), batch):
        s = torch.as_tensor(starts[i:i + batch])
        x = ids[s[:, None] + offs].to(device)                 # (b, bs)
        y = ids[s[:, None] + offs + 1].to(device)
        logits, _ = model(x)
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none").view(y.shape)
        mask = offs[None, :].to(device) >= torch.as_tensor(first[i:i + batch], device=device)[:, None]
        total += nll[mask].sum().item()
        n_scored += int(mask.sum().item())
    model.train()
    assert n_scored == N - 1                                  # every token but the first, exactly once
    n_chars = len(tok_decode(ids[1:].tolist()))               # characters covered by the scored tokens
    return total, n_chars

def nats_per_char(model, ids, stride=None):
    total, n_chars = evaluate(model, ids, stride)
    return total / n_chars

# %%
def train_gpt(model, steps, max_lr=MAX_LR, warmup=WARMUP, eval_every=100, seed=0, verbose=True):
    model.to(device).train()
    set_sdpa(model, True)                                     # fused attention for speed
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                             {"params": no_decay, "weight_decay": 0.0}], lr=max_lr, betas=(0.9, 0.99))
    gen = torch.Generator().manual_seed(seed)
    val_small = val_tok[: 16 * model.block_size * 16 + 1]
    hist = {"train": [], "val": [], "lr": []}
    t0 = time.time()
    for step in range(steps + 1):
        if step % eval_every == 0 or step == steps:
            v = nats_per_char(model, val_small, stride=model.block_size)
            hist["val"].append((step, v))
            if verbose:
                tr = np.mean(hist["train"][-eval_every:]) if hist["train"] else float("nan")
                print(f"step {step:5d}  lr {lr_at(max(step - 1, 0), max_lr, warmup, steps):.2e}  "
                      f"train {tr:.3f} nats/token  val {v:.3f} nats/char  ({time.time() - t0:5.1f} s)")
        if step == steps:
            break
        lr = lr_at(step, max_lr, warmup, steps)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = get_batch(train_tok, BATCH, model.block_size, gen)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        hist["train"].append(loss.item())
        hist["lr"].append(lr)
    hist["seconds"] = time.time() - t0
    return hist

hist = train_gpt(model, STEPS)
print(f"{1000 * hist['seconds'] / STEPS:.0f} ms per step")

# %%
chars_per_tok = len(val_text) / len(val_tok)
train_curve = np.convolve(hist["train"], np.ones(25) / 25, mode="valid") / chars_per_tok   # ≈ nats/char
plt.figure(figsize=(6, 4))
plt.plot(np.arange(len(train_curve)) + 25, train_curve, label="train (smoothed, ≈ per char)")
plt.plot(*zip(*hist["val"]), "o-", label="validation (non-overlapping windows)")
plt.axhline(math.log(V_char), color="gray", ls=":", label="uniform over characters")
plt.ylim(1.0, 4.5)
plt.xlabel("step"); plt.ylabel("NLL (nats / char)"); plt.title(f"GPT ({TOKENIZER} tokens)"); plt.legend(); plt.show()

nll_nonoverlap = nats_per_char(model, val_tok, stride=BLOCK)
val_nll_gpt = nats_per_char(model, val_tok)
print(f"full validation, non-overlapping windows: {nll_nonoverlap:.3f} nats/char")
print(f"full validation, sliding window (stride {BLOCK // 2}): {val_nll_gpt:.3f} nats/char"
      f" = {val_nll_gpt / math.log(2):.3f} bits/char")

# %% [markdown]
# Compare with the char-LSTM of LM 7, which had about the same number of parameters (~215 k), the
# same batch shape and the same number of steps, and printed its own `val NLL (nats/char)`. At this
# tiny scale the two are typically in the same range — the LSTM even has the advantage of an
# unbounded context. The Transformer's advantages show up with scale: every position is computed in
# parallel (no sequential loop over time), so it uses hardware far better, and attention can
# fetch information from any earlier position without squeezing it through a fixed-size state.
#
# How much does the context help? The loss as a function of the position inside a window
# (non-overlapping windows over part of the validation set):

# %%
with torch.no_grad():
    model.eval()
    n_win = 256
    s = torch.arange(n_win) * BLOCK
    xw = val_tok[s[:, None] + torch.arange(BLOCK)].to(device)
    yw = val_tok[s[:, None] + torch.arange(BLOCK) + 1].to(device)
    lw, _ = model(xw)
    pos_nll = F.cross_entropy(lw.reshape(-1, vocab_size), yw.reshape(-1), reduction="none").view(n_win, BLOCK).mean(0)
    model.train()
plt.figure(figsize=(6, 3.5))
plt.plot(pos_nll.cpu().numpy(), ".-")
plt.xlabel("position in the window (tokens of context)"); plt.ylabel("mean NLL (nats / token)")
plt.title("More context, lower loss"); plt.show()

# %% [markdown]
# ## 3. Generation: temperature, top-$k$ and nucleus (top-$p$) sampling
#
# Given logits $\mathbf z$ for the next token, a sampler chooses a token. All common strategies are
# small transformations of $\mathbf z$ before sampling from the softmax:
#
# * **temperature** $\tau$: $\mathbf z \leftarrow \mathbf z/\tau$ (LM 7);
# * **top-$k$**: keep the $k$ largest logits, set the rest to $-\infty$;
# * **top-$p$ / nucleus** (Holtzman et al., 2020): sort the tokens by probability and keep the
#   *smallest* set whose total probability reaches $p$. Unlike top-$k$ the number of candidates
#   adapts to the model's confidence: after `"wherefore art th"` almost all mass is on `o`, and the
#   nucleus has size 1; at the start of a new sentence it is large.

# %%
def filter_logits(logits, temperature=1.0, top_k=None, top_p=None):
    """logits: (..., V). Returns modified logits; removed tokens get -inf."""
    logits = logits / temperature
    if top_k is not None:
        kth = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[..., -1:]   # (..., 1)
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        mass_before = probs.cumsum(dim=-1) - probs          # probability of all strictly better tokens
        sorted_logits = sorted_logits.masked_fill(mass_before >= top_p, float("-inf"))   # always keeps the best
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    return logits

z = torch.tensor([3.0, 2.5, 1.0, 0.5, 0.0, -1.0])
for kw in [dict(), dict(top_k=2), dict(top_p=0.8), dict(temperature=0.5, top_p=0.8)]:
    p = F.softmax(filter_logits(z, **kw), dim=-1)
    print(f"{str(kw):34s} -> {np.round(p.numpy(), 3)}")

# %% [markdown]
# The nucleus on real predictions of our model: how many tokens are needed for 90 % of the
# probability mass, at a few positions of a validation line?

# %%
probe = val_text[val_text.find("\n", 2000) + 1:][:BLOCK]
probe_ids = torch.tensor([tok_encode(probe)[:BLOCK]], device=device)
with torch.no_grad():
    model.eval()
    probs_all = F.softmax(model(probe_ids)[0][0], dim=-1)      # (T, V)
    model.train()
sorted_p = torch.sort(probs_all, descending=True, dim=-1).values
nucleus = ((sorted_p.cumsum(-1) - sorted_p) < 0.9).sum(-1).cpu().numpy()
labels = [tok_decode([t]).replace("\n", "\\n").replace(" ", "␣") for t in probe_ids[0].tolist()]
plt.figure(figsize=(12, 3))
plt.bar(range(len(nucleus)), nucleus)
plt.xticks(range(len(labels)), labels, fontsize=7)
plt.ylabel("nucleus size (p = 0.9)"); plt.title("Number of candidate next tokens after each token of the line")
plt.show()

# %% [markdown]
# ## 4. The KV cache
#
# Generating $n$ tokens naively re-runs the model on the whole context for every new token:
# $O(n)$ forward passes over sequences of growing length, i.e. $O(n^2)$ token computations (and
# $O(n^3)$ attention work). But in a causal model the keys and values of past positions **never
# change** when new tokens are appended. So we store them — the **KV cache** — and for each new token
# compute only its own query, key and value, attend over the cached keys, and append. Each step then
# processes a *single* position. Two details matter:
#
# * the **position** of the new token is the cache length $P$ (not 0), and
# * the causal mask for $T$ new queries against $P + T$ keys is `tril[P:P+T, :P+T]` (for $T=1$,
#   simply "attend to everything").
#
# Both are implemented in `GPT.forward` / `CausalSelfAttention.forward` above. First a
# correctness check against the full forward pass — once feeding one token at a time after a
# 16-token prompt, and once in chunks of 8 tokens (which tests the offset mask):

# %%
with torch.no_grad():
    model.eval()
    seq = val_tok[:BLOCK].unsqueeze(0).to(device)                      # (1, 64)
    full, _ = model(seq)                                                # (1, 64, V)

    logits, past = model(seq[:, :16])                                   # prefill the prompt
    outs = [logits]
    for t in range(16, BLOCK):                                          # then one token at a time
        logits, past = model(seq[:, t:t + 1], past_kvs=past)
        outs.append(logits)
    inc1 = torch.cat(outs, dim=1)

    past, outs = None, []
    for t in range(0, BLOCK, 8):                                        # chunks of 8 with a cache
        logits, past = model(seq[:, t:t + 8], past_kvs=past)
        outs.append(logits)
    inc8 = torch.cat(outs, dim=1)
    model.train()

print("cache shape per layer (k):", tuple(past[0][0].shape), " = (batch, heads, positions, head_dim)")
print(f"max |full - cached, 1 token/step| = {(full - inc1).abs().max().item():.2e}")
print(f"max |full - cached, 8 tokens/step| = {(full - inc8).abs().max().item():.2e}")
assert inc1.shape == full.shape and inc8.shape == full.shape
assert (full - inc1).abs().max().item() < 1e-4
assert (full - inc8).abs().max().item() < 1e-4

# %% [markdown]
# Now the generation loop. With the cache, each step feeds only the newly sampled token. When the
# cache is full (`block_size` positions — our learned position embeddings stop there), we drop the
# cache and re-encode the last half window, a cheap way to keep generating.

# %%
@torch.no_grad()
def generate(model, prompt_ids, n_new, temperature=1.0, top_k=None, top_p=None, seed=0,
             use_cache=True, step_times=None):
    model.eval()
    bs = model.block_size
    gen = torch.Generator(device=device).manual_seed(seed)
    out = torch.tensor([prompt_ids], dtype=torch.long, device=device)   # (1, T0)
    past, ctx = None, out[:, -bs:]
    for _ in range(n_new):
        t0 = time.perf_counter()
        if use_cache:
            logits, past = model(ctx, past_kvs=past)
        else:
            logits, _ = model(out[:, -bs:])                              # recompute the whole context
        z = filter_logits(logits[:, -1, :], temperature, top_k, top_p)  # (1, V)
        nxt = torch.multinomial(F.softmax(z, dim=-1), num_samples=1, generator=gen)   # (1, 1)
        out = torch.cat([out, nxt], dim=1)
        if use_cache:
            if past[0][0].size(2) >= bs:                                 # cache full: re-encode half a window
                past, ctx = None, out[:, -(bs // 2):]
            else:
                ctx = nxt
        if step_times is not None:
            step_times.append(time.perf_counter() - t0)
    model.train()
    return out[0].tolist()

prompt_ids = tok_encode("ROMEO:")
for kw in [dict(temperature=0.6), dict(temperature=1.0), dict(temperature=1.0, top_k=10),
           dict(temperature=1.0, top_p=0.9)]:
    print(f"========== {kw} ==========")
    print(tok_decode(generate(model, prompt_ids, 250, seed=3, **kw)))
    print()

# %% [markdown]
# **How much does the cache save?** With `block_size = 64` and a tiny model, Python overhead
# dominates and the gain is modest. The benefit grows with the context length, so we time a
# (randomly initialised — speed does not depend on the weights) model with a 512-token context,
# generating 256 tokens with and without the cache.

# %%
torch.manual_seed(0)
big = GPT(vocab_size, block_size=512, n_layer=4, n_head=4, n_embd=128).to(device)
set_sdpa(big, True)
times = {}
for use_cache in (False, True):
    st = []
    generate(big, prompt_ids[:1], 8, use_cache=use_cache)                    # warm-up
    generate(big, prompt_ids[:1], 256, use_cache=use_cache, step_times=st)
    times[use_cache] = np.array(st)
    print(f"use_cache={use_cache!s:5s}: {times[use_cache].sum():.2f} s for 256 tokens")
print(f"speed-up: {times[False].sum() / times[True].sum():.1f}x")
plt.figure(figsize=(6, 3.5))
plt.plot(1000 * times[False], label="no cache: recompute the context")
plt.plot(1000 * times[True], label="KV cache")
plt.xlabel("generated token (= context length)"); plt.ylabel("ms per token"); plt.legend()
plt.title("Cost per generated token"); plt.show()

# %% [markdown]
# Without a cache the cost per token grows linearly with the context; with the cache it is roughly
# constant (attention over the cache does grow, but it is a small part of the work here). The price
# is **memory**: the cache holds $2 \times L \times T \times d$ numbers per sequence — for a
# Llama-2-70B-sized model and a long context, many gigabytes, which is why **multi-query / grouped-query
# attention** (several query heads sharing one key/value head) and **paged** caches were invented.
#
# ## 5. What do the heads attend to?
#
# We run the model with explicit attention on a line from the validation set and draw the attention
# matrices of a few heads in the first and last layer (rows: query position, columns: key position;
# the upper triangle is masked).

# %%
seg_start = val_text.find("\n", 6000) + 1
seg_ids = tok_encode(val_text[seg_start:seg_start + 400])[:40]
seg = torch.tensor([seg_ids], device=device)
labels = [tok_decode([t]).replace("\n", "\\n").replace(" ", "␣") for t in seg_ids]
set_sdpa(model, False)
with torch.no_grad():
    model.eval()
    model(seg)
    model.train()
show_heads = [(0, h) for h in range(N_HEAD)] + [(N_LAYER - 1, h) for h in range(N_HEAD)]
fig, axes = plt.subplots(2, N_HEAD, figsize=(4 * N_HEAD, 8.5))
for ax, (l, h) in zip(axes.flat, show_heads):
    A = model.blocks[l].attn.att[0, h].cpu().numpy()                     # (T, T)
    ax.imshow(A, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=6)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=6)
    ax.set_title(f"layer {l}, head {h}"); ax.grid(False)
plt.tight_layout(); plt.show()

# %% [markdown]
# A compact summary over many windows: the **mean attention distance** of each head,
# $\sum_j a_{tj}(t - j)$ averaged over queries $t$. Heads with distance ≈ 0–2 are local ("previous
# character", "current word"); heads with large distances gather context from further back
# (e.g. the speaker's name or the start of the line).

# %%
with torch.no_grad():
    model.eval()
    xs, _ = get_batch(val_tok, 64, BLOCK, torch.Generator().manual_seed(1))
    model(xs)
    model.train()
dist = torch.arange(BLOCK)[:, None] - torch.arange(BLOCK)[None, :]          # t - j
dist = dist.clamp(min=0).float().to(device)
mean_dist = np.array([[(blk.attn.att * dist).sum(-1).mean(dim=(0, 2))[h].item() for h in range(N_HEAD)]
                      for blk in model.blocks])                               # (L, h)
plt.figure(figsize=(6, 3.5))
for h in range(N_HEAD):
    plt.bar(np.arange(N_LAYER) + 0.2 * (h - 1.5), mean_dist[:, h], width=0.2, label=f"head {h}")
plt.xticks(range(N_LAYER), [f"layer {l}" for l in range(N_LAYER)])
plt.ylabel("mean attention distance (tokens)"); plt.legend(fontsize=8); plt.title("Local and long-range heads")
plt.show()
set_sdpa(model, True)

# %% [markdown]
# ## 6. A glimpse of scaling
#
# Kaplan et al. (2020) and Hoffmann et al. (2022, "Chinchilla") found that the loss of a language
# model falls as a smooth **power law** in model size, data and compute. We cannot afford a real
# study here, but three very short runs (300 steps, 2 layers, widths 16 / 32 / 64) already show the
# trend: bigger models learn more from the *same* data and number of steps.

# %%
RUN_SCALING = True
if RUN_SCALING:
    sizes, losses = [], []
    for n_embd, n_head in [(16, 2), (32, 4), (64, 4)]:
        torch.manual_seed(0)
        m = GPT(vocab_size, BLOCK, n_layer=2, n_head=n_head, n_embd=n_embd)
        h = train_gpt(m, 300, warmup=30, eval_every=300, verbose=False)
        n_nonemb = sum(p.numel() for p in m.blocks.parameters())
        sizes.append(n_nonemb); losses.append(h["val"][-1][1])
        print(f"n_embd {n_embd:3d}: {n_nonemb:7,d} non-embedding parameters -> val {losses[-1]:.3f} nats/char"
              f" ({h['seconds']:.0f} s)")
    plt.figure(figsize=(5, 3.5))
    plt.loglog(sizes, losses, "o-")
    plt.xlabel("non-embedding parameters"); plt.ylabel("val NLL (nats / char)")
    plt.title("Loss vs model size (300 steps)"); plt.show()

# %% [markdown]
# ## 7. Result

# %%
assert val_nll_gpt < 2.5, val_nll_gpt
print(f"tokenizer: {TOKENIZER}, {n_params:,} parameters")
print(f"val NLL (nats/char): {val_nll_gpt:.3f}")

# %% [markdown]
# ## From here to LLMs
#
# * **The architecture is GPT-2.** Our `GPT` class differs from GPT-2 small only in size
#   ($L = 12$, $d = 768$, 12 heads, 1024 positions, 50 257 BPE tokens: 124 M parameters). Modern models
#   (Llama, Mistral, Qwen) change a few components — RMSNorm instead of LayerNorm, rotary position
#   embeddings (RoPE) instead of learned ones, SwiGLU instead of GELU MLPs, grouped-query attention —
#   but the skeleton is exactly what you wrote.
# * **The tokeniser is byte-level BPE.** GPT-2's tokeniser is the algorithm of Section 1 with a
#   better regex, 50 k merges and a trick to map bytes to printable characters; `tiktoken` and
#   SentencePiece implement the same idea efficiently. Tokenisation explains many LLM quirks: poor
#   letter counting and arithmetic, higher cost for non-English text (fewer characters per token),
#   and "glitch tokens" that were merged but almost never trained.
# * **The training recipe is the same**: AdamW, warm-up, cosine decay, clipping, mixed precision,
#   at a scale of trillions of tokens; scaling laws are used to choose model size for a compute budget.
# * **Inference is KV-cached sampling** with temperature / top-$p$; serving systems add paged KV
#   memory, batching of many users, speculative decoding and quantisation.
# * What is missing for a chat assistant: pretraining at scale, then **instruction tuning** and
#   **preference optimisation** (RLHF / DPO) — topics for the LLM course.
#
# ## Try it yourself
#
# 1. Set `TOKENIZER = "bpe"` and retrain. Compare the validation NLL **per character** with the char
#    model. How much more text fits into the 64-token context?
# 2. Change `N_MERGES` (e.g. 100, 1000, 3000) and plot chars/token and the GPT's nats/char against the
#    vocabulary size. Where is the sweet spot for this corpus and budget?
# 3. Remove weight tying, or the warm-up, or the $1/\sqrt{2L}$ initialisation scaling, and compare
#    the training curves.
# 4. Replace the learned position embedding by sinusoidal or rotary (RoPE) embeddings. Can the model
#    now be evaluated on windows longer than it was trained on?
# 5. Implement **greedy** and **beam search** decoding and compare their text with nucleus sampling.
#    Why does maximising likelihood produce repetitive text?
# 6. Measure the KV-cache memory of `big` for 512 tokens, and implement grouped-query attention
#    (2 key/value heads shared by 4 query heads). How much memory does it save?
#
# **Further reading.**
# Vaswani et al. (2017), *Attention is all you need*;
# Radford et al. (2018), *Improving language understanding by generative pre-training* (GPT-1);
# Radford et al. (2019), *Language models are unsupervised multitask learners* (GPT-2);
# Sennrich, Haddow & Birch (2016), *Neural machine translation of rare words with subword units* (BPE);
# Gage (1994), *A new algorithm for data compression*;
# Press & Wolf (2017), *Using the output embedding to improve language models* (weight tying);
# Loshchilov & Hutter (2019), *Decoupled weight decay regularization* (AdamW);
# Holtzman et al. (2020), *The curious case of neural text degeneration* (nucleus sampling);
# Kaplan et al. (2020), *Scaling laws for neural language models*;
# Hoffmann et al. (2022), *Training compute-optimal large language models*;
# Pope et al. (2023), *Efficiently scaling Transformer inference* (KV cache);
# Karpathy, *nanoGPT*, *minbpe* and the "Let's build GPT" / "Let's build the GPT tokenizer" lectures.
