# Language-model track — conventions

One notebook per lecture. Each notebook applies that lecture's ideas to **language
modelling**, so that by Lecture 10 students have built every ingredient of a modern LLM
themselves. Sources here are built by `python build.py` into `language_models/`.

| File | Companion to | Model | Data | Framework |
|---|---|---|---|---|
| `01_bigram.py` | L1 linear models | bigram by counting, then as softmax regression | names | NumPy |
| `02_mlp_lm.py` | L2 MLP & backprop | Bengio et al. (2003) MLP with embeddings, manual backprop | names | NumPy |
| `03_training_lm.py` | L3 optimisation & init | training dynamics of the MLP LM | names | NumPy |
| `04_pytorch_lm.py` | L4 PyTorch | the MLP LM in PyTorch | names | PyTorch |
| `05_word_lm.py` | L5 regularisation | word-level LM, overfitting and its cures | Shakespeare | PyTorch |
| `06_wavenet_lm.py` | L6 CNNs | causal dilated convolutions (WaveNet) | names | PyTorch |
| `07_char_rnn.py` | L7 RNNs | character-level LSTM | Shakespeare | PyTorch |
| `08_gpt.py` | L8 Transformers | BPE tokeniser + GPT from scratch | Shakespeare | PyTorch |
| `09_names_vae.py` | L9 VAEs | sequence VAE over names | names | PyTorch |
| `10_diffusion_lm.py` | L10 diffusion | masked (absorbing-state) discrete diffusion LM | names | PyTorch |

## Shared conventions (every notebook)

**Header.** First markdown cell: `# LM N — <title>`, a line
`*Companion to Lecture N — <lecture title>.*`, two or three sentences on what we build and
why it matters for LLMs, and the track roadmap with the current notebook in bold:

> **Language-model track:** 1 Bigram · 2 MLP LM · 3 Training dynamics · 4 PyTorch · 5 Word-level & regularisation · 6 WaveNet · 7 Char-RNN · 8 GPT · 9 Sequence VAE · 10 Diffusion LM

**Data loading.** Use exactly this cell, so notebooks work in the repository, in CI and on Colab:

```python
from pathlib import Path
import urllib.request

def load_text(name):
    """Load a corpus from ../data (repository checkout) or download it (e.g. on Colab)."""
    for folder in (Path("data"), Path("../data")):
        if (folder / name).exists():
            return (folder / name).read_text(encoding="utf-8")
    url = f"https://raw.githubusercontent.com/shuishen112/neural-networks/main/data/{name}"
    return urllib.request.urlopen(url).read().decode("utf-8")
```

* `names.txt` — 32 033 lower-case first names, one per line (US Social Security Administration data).
* `tinyshakespeare.txt` — 1.1 M characters of Shakespeare (public domain).

**Splits.** Names: shuffle with `random.Random(42).shuffle(words)` (Python's `random`, so
the split is identical in NumPy and PyTorch notebooks), then 80 % / 10 % / 10 %
train / val / test. Shakespeare: first 90 % of characters train, last 10 % val.

**Character vocabulary for names:** `"."` (index 0) is the start/end token, then `a`–`z`
(indices 1–26), `V = 27`.

**Metrics.** Always report the **validation negative log-likelihood in nats per token**
(the mean cross-entropy) and, where natural, perplexity $e^{\text{NLL}}$. For names,
print a final line `val NLL (nats/char): x.xxx` so numbers can be compared across the
track (uniform guessing is $\ln 27 \approx 3.30$; bigram ≈ 2.45; a good MLP ≈ 2.1–2.2).

**Samples.** Every notebook generates text from its model at the end of training.

**Closing sections.** `## From here to LLMs` (how this notebook's idea shows up in modern
large language models, concretely) and `## Try it yourself` (3–6 short, open-ended
suggestions for students). Then `**Further reading.**`.

**Constraints.** Self-contained (only numpy, matplotlib, scikit-learn, torch); no IPython
magics; fixed seeds; each notebook runs in ≤ 3 minutes on a 2-core CPU. Use
`device = "cuda" if torch.cuda.is_available() else "cpu"` in PyTorch notebooks.

Quick local run of one source: `python build.py --lm 05`.
