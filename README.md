# Neural Networks — course notebooks

Jupyter notebooks for a master's-level course on neural networks. The course starts from
linear models and hand-written backpropagation in NumPy, then moves to PyTorch for
convolutional, recurrent, attention-based and generative models.

Every lecture has three notebooks:

| Folder | Contents |
|---|---|
| [`lectures/`](lectures) | Lecture notebook: theory, derivations and runnable demos |
| [`exercises/`](exercises) | Exercise notebook for students, with `# YOUR CODE HERE` gaps and checks |
| [`solutions/`](solutions) | The same exercises with full solutions (instructor version) |

## Schedule

| # | Topic | Framework | Lecture | Exercises |
|---|---|---|---|---|
| 1 | Linear models and the perceptron | NumPy | [notebook](lectures/01_linear_models.ipynb) | [notebook](exercises/01_linear_models_exercises.ipynb) |
| 2 | Multilayer perceptrons and backpropagation | NumPy | [notebook](lectures/02_mlp_backprop.ipynb) | [notebook](exercises/02_mlp_backprop_exercises.ipynb) |
| 3 | Optimisation and initialisation | NumPy | [notebook](lectures/03_optimization.ipynb) | [notebook](exercises/03_optimization_exercises.ipynb) |
| 4 | PyTorch fundamentals | PyTorch | [notebook](lectures/04_pytorch_fundamentals.ipynb) | [notebook](exercises/04_pytorch_fundamentals_exercises.ipynb) |
| 5 | Generalisation and regularisation | PyTorch | [notebook](lectures/05_regularization.ipynb) | [notebook](exercises/05_regularization_exercises.ipynb) |
| 6 | Convolutional neural networks | PyTorch | [notebook](lectures/06_cnns.ipynb) | [notebook](exercises/06_cnns_exercises.ipynb) |
| 7 | Recurrent neural networks | PyTorch | [notebook](lectures/07_rnns.ipynb) | [notebook](exercises/07_rnns_exercises.ipynb) |
| 8 | Attention and Transformers | PyTorch | [notebook](lectures/08_transformers.ipynb) | [notebook](exercises/08_transformers_exercises.ipynb) |
| 9 | Autoencoders and variational autoencoders | PyTorch | [notebook](lectures/09_autoencoders_vae.ipynb) | [notebook](exercises/09_autoencoders_vae_exercises.ipynb) |
| 10 | Denoising diffusion models | PyTorch | [notebook](lectures/10_diffusion.ipynb) | [notebook](exercises/10_diffusion_exercises.ipynb) |

## Language-model track

A companion notebook for every lecture applies that lecture's ideas to **language
modelling**, step by step from a bigram table to a GPT and a diffusion language model.
It is meant as a foundation for a follow-up course on large language models. Everything is
character- or word-level and trains on a laptop CPU.

| # | Notebook | Idea | Data |
|---|---|---|---|
| 1 | [Bigram](language_models/01_bigram.ipynb) | counting vs softmax regression; NLL and perplexity | names |
| 2 | [MLP LM](language_models/02_mlp_lm.ipynb) | Bengio et al. (2003): embeddings + MLP, manual backprop in NumPy | names |
| 3 | [Training dynamics](language_models/03_training_lm.ipynb) | initial loss, tanh saturation, Adam, LR schedules, update/data ratio | names |
| 4 | [PyTorch LM](language_models/04_pytorch_lm.ipynb) | the MLP LM in PyTorch: `nn.Embedding`, `cross_entropy`, sampling | names |
| 5 | [Word-level LM](language_models/05_word_lm.ipynb) | word tokenisation, Zipf, overfitting, dropout, weight tying | Shakespeare |
| 6 | [WaveNet](language_models/06_wavenet_lm.ipynb) | causal dilated convolutions, predicting every position in parallel | names |
| 7 | [Char-RNN](language_models/07_char_rnn.ipynb) | LSTM, truncated BPTT with carried state, sampling | Shakespeare |
| 8 | [GPT](language_models/08_gpt.ipynb) | BPE tokeniser + decoder-only Transformer, top-p, KV cache | Shakespeare |
| 9 | [Sequence VAE](language_models/09_names_vae.ipynb) | latent-variable LM, posterior collapse, interpolation | names |
| 10 | [Diffusion LM](language_models/10_diffusion_lm.ipynb) | masked discrete diffusion, parallel decoding, infilling | names |

Data lives in [`data/`](data): `names.txt` (US Social Security first names) and
`tinyshakespeare.txt` (public domain). The notebooks read it from the repository, or download it
from GitHub when opened in Colab.

Each notebook has an **Open in Colab** badge at the top. All datasets ship with
scikit-learn or are generated in the notebook, so nothing is downloaded and every notebook
runs on a laptop CPU in a few minutes.

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

## For instructors: editing the notebooks

The notebooks are generated — **edit the sources in [`src/`](src), not the `.ipynb` files.**
Each lecture folder `src/NN_topic/` contains `lecture.py` and `exercises.py` in the
"percent" cell format (`# %%` for code, `# %% [markdown]` for text), which VS Code and
PyCharm open as notebooks.

In `exercises.py`, wrap solution code in `### BEGIN SOLUTION` / `### END SOLUTION` and
written answers in `<!-- BEGIN SOLUTION -->` / `<!-- END SOLUTION -->`. The build turns
them into gaps in the student notebook and keeps them in the solution notebook.

```bash
python build.py                  # regenerate lectures/, exercises/, solutions/
python build.py --run 06         # quick check: run lecture 06 + solutions as a script
python build.py --check          # fail if any notebook is out of date
```

GitHub Actions executes every lecture and solution notebook on each push
([`.github/workflows/notebooks.yml`](.github/workflows/notebooks.yml)), so a red build means
something in the course no longer runs.

> **Tip:** you may not want to publish `solutions/` to students. Keep this repository
> private for instructors and publish a copy without `solutions/` and `src/`, or remove
> those folders from the student-facing fork.
