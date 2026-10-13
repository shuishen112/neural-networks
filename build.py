#!/usr/bin/env python3
"""Build the course notebooks from the plain-text sources in ``src/``.

Every lecture lives in its own folder ``src/NN_topic/`` with two files:

* ``lecture.py``   -> ``lectures/NN_topic.ipynb``
* ``exercises.py`` -> ``exercises/NN_topic_exercises.ipynb``  (student version)
                   -> ``solutions/NN_topic_solutions.ipynb``  (instructor version)

The companion language-model track lives in ``src/lm/NN_name.py`` and is built
to ``language_models/NN_name.ipynb`` (one notebook per lecture, no exercises).

Sources use the "percent" cell format (the same one VS Code, Spyder and
jupytext understand)::

    # %% [markdown]
    # # A heading
    # Some *markdown* text.

    # %%
    x = 1 + 1

Solution code inside exercises is wrapped in marker lines::

    ### BEGIN SOLUTION
    w = w - lr * grad
    ### END SOLUTION

In the student notebook the block is replaced by ``# YOUR CODE HERE`` and
``raise NotImplementedError()`` (keeping the indentation). Written answers in
markdown cells use ``<!-- BEGIN SOLUTION -->`` / ``<!-- END SOLUTION -->``.

Usage::

    python build.py                 # (re)build all notebooks
    python build.py --run 03        # run a source as a plain script (quick check)
    python build.py --lm 03         # run the language-model notebook source for lecture 03
    python build.py --check         # fail if notebooks are out of date
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
REPO = "shuishen112/neural-networks"
BRANCH = "main"

CELL_RE = re.compile(r"^# %%(?P<rest>.*)$")
SOL_BEGIN, SOL_END = "### BEGIN SOLUTION", "### END SOLUTION"
MD_BEGIN, MD_END = "<!-- BEGIN SOLUTION -->", "<!-- END SOLUTION -->"


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------
def parse_cells(text: str) -> list[tuple[str, str]]:
    """Split a percent-format source into ``(cell_type, source)`` pairs."""
    cells: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        m = CELL_RE.match(line)
        if m:
            kind = "markdown" if "[markdown]" in m.group("rest") else "code"
            cells.append((kind, []))
            continue
        if not cells:  # header comments before the first cell are ignored
            continue
        cells[-1][1].append(line)

    out = []
    for kind, lines in cells:
        if kind == "markdown":
            lines = [_strip_md(l) for l in lines]
        body = "\n".join(lines).strip("\n")
        if body.strip():
            out.append((kind, body))
    return out


def _strip_md(line: str) -> str:
    if line.startswith("# "):
        return line[2:]
    if line.startswith("#"):
        return line[1:]
    return line


# --------------------------------------------------------------------------
# Solution stripping
# --------------------------------------------------------------------------
def strip_code_solution(src: str) -> str:
    out, inside = [], False
    for line in src.splitlines():
        stripped = line.strip()
        if stripped == SOL_BEGIN:
            indent = line[: len(line) - len(line.lstrip())]
            out.append(f"{indent}# YOUR CODE HERE")
            out.append(f"{indent}raise NotImplementedError()")
            inside = True
        elif stripped == SOL_END:
            inside = False
        elif not inside:
            out.append(line)
    if inside:
        raise ValueError("unterminated solution block:\n" + src)
    return "\n".join(out)


def keep_code_solution(src: str) -> str:
    return "\n".join(
        l for l in src.splitlines() if l.strip() not in (SOL_BEGIN, SOL_END)
    )


def strip_md_solution(src: str) -> str:
    pattern = re.compile(re.escape(MD_BEGIN) + r".*?" + re.escape(MD_END), re.S)
    return pattern.sub("*Your answer here.*", src)


def keep_md_solution(src: str) -> str:
    return src.replace(MD_BEGIN, "**Solution.** ").replace(MD_END, "")


# --------------------------------------------------------------------------
# Notebook writing (plain JSON, no nbformat dependency)
# --------------------------------------------------------------------------
def make_notebook(cells: list[tuple[str, str]]) -> dict:
    nb_cells = []
    for i, (kind, src) in enumerate(cells):
        lines = src.splitlines(keepends=True)
        cell = {"cell_type": kind, "id": f"cell-{i:03d}", "metadata": {}, "source": lines}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        nb_cells.append(cell)
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def colab_badge(path: str) -> tuple[str, str]:
    url = f"https://colab.research.google.com/github/{REPO}/blob/{BRANCH}/{path}"
    badge = "https://colab.research.google.com/assets/colab-badge.svg"
    return ("markdown", f"[![Open In Colab]({badge})]({url})")


def write_nb(cells, rel_path: str) -> str:
    nb = make_notebook([colab_badge(rel_path)] + cells)
    text = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
    return text


def lecture_dirs() -> list[Path]:
    return sorted(p for p in SRC.iterdir() if p.is_dir() and re.match(r"\d\d_", p.name))


def targets() -> dict[str, str]:
    """Map output path -> notebook JSON text for every source."""
    out: dict[str, str] = {}
    for d in lecture_dirs():
        name = d.name
        lec = d / "lecture.py"
        if lec.exists():
            rel = f"lectures/{name}.ipynb"
            out[rel] = write_nb(parse_cells(lec.read_text()), rel)
        ex = d / "exercises.py"
        if ex.exists():
            cells = parse_cells(ex.read_text())
            student = [
                (k, strip_code_solution(s) if k == "code" else strip_md_solution(s))
                for k, s in cells
            ]
            solution = [
                (k, keep_code_solution(s) if k == "code" else keep_md_solution(s))
                for k, s in cells
            ]
            rel_s = f"exercises/{name}_exercises.ipynb"
            rel_t = f"solutions/{name}_solutions.ipynb"
            out[rel_s] = write_nb(student, rel_s)
            out[rel_t] = write_nb(solution, rel_t)
    for src in lm_sources():
        rel = f"language_models/{src.stem}.ipynb"
        out[rel] = write_nb(parse_cells(src.read_text()), rel)
    return out


def lm_sources() -> list[Path]:
    d = SRC / "lm"
    return sorted(d.glob("[0-9][0-9]_*.py")) if d.exists() else []


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------
def cmd_build() -> None:
    for rel, text in targets().items():
        path = ROOT / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.read_text() != text:
            path.write_text(text)
            print("wrote", rel)


def cmd_check() -> None:
    stale = [rel for rel, text in targets().items()
             if not (ROOT / rel).exists() or (ROOT / rel).read_text() != text]
    if stale:
        print("Out-of-date notebooks (run `python build.py`):", *stale, sep="\n  ")
        sys.exit(1)
    print("All notebooks up to date.")


def cmd_run(prefix: str, which: str) -> None:
    """Run the code cells of one source as a plain Python script."""
    [d] = [d for d in lecture_dirs() if d.name.startswith(prefix)]
    files = ["lecture.py", "exercises.py"] if which == "both" else [f"{which}.py"]
    for f in files:
        src = d / f
        if not src.exists():
            continue
        code = [s for k, s in parse_cells(src.read_text()) if k == "code"]
        if f == "exercises.py":
            code = [keep_code_solution(c) for c in code]
        script = "import matplotlib\nmatplotlib.use('Agg')\n" + "\n\n".join(code)
        _run_script(script, f"{d.name}/{f}")


def cmd_lm(prefix: str) -> None:
    [src] = [s for s in lm_sources() if s.name.startswith(prefix)]
    code = [s for k, s in parse_cells(src.read_text()) if k == "code"]
    _run_script("import matplotlib\nmatplotlib.use('Agg')\n" + "\n\n".join(code), f"lm/{src.name}")


def _run_script(script: str, label: str) -> None:
    """Run code from inside a notebook folder so relative paths like ../data work."""
    cwd = ROOT / "lectures"
    cwd.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "nb.py"
        p.write_text(script)
        print(f"== running {label}")
        env = dict(os.environ, MPLBACKEND="Agg")
        r = subprocess.run([sys.executable, str(p)], cwd=cwd, env=env)
        if r.returncode:
            sys.exit(r.returncode)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--run", metavar="NN")
    ap.add_argument("--lm", metavar="NN")
    ap.add_argument("--which", default="both", choices=["both", "lecture", "exercises"])
    a = ap.parse_args()
    if a.check:
        cmd_check()
    elif a.lm:
        cmd_lm(a.lm)
    elif a.run:
        cmd_run(a.run, a.which)
    else:
        cmd_build()


if __name__ == "__main__":
    main()
