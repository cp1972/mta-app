# MTA — Multi-Text Analyser

A Python toolkit for **topic-modelling analysis** of text corpora,
following the KISS principle. NMF, LDA, cross-validation, word and
document embeddings, semantic context, group comparison — all running
locally on your machine, with results saved as CSV/JSON tables and
PDF/PNG plots.

> **Looking for the user guide?** Go to
> **[`MTA-for-Master/README.md`](MTA-for-Master/README.md)** — that's
> where you learn how to install and run MTA in three steps.

## Repository layout

This repository contains **two versions** of MTA, kept side by side so
that older work remains reproducible.

| Folder              | What it contains                                      | For whom                                 |
| ------------------- | ----------------------------------------------------- | ---------------------------------------- |
| `MTA-for-Master/`   | **Version 3.0** — Streamlit web app + modern CLI      | Everyone, current users                  |
| `archive/`          | **Versions ≤ 2.0** — original single-script `MTA.py`  | Users with legacy workflows or do-files  |
| `changes.md`        | Full version history (3.0 down to 0.1, January 2017)  | Anyone tracking what changed             |
| `LICENSE`           | License                                               | —                                        |

If you arrived here for the first time, **use `MTA-for-Master/`**. The
`archive/` folder is kept publicly available only so that students,
collaborators and reviewers who rely on the previous `MTA.py` script
can still find it, its documentation, and reproduce earlier analyses.

## What MTA does

  - **Topic-model analysis** with NMF and LDA algorithms; word and
    document embeddings (co-occurrence + PCA by default, Word2Vec
    optional).
  - **Cross-validation metrics** (Elbow, Silhouette, Calinski-Harabasz,
    Davies-Bouldin, plus Cophenet for NMF and LDA) to help you pick
    the best number of topics for a given corpus.
  - **Similarity analysis** of documents, words, and semantic clusters
    based on the embeddings.
  - **Topic evolution** across documents, with optional yearly
    aggregation when filenames carry a year stamp.
  - **Group comparison** (3.0+): Welch's t-test with Benjamini-Hochberg
    correction, plus automatic box-plots for topics where groups differ
    significantly.
  - **Visualisation** with high-quality plots (PDF + PNG, multilingual:
    English / French / German) and **results saved as tables** (CSV +
    JSON) for further use in Stata, R, Gephi, or any spreadsheet.

MTA handles almost every kind of text source, from short word lists to
full books or large collections of files. It analyses corpora in most
common languages and can be adapted for less common ones — over the
years it has been used by PhD scholars on Albanian, Japanese, Polish
and Farsi texts.

## Which version should I use?

  - **You are a student or a new user → version 3.0**
    (`MTA-for-Master/`). It installs in a few clicks, no Anaconda
    needed, and the Streamlit interface walks you through the workflow
    page by page. Open
    [`MTA-for-Master/README.md`](MTA-for-Master/README.md).

  - **You have an existing Stata do-file or shell pipeline that pipes
    inputs into `MTA.py` → use the archive**
    (`archive/`). The old script and its three reference documents
    (`install.md`, `doc.md`, `automate.md`) are kept there unchanged.
    See [`archive/README.md`](archive/README.md). The new `MTA_v3.py`
    in version 3.0 offers an argument-based batch mode that is the
    recommended replacement when you have time to port your scripts.

  - **You want to know what changed between versions →** read
    [`changes.md`](changes.md). The 3.0 entry at the top summarises
    the reorganization; older entries (2.0, 1.9, …, 0.1) are kept
    intact below it.

## A note on the new version (3.0)

Versions 0.1 through 2.0 of MTA were a single terminal script
(`MTA.py`) that prompted the user step by step. Version 3.0 keeps that
analytical core but separates it from the interface:

  - **`mta_core.py`** — pure-Python engine: same algorithms, no I/O,
    no `print`, no `input`. Can be imported and reused.
  - **Streamlit app** — visual front-end for class teaching and
    exploratory work.
  - **`MTA_v3.py`** — modern CLI with two modes: an interactive menu
    (in the spirit of the original `MTA.py`) and a non-interactive
    batch mode taking `--corpus`, `--stopwords`, `--action`, etc., for
    automated pipelines.
  - **Double-clickable launchers** (`.bat`, `.command`, `.sh`) and a
    self-contained installer based on [`uv`](https://github.com/astral-sh/uv)
    — no Anaconda, no global Python install, no system pollution.

See [`changes.md`](changes.md) for the full picture.

## If you want to quote this software in your publication

Papilloud, C., 2017-2026, MTA: Multi-Text Analyser,
<http://soziologie.uni-halle.de/professuren/theorie/>

BibTeX entry for the software:

```bibtex
@misc{Papilloud1726,
  author       = {Papilloud, C.},
  title        = {MTA: Multi-Text Analyser},
  howpublished = {\url{http://soziologie.uni-halle.de/professuren/theorie/}},
  year         = {2017--2026}
}
```

BibTeX entry for the related book on the method used:

```bibtex
@book{PapilloudHinneburg2018,
  address   = {Wiesbaden},
  author    = {Papilloud, C. and Hinneburg, A.},
  publisher = {Springer},
  title     = {Einführung in die qualitative Analyse von Texten mit Topic-Modellen},
  year      = {2018}
}
```

## Contact

MTA is a research project at the Institute of Sociology, Martin Luther
University Halle-Wittenberg. It has given life to other text-analysis
tools, from simple shell scripts to multi-platform R and Python
applications. Interested in such tools or in our research? Feel free
to get in touch.
