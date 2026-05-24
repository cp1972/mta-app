# MTA — Student Guide

Welcome. This folder contains everything you need to do **topic
modelling** on your own computer, without depending on any online
service. Once installed, MTA works offline.

## In 3 steps

### 1. First time: INSTALL MTA (only once)

Double-click on the file that matches your operating system:

| Operating system | File to double-click       |
| ---------------- | -------------------------- |
| **Windows**      | `install_first_run.bat`    |
| **Mac**          | `install_first_run.command` |
| **Linux**        | `install_first_run.sh`     |

A black window opens. That's normal. It tells you what it is going to
do and asks you to press Enter. The installation takes **5 to 10
minutes**. You can go grab a coffee.

> **For Mac users:** the first time, macOS may refuse to open the file,
> saying it comes from an unidentified developer. If that happens:
> **right-click** the file → **Open** → confirm **Open**. You only need
> to do this once.

### 2. Every time you use it: START MTA

Double-click on the file that matches your operating system:

| Operating system | File to double-click |
| ---------------- | -------------------- |
| **Windows**      | `start_MTA.bat`      |
| **Mac**          | `start_MTA.command`  |
| **Linux**        | `start_MTA.sh`       |

A black window opens and shows a menu offering you a choice between
**two interfaces**:

- **[1] Streamlit web app** *(default)* — the visual interface in
  your browser, recommended for first-time users. Just press Enter
  to launch it.
- **[2] Command-line interactive menu** — a text-based menu running
  inside the same terminal window, in the spirit of the original
  MTA.py script. Useful if you prefer keyboard-only interaction or
  if you are working over SSH.
- **[3] Show batch / scripting usage** — prints reference help for
  running MTA non-interactively from Stata, R, or shell scripts.

In all cases, **DO NOT CLOSE the black window** while you are using MTA.

### 3. To stop MTA

Simply close the black window. You can also close the browser tab, but
the black window is what holds the engine.

## Advanced: batch / scripting mode

For non-interactive use (e.g. from Stata, R, or shell scripts), the
command-line script `code/MTA_v3.py` accepts arguments. From the MTA
folder, run for example:

```bash
# On Mac/Linux:
./.venv/bin/python code/MTA_v3.py --corpus PATH --stopwords PATH \
    --action nmf --n-topics 5 --json

# On Windows:
.venv\Scripts\python.exe code\MTA_v3.py --corpus PATH --stopwords PATH ^
    --action nmf --n-topics 5 --json
```

Outputs (CSV, JSON, PDF, PNG) are written to a timestamped folder. Pick
option **[3]** from the launcher menu to see the full list of arguments.

## Your first analyses

The `examples/` folder contains:

- **`demo_corpus_en/`** — six short English texts to try MTA out
  (sociology, economics, computational methods)
- **`demo_corpus_fr/`** — the same six texts in French
- **`stopwords_en.txt`** — a list of English stopwords
- **`stopwords_fr.txt`** — a list of French stopwords

To run your first analysis after starting MTA:

1. In step 1 of the interface, drop the six files from
   `demo_corpus_en/` (or `demo_corpus_fr/`) into the "Your texts" area
2. Drop the matching stopwords file (`stopwords_en.txt` or
   `stopwords_fr.txt`) into the "Your stopwords" area
3. In step 2, click "Build the matrices"
4. In step 4, request **3 topics** for NMF and click "Run NMF"
5. You should see three distinct themes corresponding to the three
   subjects of the demonstration texts (sociology, economics,
   computational methods)
6. Visit the **🕸 Network views** page to see the same model as a
   bipartite graph: topics in color, documents and top-words around them

## The Network views page

After running NMF or LDA, the **🕸 Network views** page renders the
topic model as three publication-ready bipartite graphs:

- **Topic ↔ Document** — each document linked to the topic(s) it
  weighs strongly on; helps you see which documents migrate between
  topics
- **Topic ↔ Top-N words** — each topic linked to its most
  representative words; helps you read the *content* of each topic
- **Combined** — topics, documents (circles) and words (squares) on
  one canvas, for an overview of the model

The graphs use the same ForceAtlas2 layout as Gephi, with the
Solarized color palette. Node sizes encode the cumulated weight of
each topic; edge thicknesses encode the strength of each link. Use
the **Emphasize size differences** toggle when working on a balanced
corpus where masses are similar — it stretches the smallest and
largest topics apart so even modest differences become visible.

Each graph can be downloaded as PNG (for slides and Word documents)
or PDF (vector, for publication). The CLI batch mode produces the
same three figures via `--action network`.

## Working with your own data

Your texts must be `.txt` files (plain text) encoded in UTF-8. If you
have Word, PDF or other files, convert them to `.txt` first
(LibreOffice or Word both offer "Save as → Text").

Good practice: create one folder per analysis project, with a subfolder
for the texts and a stopwords file matching your language and corpus.

## Troubleshooting

- **The installer stops with a download error**: check your internet
  connection and re-run the installer. It will resume where it stopped.

- **MTA refuses to start** ("MTA is not installed"):
  you skipped the install step. Run the installer first.

- **The browser does not open automatically**:
  open it manually and type `http://localhost:8501` in the address bar.

- **The app feels slow**: your corpus is probably very large. Reduce
  the maximum number of topics tested in step 3, or work on a sample.

- **You see an "Address already in use" message**: another MTA
  instance is already running. Close other MTA windows and try again.

## Uninstalling MTA

To remove everything: drag this entire folder to the trash. MTA leaves
nothing else on your computer.

## To go further

Full MTA documentation and related publications:
<https://github.com/cp1972/mta-app>

C. Papilloud & A. Hinneburg, *Einführung in die qualitative Analyse von
Texten mit Topic-Modellen*, Springer, 2018.
