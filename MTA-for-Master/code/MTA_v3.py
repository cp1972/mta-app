#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MTA_v3.py
=========

Command-line interface for MTA (Multi-Text Analyser).

This is the modern CLI counterpart to the Streamlit app, sharing the same
mta_core.py business logic. Two modes:

  • INTERACTIVE: run without arguments → menu-driven (like original MTA.py)
  • BATCH:       run with arguments    → silent execution for scripting
                                          (Stata, R, shell pipelines)

Usage examples
--------------

Interactive (in class with students):

    python MTA_v3.py

Batch — NMF with 5 topics, German labels, both PDF and PNG plots:

    python MTA_v3.py --corpus /data/articles --stopwords /data/stop_de.txt \\
                     --action nmf --n-topics 5 --language de

Batch — word-weight analysis for a few keywords:

    python MTA_v3.py --corpus /data/articles --stopwords /data/stop_de.txt \\
                     --action word-weights --words "Impfung,Maskenpflicht"

Batch — full pipeline:

    python MTA_v3.py --corpus /data/articles --stopwords /data/stop_de.txt \\
                     --action all --n-topics 7 --output /results/covid
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for headless batch
import matplotlib.pyplot as plt

import mta_core as mta
import mta_network as mtanet


# =============================================================================
# PROGRESS BAR (minimalist, in the spirit of original MTA.py)
# =============================================================================

def progress_bar(iteration: int, total: int, prefix: str = "",
                 suffix: str = "", length: int = 30, fill: str = "█") -> None:
    """ASCII progress bar — re-implements the original MTA.py helper."""
    if total <= 0:
        return
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled = int(length * iteration // total)
    bar = fill * filled + "-" * (length - filled)
    sys.stdout.write(f"\r{prefix} |{bar}| {percent}% {suffix}")
    sys.stdout.flush()
    if iteration >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()


# =============================================================================
# EXPORT HELPERS — write a DataFrame as CSV + JSON, save a figure as PDF + PNG
# =============================================================================

def save_dataframe(df: pd.DataFrame, name: str, output_dir: Path,
                   formats: set[str]) -> None:
    """
    Write `df` to disk. `formats` is a subset of {'csv', 'json'}.
    JSON is written in 'split' orientation: easiest to read from Stata/R/etc.
    """
    if df is None or df.empty:
        return
    if "csv" in formats:
        df.to_csv(output_dir / f"{name}.csv", encoding="utf-8")
    if "json" in formats:
        df.to_json(output_dir / f"{name}.json", orient="split",
                   force_ascii=False, indent=2)


def save_figure(fig: plt.Figure, name: str, output_dir: Path,
                formats: set[str]) -> None:
    """Save figure to PDF and/or PNG. `formats` is a subset of {'pdf', 'png'}."""
    if fig is None:
        return
    if "pdf" in formats:
        fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    if "png" in formats:
        fig.savefig(output_dir / f"{name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def prepare_output_dir(user_path: Optional[str]) -> Path:
    """Create (and return) the output directory. Default = MTA-Results_<ts>."""
    if user_path:
        out = Path(user_path)
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path(f"MTA-Results_{ts}")
    out.mkdir(parents=True, exist_ok=True)
    return out


# =============================================================================
# CORPUS LOADING — bridges the filesystem to mta_core's pure functions
# =============================================================================

def load_corpus(corpus_path: str, verbose: bool = True
                ) -> tuple[list[str], list[str]]:
    """
    Read every .txt file in `corpus_path`, sorted alphabetically.
    Returns (raw_texts, doc_labels).
    """
    paths = sorted(glob.glob(os.path.join(corpus_path, "*.txt")))
    if not paths:
        raise FileNotFoundError(
            f"No .txt files found in {corpus_path!r}. "
            f"Make sure the path is correct and the corpus is in plain text."
        )
    raw_texts = []
    labels = []
    n = len(paths)
    for i, p in enumerate(paths, 1):
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            raw_texts.append(f.read())
        labels.append(os.path.basename(p))
        if verbose:
            progress_bar(i, n, prefix="Loading corpus",
                         suffix=f"({i}/{n})", length=25)
    return raw_texts, labels


def load_stopwords(stopwords_path: str) -> list[str]:
    """Read a stopwords file (one word per line)."""
    with open(stopwords_path, "r", encoding="utf-8", errors="replace") as f:
        return [line.strip() for line in f if line.strip()]


# =============================================================================
# BATCH ACTION DISPATCHER
# =============================================================================

def run_batch(args: argparse.Namespace) -> int:
    """
    Execute the requested action in batch mode. Returns an exit code.
    Each action is delegated to mta_core.* for the science, and writes
    its output via save_dataframe / save_figure.
    """
    # Prepare output formats
    csv_json_formats = set()
    if args.format in ("csv", "both"):
        csv_json_formats.add("csv")
    if args.json:
        csv_json_formats.add("json")
    plot_formats = set()
    if args.format in ("plots", "both"):
        if args.plot_format in ("pdf", "both"):
            plot_formats.add("pdf")
        if args.plot_format in ("png", "both"):
            plot_formats.add("png")

    output_dir = prepare_output_dir(args.output)
    print(f"📁 Output directory: {output_dir.resolve()}")

    # Load corpus + stopwords
    print(f"\nReading corpus from {args.corpus}…")
    raw_texts, labels = load_corpus(args.corpus, verbose=True)
    stopwords = load_stopwords(args.stopwords)
    print(f"  ✓ {len(raw_texts):,} documents, {len(stopwords):,} stopwords")

    # Preprocess + matrices
    print("\nCleaning corpus and building TF-IDF / Count matrices…")
    corpus_wo, corpus_re = mta.preprocess_corpus(
        raw_texts, stopwords, min_word_length=args.min_word_length
    )
    matrices = mta.build_matrices(
        corpus_wo, stopwords,
        min_df=args.min_df, max_df=args.max_df,
    )
    print(f"  ✓ Vocabulary: {len(matrices['tf_names']):,} words")

    # Save the corpus-level artefacts
    save_dataframe(matrices["df_tfidf"], "tfidf_matrix",
                   output_dir, csv_json_formats)

    # Dispatch to the requested action.
    # Note: 'axis-projection' is NOT included in 'all' because it
    # requires user-defined --axis-x (and optionally --axis-y, --axis-z)
    # — there's no sensible default. It must be invoked explicitly.
    actions = (["nmf", "lda", "evolution", "word-weights", "semantic",
                "compare-groups", "network"]
               if args.action == "all" else [args.action])

    for action in actions:
        print(f"\n--- Running action: {action} ---")
        if action == "nmf":
            _action_nmf(args, matrices, corpus_re, labels,
                        output_dir, csv_json_formats, plot_formats)
        elif action == "lda":
            _action_lda(args, matrices, labels,
                        output_dir, csv_json_formats, plot_formats)
        elif action == "evolution":
            _action_evolution(args, matrices, labels,
                              output_dir, csv_json_formats, plot_formats)
        elif action == "word-weights":
            _action_word_weights(args, matrices, labels,
                                 output_dir, csv_json_formats, plot_formats)
        elif action == "semantic":
            _action_semantic(args, matrices, corpus_wo, labels,
                             output_dir, csv_json_formats, plot_formats)
        elif action == "compare-groups":
            _action_compare_groups(args, matrices, labels,
                                   output_dir, csv_json_formats, plot_formats)
        elif action == "network":
            _action_network(args, matrices, labels,
                            output_dir, plot_formats)
        elif action == "axis-analysis":
            _action_axis_analysis(args, matrices, labels,
                                   output_dir, csv_json_formats,
                                   plot_formats)
        elif action == "axis-projection":
            # Deprecated alias kept for scripts that still use the
            # 3.2 name. Runs the unified axis-analysis with a notice.
            print("  ⚠ --action axis-projection is a deprecated alias of "
                  "axis-analysis since MTA 3.4. It now runs the full "
                  "axis-analysis (projection + statistics) in one pass. "
                  "Update your scripts when convenient.",
                  file=sys.stderr)
            _action_axis_analysis(args, matrices, labels,
                                   output_dir, csv_json_formats,
                                   plot_formats)
        elif action == "axis-stats":
            # Deprecated alias kept for scripts that still use the
            # 3.3 name. Same comment as above.
            print("  ⚠ --action axis-stats is a deprecated alias of "
                  "axis-analysis since MTA 3.4. It now runs the full "
                  "axis-analysis (projection + statistics) in one pass. "
                  "Update your scripts when convenient.",
                  file=sys.stderr)
            _action_axis_analysis(args, matrices, labels,
                                   output_dir, csv_json_formats,
                                   plot_formats)
        else:
            print(f"  ✗ Unknown action: {action}", file=sys.stderr)
            return 2

    print(f"\n✓ Done. All outputs are in {output_dir.resolve()}")
    return 0


# =============================================================================
# PER-ACTION HANDLERS (one per menu of original MTA.py)
# =============================================================================

def _plot_metrics_matplotlib(metrics: dict, language: str = "en") -> plt.Figure:
    """Re-render the 6 cross-validation metrics in a 2x3 matplotlib figure."""
    lbl = mta.get_labels(language)
    fig, axs = plt.subplots(2, 3, sharex=True, figsize=(11, 6))

    def _scat(ax, d, title, color):
        ax.scatter(list(d.keys()), list(d.values()),
                   s=18, edgecolor=color, alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel(lbl["number_of_topics"])
        ax.set_ylabel(lbl["score"])

    _scat(axs[0, 0], metrics["elbow"],        "Elbow",             "#b58900")
    _scat(axs[0, 1], metrics["silhouette"],   "Silhouette",        "#cb4b16")
    _scat(axs[0, 2], metrics["cophenet_nmf"], "Cophenet NMF",      "#268bd2")
    _scat(axs[1, 0], metrics["calinski"],     "Calinski-Harabasz", "#dc322f")
    _scat(axs[1, 1], metrics["bouldin"],      "Davies-Bouldin",    "#d33682")
    _scat(axs[1, 2], metrics["cophenet_lda"], "Cophenet LDA",      "#2aa198")
    fig.tight_layout()
    return fig


def _action_nmf(args, matrices, corpus_re, labels,
                output_dir, csv_json_formats, plot_formats):
    """NMF + (optionally) cross-validation suggestions."""
    # Cross-validation if max_topics is requested
    if args.max_topics:
        print(f"  Cross-validation up to k={args.max_topics}…")
        n_total = (args.max_topics - 1) * 3
        cb = lambda i, total, label: progress_bar(
            i, total, prefix=f"    {label}", suffix=f"({i}/{total})", length=20
        )
        metrics = mta.compute_topic_metrics(
            matrices["tf_matrix"], matrices["lda_matrix"], matrices["dense_a"],
            max_topics=args.max_topics, progress_callback=cb,
        )
        # Save metrics table
        ks = metrics["ks"]
        cv_df = pd.DataFrame({
            "Elbow":             list(metrics["elbow"].values()),
            "Silhouette":        list(metrics["silhouette"].values()),
            "Calinski-Harabasz": list(metrics["calinski"].values()),
            "Davies-Bouldin":    list(metrics["bouldin"].values()),
            "Cophenet NMF":      list(metrics["cophenet_nmf"].values()),
            "Cophenet LDA":      list(metrics["cophenet_lda"].values()),
        }, index=ks)
        cv_df.index.name = "Number of topics"
        save_dataframe(cv_df, "cv_metrics", output_dir, csv_json_formats)
        # Save suggestion summary
        sugg_df = pd.DataFrame({k: pd.Series(v) for k, v in
                                metrics["suggestions"].items()})
        save_dataframe(sugg_df, "cv_suggestions",
                       output_dir, csv_json_formats)
        # Plot
        fig = _plot_metrics_matplotlib(metrics, language=args.language)
        save_figure(fig, "cv_metrics", output_dir, plot_formats)

    print(f"  Running NMF with k={args.n_topics}…")
    res = mta.run_nmf(matrices["tf_matrix"], args.n_topics)
    words = mta.top_words_per_topic(res["topicwords"], matrices["tf_names"])
    dist = mta.topic_distribution_per_doc(res["doctopic"], labels)
    dom = mta.dominant_topic_per_doc(dist)
    sentences = mta.best_sentences_per_topic(words, corpus_re)

    save_dataframe(words,     "nmf_top_words",         output_dir, csv_json_formats)
    save_dataframe(dist,      "nmf_distribution",      output_dir, csv_json_formats)
    save_dataframe(dom,       "nmf_dominant_topics",   output_dir, csv_json_formats)
    save_dataframe(sentences, "nmf_best_sentences",    output_dir, csv_json_formats)

    fig = mta.plot_topic_distribution(dist, language=args.language)
    save_figure(fig, "nmf_distribution", output_dir, plot_formats)

    # Single-value summary as JSON for downstream pipelines
    summary = {
        "action": "nmf",
        "n_topics": args.n_topics,
        "cophenet": float(res["cophenet"]),
        "vocabulary_size": int(len(matrices["tf_names"])),
        "n_documents": int(len(labels)),
    }
    (output_dir / "nmf_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  ✓ NMF complete (cophenet={res['cophenet']:.3f})")


def _action_lda(args, matrices, labels,
                output_dir, csv_json_formats, plot_formats):
    """LDA."""
    print(f"  Running LDA with k={args.n_topics}…")
    res = mta.run_lda(matrices["lda_matrix"], args.n_topics)
    words = mta.top_words_per_topic(res["topicwords"], matrices["lda_names"])
    dist = mta.topic_distribution_per_doc(res["doctopic"], labels)
    dom = mta.dominant_topic_per_doc(dist)

    save_dataframe(words, "lda_top_words",       output_dir, csv_json_formats)
    save_dataframe(dist,  "lda_distribution",    output_dir, csv_json_formats)
    save_dataframe(dom,   "lda_dominant_topics", output_dir, csv_json_formats)

    fig = mta.plot_topic_distribution(dist, language=args.language)
    save_figure(fig, "lda_distribution", output_dir, plot_formats)

    summary = {
        "action": "lda",
        "n_topics": args.n_topics,
        "cophenet": float(res["cophenet"]),
        "vocabulary_size": int(len(matrices["lda_names"])),
        "n_documents": int(len(labels)),
    }
    (output_dir / "lda_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  ✓ LDA complete (cophenet={res['cophenet']:.3f})")


def _plot_rolling_mean(
    rm: pd.DataFrame, language: str = "en",
) -> plt.Figure:
    """Stacked bar chart of rolling-mean topic weights — used by both
    batch and interactive modes for consistency."""
    fig, ax = plt.subplots(figsize=mta.auto_figsize(len(rm)))
    display_rm = rm.copy()
    display_rm.index = mta.truncate_labels(display_rm.index)
    display_rm.plot(kind="bar", stacked=True, colormap="Paired",
                    ax=ax, width=0.85)
    lbl = mta.get_labels(language)
    ax.set_xlabel(lbl["documents_sorted"])
    ax.set_ylabel(mta.wrap_ylabel(lbl["weight_rm"]))
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5),
              prop={"size": 8}, frameon=False)
    plt.xticks(rotation=90, fontsize=7 if len(rm) > 40 else 8)
    fig.subplots_adjust(left=0.10, right=0.82, top=0.95, bottom=0.30)
    return fig


def _plot_yearly_evolution(
    yearly: pd.DataFrame, language: str = "en",
) -> plt.Figure:
    """Line chart of yearly topic evolution — used by both batch and
    interactive modes for consistency."""
    fig, ax = plt.subplots(figsize=(8, 5))
    yearly.plot(ax=ax, marker="o", markersize=4, linewidth=1.5)
    lbl = mta.get_labels(language)
    ax.set_xlabel(lbl["year"])
    ax.set_ylabel(mta.wrap_ylabel(lbl["yearly_weight"]))
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5),
              prop={"size": 9})
    fig.subplots_adjust(left=0.12, right=0.82, top=0.95, bottom=0.12)
    return fig


def _action_evolution(args, matrices, labels,
                      output_dir, csv_json_formats, plot_formats):
    """Topic evolution through texts — needs NMF or LDA result."""
    # We need a model. If neither was run before in this batch, run NMF.
    print(f"  Computing NMF for evolution (k={args.n_topics})…")
    res = mta.run_nmf(matrices["tf_matrix"], args.n_topics)
    rm = mta.rolling_mean_distribution(
        res["doctopic"], labels, window=args.window,
    )
    save_dataframe(rm, "nmf_rolling_mean", output_dir, csv_json_formats)
    fig = _plot_rolling_mean(rm, language=args.language)
    save_figure(fig, "nmf_rolling_mean", output_dir, plot_formats)

    # Yearly aggregation (only if filenames look like YYYY-...)
    yearly, bad = mta.yearly_topic_evolution(rm)
    if not yearly.empty:
        save_dataframe(yearly, "nmf_yearly_evolution",
                       output_dir, csv_json_formats)
        fig = _plot_yearly_evolution(yearly, language=args.language)
        save_figure(fig, "nmf_yearly_evolution", output_dir, plot_formats)
        print(f"  ✓ Evolution complete ({len(yearly)} years detected, "
              f"{len(bad)} filenames without year)")
    else:
        print(f"  ⚠ Yearly aggregation skipped — no filenames start with YYYY")


def _plot_word_weights_heatmap(
    df: pd.DataFrame,
    title: str,
    xlabel: str,
    ylabel: str,
    is_documents_axis: bool = False,
) -> plt.Figure:
    """
    Render a matplotlib heatmap of `df` (rows × columns) with cell
    values annotated in black. Used for the word-weights action of
    MTA_v3.py, both for words×topics and words×documents tables.

    is_documents_axis is True when one of the axes is documents (then
    we truncate the labels). The opposite axis is always words or topics
    (short by construction).
    """
    if df.empty:
        return None

    # Sensible figsize: cells should look square-ish but capped
    n_rows, n_cols = df.shape
    cell_w = 0.6
    cell_h = 0.35
    width = min(20.0, max(6.0, 2.0 + n_cols * cell_w))
    height = min(18.0, max(3.5, 1.5 + n_rows * cell_h))
    fig, ax = plt.subplots(figsize=(width, height))

    # Truncate labels only on the documents axis
    display_index = (mta.truncate_labels(df.index) if is_documents_axis
                     else list(df.index))
    display_cols = (mta.truncate_labels(df.columns) if not is_documents_axis
                    and any(len(str(c)) > 30 for c in df.columns)
                    else list(df.columns))

    im = ax.imshow(df.values, cmap="PiYG", aspect="auto",
                   vmin=0, vmax=float(df.values.max()) if df.values.size else 1.0)

    # Tick labels
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(display_cols, rotation=45, ha="right",
                       fontsize=8)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(display_index,
                       fontsize=7 if n_rows > 30 else 9)

    # Annotate each cell in black (consistent with Streamlit heatmaps)
    if n_rows * n_cols <= 400:  # don't annotate huge heatmaps
        for i in range(n_rows):
            for j in range(n_cols):
                v = df.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="black", fontsize=7, fontweight="bold")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(mta.wrap_ylabel(ylabel))
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.6)
    fig.subplots_adjust(left=0.20, right=0.95, top=0.92, bottom=0.20)
    return fig


def _action_word_weights(args, matrices, labels,
                         output_dir, csv_json_formats, plot_formats):
    """Word weights in topics and in documents — with heatmap plots."""
    if not args.words:
        print("  ✗ --words is required for word-weights action",
              file=sys.stderr)
        return
    word_list = [w.strip().lower() for chunk in args.words.split(",")
                 for w in chunk.split()]
    word_list = [w for w in word_list if w]
    print(f"  Words: {word_list}")

    # Heads-up message for potentially slow rendering on large corpora
    n_docs = len(labels)
    if n_docs > 200:
        print(f"  ℹ Note: the words×documents heatmap may take a few "
              f"seconds to render with {n_docs} documents.")

    # Need an NMF model to get topicwords
    print(f"  Running NMF (k={args.n_topics})…")
    res = mta.run_nmf(matrices["tf_matrix"], args.n_topics)

    df_topics, missing_topics = mta.words_weight_per_topic(
        res["topicwords"], matrices["tf_names"], word_list,
    )
    df_docs, missing_docs = mta.words_weight_per_document(
        matrices["tf_matrix"], matrices["tf_names"], labels, word_list,
    )

    if missing_topics:
        print(f"  ⚠ Words not in NMF vocabulary: {missing_topics}")

    save_dataframe(df_topics, "word_weights_topics",
                   output_dir, csv_json_formats)
    save_dataframe(df_docs,   "word_weights_documents",
                   output_dir, csv_json_formats)

    # Heatmap: words × topics
    if not df_topics.empty:
        lbl = mta.get_labels(args.language)
        fig = _plot_word_weights_heatmap(
            df_topics,
            title=f"{lbl['weight']} — {lbl['topic']}",
            xlabel=lbl["topic"], ylabel=lbl["word"],
            is_documents_axis=False,
        )
        save_figure(fig, "word_weights_topics", output_dir, plot_formats)

    # Heatmap: words × documents (potentially large)
    if not df_docs.empty:
        lbl = mta.get_labels(args.language)
        # For large corpora, keep only the top-N documents by total weight
        df_docs_view = df_docs
        if n_docs > 50:
            df_docs_view = (
                df_docs.assign(_total=df_docs.sum(axis=1))
                .sort_values("_total", ascending=False)
                .head(50)
                .drop(columns="_total")
            )
            print(f"  ℹ Heatmap shows top-50 documents (out of {n_docs}) "
                  f"by total weight. Full table in CSV/JSON.")
        fig = _plot_word_weights_heatmap(
            df_docs_view,
            title=f"{lbl['weight']} — {lbl['documents']}",
            xlabel=lbl["word"], ylabel=lbl["documents"],
            is_documents_axis=True,
        )
        save_figure(fig, "word_weights_documents",
                    output_dir, plot_formats)

    print(f"  ✓ Word weights complete")


def _plot_semantic_cloud(*args, **kwargs):
    """
    Thin wrapper around mta.plot_semantic_cloud — kept under its original
    private name so existing batch-mode and interactive-menu call sites
    don't need to change. The implementation now lives in mta_core so
    that the Streamlit page can call it too.
    """
    return mta.plot_semantic_cloud(*args, **kwargs)


def _action_semantic(args, matrices, corpus_wo, labels,
                     output_dir, csv_json_formats, plot_formats):
    """Semantic context: similar words, 2D cloud, best documents."""
    if not args.words:
        print("  ✗ --words is required for semantic action",
              file=sys.stderr)
        return
    word_list = [w.strip().lower() for chunk in args.words.split(",")
                 for w in chunk.split()]
    word_list = [w for w in word_list if w]
    print(f"  Seed words: {word_list}")

    if args.embedding_method == "word2vec":
        print("  Training Word2Vec (gensim)…")
        try:
            from gensim.models import Word2Vec
        except ImportError:
            print("  ✗ gensim is not installed — install it or use "
                  "--embedding-method cooccurrence", file=sys.stderr)
            return
        tokenized = mta.tokenize_for_cooccurrence(corpus_wo)
        model = Word2Vec(tokenized, vector_size=100, window=5,
                         min_count=2, workers=2, sample=1e-5, epochs=5)
        embeddings = {w: model.wv[w].copy() for w in model.wv.index_to_key}
    else:
        print("  Computing co-occurrence embeddings…")
        embeddings, _ = mta.build_cooccurrence_embeddings(
            corpus_wo, window=5, min_count=2, n_dims=100,
        )

    # Similar words
    rows = []
    similar_map = {}
    for w in word_list:
        sims = mta.most_similar_words(embeddings, w, topn=10)
        similar_map[w] = [s[0] for s in sims]
        for rank, (sw, score) in enumerate(sims, start=1):
            rows.append({"Query word": w, "Rank": rank,
                         "Similar word": sw, "Similarity": score})
    df_similar = pd.DataFrame(rows)
    save_dataframe(df_similar, "semantic_similar_words",
                   output_dir, csv_json_formats)

    # 2D cloud
    df_cloud = mta.pca_project_word_clusters(
        embeddings, word_list, neighbours_per_seed=50,
    )
    save_dataframe(df_cloud, "semantic_cloud_coordinates",
                   output_dir, csv_json_formats)
    fig = _plot_semantic_cloud(
        df_cloud,
        max_annotations_per_cluster=int(getattr(
            args, "max_labels_per_cluster", 15)),
    )
    if fig is not None:
        save_figure(fig, "semantic_cloud", output_dir, plot_formats)

    # Best documents
    df_best, _ = mta.best_documents_for_words(
        matrices["tf_matrix"], matrices["tf_names"], labels,
        word_list, similar_map,
    )
    save_dataframe(df_best, "semantic_best_documents",
                   output_dir, csv_json_formats)
    print(f"  ✓ Semantic context complete")


def _plot_group_boxplot(
    distribution: pd.DataFrame,
    groups: dict,
    topic: str,
    tests_for_topic: pd.DataFrame,
    language: str = "en",
) -> "plt.Figure":
    """
    Build a matplotlib boxplot for ONE topic, comparing all groups,
    with individual data points jittered on top of the boxes.

    The title includes the topic name and a summary of which pairs
    of groups are significant after BH correction.

    Parameters
    ----------
    distribution : pd.DataFrame
        Topic distribution per document (rows = documents).
    groups : dict[str, str]
        Mapping filename → group code.
    topic : str
        The column of `distribution` to plot.
    tests_for_topic : pd.DataFrame
        Subset of the pairwise test results for this topic.
    """
    import matplotlib.pyplot as plt
    from matplotlib import colormaps

    lbl = mta.get_labels(language)

    # Build per-group lists of values
    df = distribution[[topic]].copy()
    df["__group__"] = df.index.map(groups)
    df = df.dropna(subset=["__group__"])
    group_codes = sorted(df["__group__"].unique())
    values_per_group = [df[df["__group__"] == g][topic].values
                        for g in group_codes]
    n_per_group = [len(v) for v in values_per_group]

    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(group_codes)), 5))
    palette = colormaps["tab10"]
    colors = [palette(i % 10) for i in range(len(group_codes))]

    # Boxplot (using patch_artist to colour fill)
    bp = ax.boxplot(
        values_per_group,
        patch_artist=True,
        widths=0.5,
        showfliers=False,  # outliers will appear via the scatter below
    )
    # Set tick labels separately (avoids labels= vs tick_labels= changes
    # between matplotlib versions)
    ax.set_xticks(range(1, len(group_codes) + 1))
    ax.set_xticklabels(
        [f"{g}\n(n={n})" for g, n in zip(group_codes, n_per_group)]
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.4)
        patch.set_edgecolor("black")
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    # Jittered points on top of each box
    rng = np.random.default_rng(seed=0)
    for i, vals in enumerate(values_per_group):
        if len(vals) == 0:
            continue
        x_jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(np.full_like(vals, i + 1) + x_jitter,
                   vals, color=colors[i], alpha=0.65, s=18, edgecolor="white",
                   linewidths=0.5, zorder=3)

    ax.set_ylabel(f"{lbl['weight']} — {topic}")

    # Build a short title that summarizes the significant pairs
    sig_pairs = []
    for _, row in tests_for_topic.iterrows():
        if not pd.isna(row["p_welch_BH"]) and row["p_welch_BH"] < 0.05:
            stars = "***" if row["p_welch_BH"] < 0.001 else (
                    "**"  if row["p_welch_BH"] < 0.01  else "*")
            sig_pairs.append(f"{row['group_A']} vs {row['group_B']} {stars}")
    if sig_pairs:
        ax.set_title(f"{topic} — significant pairs (BH): "
                     + "; ".join(sig_pairs),
                     fontsize=10)
    else:
        ax.set_title(topic, fontsize=11)

    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig


def _action_compare_groups(args, matrices, labels,
                            output_dir, csv_json_formats, plot_formats):
    """
    Group comparison: for each topic, test whether documents differ
    significantly between groups (Welch's t-test + Mann-Whitney U,
    with Benjamini-Hochberg correction).
    """
    # Need an NMF model for the topic distribution
    print(f"  Running NMF (k={args.n_topics}) for distribution…")
    res = mta.run_nmf(matrices["tf_matrix"], args.n_topics)
    distribution = mta.topic_distribution_per_doc(res["doctopic"], labels)

    # Define groups
    if args.groups_from == "filenames":
        print(f"  Extracting groups from filename position "
              f"{args.group_position} (sep={args.group_separator!r})…")
        groups, skipped = mta.extract_groups_from_filenames(
            labels, position=args.group_position,
            separator=args.group_separator,
        )
        if skipped:
            print(f"  ⚠ {len(skipped)} file(s) skipped "
                  f"(no part at position {args.group_position}): "
                  f"{', '.join(skipped[:3])}"
                  + (" …" if len(skipped) > 3 else ""))
        groupings_to_run = {f"position_{args.group_position}": groups}
    else:
        if not args.groups_csv:
            print("  ✗ --groups-csv is required when --groups-from csv",
                  file=sys.stderr)
            return
        print(f"  Loading groups from CSV: {args.groups_csv}")
        try:
            groupings_to_run, skipped = mta.extract_groups_from_csv(
                args.groups_csv, labels,
            )
        except ValueError as e:
            print(f"  ✗ CSV error: {e}", file=sys.stderr)
            return
        if skipped:
            print(f"  ⚠ {len(skipped)} file(s) in corpus not listed "
                  f"in CSV: {', '.join(skipped[:3])}"
                  + (" …" if len(skipped) > 3 else ""))

    # Run statistics and tests for each grouping
    for grouping_name, groups in groupings_to_run.items():
        print(f"\n  Grouping '{grouping_name}':")
        if not groups:
            print(f"    ⚠ No documents could be grouped — skipping.")
            continue
        group_counts = pd.Series(list(groups.values())).value_counts()
        print(f"    Counts: "
              + ", ".join(f"{g}: {n}" for g, n in group_counts.items()))
        if len(group_counts) < 2:
            print(f"    ⚠ Need at least 2 groups — skipping.")
            continue
        small = group_counts[group_counts < 30]
        if len(small) > 0:
            print(f"    ⚠ Small samples (n<30): "
                  + ", ".join(f"{g}: {n}" for g, n in small.items())
                  + " — tests will be flagged but still computed.")

        stats_df = mta.compute_group_statistics(distribution, groups)
        save_dataframe(stats_df,
                       f"group_stats_{grouping_name}",
                       output_dir, csv_json_formats)

        tests_df = mta.compare_groups_pairwise(distribution, groups)
        save_dataframe(tests_df,
                       f"group_tests_{grouping_name}",
                       output_dir, csv_json_formats)

        # Bar chart with error bars (mean ± std per topic per group)
        if not stats_df.empty:
            mean_df = stats_df.xs("mean", level=0)
            std_df = stats_df.xs("std", level=0)
            fig, ax = plt.subplots(figsize=mta.auto_figsize(len(mean_df) * 2))
            x = np.arange(len(mean_df.index))
            n_groups = len(mean_df.columns)
            width = 0.8 / max(n_groups, 1)
            from matplotlib import colormaps
            palette = colormaps["tab10"]
            for i, group in enumerate(mean_df.columns):
                offset = (i - n_groups / 2 + 0.5) * width
                ax.bar(x + offset, mean_df[group], width,
                       yerr=std_df[group], capsize=3, label=str(group),
                       color=palette(i % 10))
            ax.set_xticks(x)
            ax.set_xticklabels(mean_df.index, rotation=45, ha="right")
            lbl = mta.get_labels(args.language)
            ax.set_xlabel(lbl["topic"])
            ax.set_ylabel(lbl["weight"])
            ax.legend(title=str(grouping_name), loc="best", fontsize=9)
            fig.tight_layout()
            save_figure(fig, f"group_stats_{grouping_name}",
                        output_dir, plot_formats)

        # Count significant findings at the standard alpha = 0.05
        if not tests_df.empty:
            sig_welch    = (tests_df["p_welch"]       < 0.05).sum()
            sig_welch_bh = (tests_df["p_welch_BH"]    < 0.05).sum()
            sig_mwu      = (tests_df["p_mannwhitney"] < 0.05).sum()
            sig_mwu_bh   = (tests_df["p_mannwhitney_BH"] < 0.05).sum()
            n_total = len(tests_df)
            print(f"    Significant (α=0.05) — Welch: {sig_welch}/{n_total} "
                  f"(BH-corrected: {sig_welch_bh}); "
                  f"MWU: {sig_mwu}/{n_total} "
                  f"(BH-corrected: {sig_mwu_bh})")

            # Box-plots: one per topic that has at least ONE significant
            # pairwise comparison (Welch BH-corrected < 0.05). This
            # focuses the visual output on the findings worth looking at.
            sig_topics = tests_df.loc[
                tests_df["p_welch_BH"] < 0.05, "topic"
            ].unique().tolist()
            if sig_topics:
                print(f"    Generating box-plots for {len(sig_topics)} "
                      f"significant topic(s): {', '.join(sig_topics)}")
                for topic in sig_topics:
                    tests_for_topic = tests_df[tests_df["topic"] == topic]
                    fig = _plot_group_boxplot(
                        distribution, groups, topic, tests_for_topic,
                        language=args.language,
                    )
                    save_figure(
                        fig,
                        f"boxplot_{grouping_name}_{topic}",
                        output_dir, plot_formats,
                    )
            else:
                print(f"    No topic survives BH correction "
                      f"— no boxplots generated.")

    print(f"  ✓ Group comparison complete")


# -----------------------------------------------------------------------------
# Network views — bipartite topic↔document, topic↔words, combined
# -----------------------------------------------------------------------------

def _action_network(args, matrices, labels,
                    output_dir, plot_formats):
    """
    Render bipartite network views of the topic model.

    Three figures (controlled by --network-kind):
      • doc      — topics ↔ documents
      • word     — topics ↔ top-N representative words per topic
      • combined — topics + documents + top-N words on one canvas
      • all      — produce the three above (default)

    Topic node sizes encode the cumulated edge weights attached to each
    topic. Use --emphasize-differences to amplify the contrast on
    balanced corpora.
    """
    # 1. Run the chosen topic model
    method = getattr(args, "network_method", "nmf")
    if method == "lda":
        print(f"  Running LDA with k={args.n_topics} for network views…")
        res = mta.run_lda(matrices["lda_matrix"], args.n_topics)
    else:
        print(f"  Running NMF with k={args.n_topics} for network views…")
        res = mta.run_nmf(matrices["tf_matrix"], args.n_topics)

    doctopic = res["doctopic"]
    topicwords = res["topicwords"]
    vocab = matrices["tf_names"] if method == "nmf" else matrices["lda_names"]

    # 2. Auto-derive topic display names from each topic's top 3 words
    topic_names = []
    for k in range(topicwords.shape[0]):
        top_idx = np.argsort(topicwords[k])[::-1][:3]
        topic_names.append(" / ".join([vocab[i] for i in top_idx]))

    kind = getattr(args, "network_kind", "all")
    kinds = ["doc", "word", "combined"] if kind == "all" else [kind]

    title_method = method.upper()
    emph = bool(getattr(args, "emphasize_differences", False))
    top_n = int(getattr(args, "network_top_n", 50))
    min_edge = float(getattr(args, "network_min_edge", 0.10))

    if "doc" in kinds:
        print(f"  Rendering topic↔document network "
              f"(min_edge={min_edge:.0%})…")
        fig = mtanet.plot_topic_document_network(
            doctopic=doctopic,
            labels=labels,
            topic_names=topic_names,
            min_weight_pct=min_edge,
            title=f"Topics ↔ Documents ({title_method}, K={args.n_topics})",
            emphasize_differences=emph,
        )
        save_figure(fig, f"network_{method}_topic_document",
                    output_dir, plot_formats)
        plt.close(fig)

    if "word" in kinds:
        print(f"  Rendering topic↔word network (top {top_n} words)…")
        fig = mtanet.plot_topic_word_network(
            topicwords=topicwords,
            vocab=vocab,
            topic_names=topic_names,
            top_n=top_n,
            title=f"Topics ↔ Top-{top_n} words "
                  f"({title_method}, K={args.n_topics})",
            emphasize_differences=emph,
        )
        save_figure(fig, f"network_{method}_topic_words",
                    output_dir, plot_formats)
        plt.close(fig)

    if "combined" in kinds:
        # Use smaller top_n for combined view to keep it readable
        comb_top_n = min(25, top_n)
        print(f"  Rendering combined network (top {comb_top_n} words "
              f"+ docs, min_edge={max(min_edge, 0.20):.0%})…")
        fig = mtanet.plot_combined_network(
            doctopic=doctopic,
            topicwords=topicwords,
            labels=labels,
            vocab=vocab,
            topic_names=topic_names,
            top_n_words=comb_top_n,
            min_doc_weight_pct=max(min_edge, 0.20),
            title=f"Topics + Documents + Top-words "
                  f"({title_method}, K={args.n_topics})",
            emphasize_differences=emph,
        )
        save_figure(fig, f"network_{method}_combined",
                    output_dir, plot_formats)
        plt.close(fig)

    print(f"  ✓ Network views complete")


# -----------------------------------------------------------------------------
# Axis projection — user-defined semantic axes on the doctopic matrix
# -----------------------------------------------------------------------------

def _parse_axis_spec(spec: str, n_topics: int) -> tuple:
    """
    Parse a CLI axis spec like "0,1 / 2,3" into (left_pole, right_pole).

    Format: "L / R" where each side is a comma-separated list of
    topic indices (0-based). Either side may be empty.

    Raises ValueError on malformed input.
    """
    if "/" not in spec:
        raise ValueError(
            f"Axis spec {spec!r} must contain '/' separating left "
            "and right poles. Example: '0,1 / 2,3'"
        )
    left_str, right_str = spec.split("/", 1)

    def _to_list(s):
        s = s.strip()
        if not s:
            return []
        items = [x.strip() for x in s.split(",") if x.strip()]
        try:
            idx = [int(x) for x in items]
        except ValueError:
            raise ValueError(
                f"Could not parse topic indices in {s!r}: "
                "expected integers"
            )
        for k in idx:
            if not (0 <= k < n_topics):
                raise ValueError(
                    f"Topic index {k} out of range [0, {n_topics - 1}]"
                )
        return idx

    return _to_list(left_str), _to_list(right_str)


def _axis_label_from_words(pole_words: list, max_chars: int = 25) -> str:
    """Build a short axis label from the top words of a pole."""
    if not pole_words:
        return ""
    words = [w for w, _ in pole_words[:3]]
    label = "/".join(words)
    if len(label) > max_chars:
        label = label[:max_chars - 1] + "…"
    return label


def _action_axis_analysis(args, matrices, labels,
                            output_dir, csv_json_formats, plot_formats):
    """
    Unified axis analysis: projects documents onto user-defined semantic
    axes AND runs statistics on the resulting coordinates (enriched
    CSV export + one-way ANOVA per axis with both classical F + Tukey
    HSD and Welch F + BH-corrected pairwise t).

    Replaces the former separate `_action_axis_projection` and
    `_action_axis_stats` in MTA 3.4 — they were almost-redundant since
    the projection coordinates are also the input to the ANOVA. The two
    are now produced in a single pass, which is faster (the topic model
    is fit once) and pedagogically clearer.

    Outputs (all in `output_dir`):
        - Projection figure (PDF/PNG) — the visual plot
        - axis_export_<method>.csv/json — the enriched export
        - axis_anova_summary_<method>.csv/json — one row per axis
        - axis_anova_welch_pairwise_<method>.csv/json
        - axis_anova_tukey_pairwise_<method>.csv/json
        - axis_anova_group_summary_<method>.csv/json
        - axis_anova_boxplots_<method>.pdf/png — boxplots per group

    Required CLI options: --axis-x (and optionally --axis-y, --axis-z).
    Group factor for ANOVA: --axis-stats-group-position (defaults to
    --group-position).
    """
    if not args.axis_x:
        print("  ✗ --axis-x is required for axis-analysis action",
              file=sys.stderr)
        print("    Example: --axis-x \"0,1 / 2,3\"", file=sys.stderr)
        return

    # 1. Run the topic model (NMF or LDA)
    method = getattr(args, "axis_method", "nmf")
    if method == "lda":
        print(f"  Running LDA with k={args.n_topics}…")
        res = mta.run_lda(matrices["lda_matrix"], args.n_topics)
        vocab = list(matrices["lda_names"])
    else:
        print(f"  Running NMF with k={args.n_topics}…")
        res = mta.run_nmf(matrices["tf_matrix"], args.n_topics)
        vocab = list(matrices["tf_names"])
    doctopic = res["doctopic"]
    topicwords = res["topicwords"]
    n_topics = topicwords.shape[0]

    # 2. Parse axes
    axis_specs = [args.axis_x]
    axis_labels_user = [args.axis_x_label]
    if args.axis_y:
        axis_specs.append(args.axis_y)
        axis_labels_user.append(args.axis_y_label)
    if args.axis_z:
        if not args.axis_y:
            print("  ✗ --axis-z requires --axis-y to be set first",
                  file=sys.stderr)
            return
        axis_specs.append(args.axis_z)
        axis_labels_user.append(args.axis_z_label)

    axes = []
    for spec in axis_specs:
        try:
            axes.append(_parse_axis_spec(spec, n_topics))
        except ValueError as e:
            print(f"  ✗ Axis spec error: {e}", file=sys.stderr)
            return

    print(f"  {len(axes)} axes defined:")
    for j, (left, right) in enumerate(axes):
        print(f"    {'XYZ'[j]}: left={left}  right={right}")

    # 3. Compute endpoint words and resolve axis titles
    endpoint_words = []
    n_endpoint = int(getattr(args, "axis_endpoint_words", 5))
    for left, right in axes:
        ew = mta.axis_endpoint_words(
            topicwords, vocab, left, right,
            top_n=max(15, n_endpoint),
        )
        endpoint_words.append(ew)

    axis_titles = []
    for j, (left, right) in enumerate(axes):
        if axis_labels_user[j]:
            axis_titles.append(axis_labels_user[j])
        else:
            l = _axis_label_from_words(endpoint_words[j].get("left", []))
            r = _axis_label_from_words(endpoint_words[j].get("right", []))
            if l and r:
                axis_titles.append(f"{l} ↔ {r}")
            elif r:
                axis_titles.append(f"→ {r}")
            elif l:
                axis_titles.append(f"→ {l}")
            else:
                axis_titles.append(f"Axis {'XYZ'[j]}")

    # 4. Project documents
    coords = mta.project_documents_on_axes(doctopic, axes)

    # ------------------------------------------------------------------
    # PART A — PROJECTION (visual)
    # ------------------------------------------------------------------
    print("\n  ── Projection ──")

    # Color values for the scatter
    color_by = getattr(args, "axis_color_by", "dominant-topic")
    color_values = None
    color_label = "Group"
    if color_by == "dominant-topic":
        dom = np.argmax(doctopic, axis=1)
        topic_names_short = []
        for k in range(n_topics):
            top_idx = np.argsort(topicwords[k])[::-1][:2]
            topic_names_short.append(f"T{k+1}: " +
                                     "/".join(vocab[i] for i in top_idx))
        color_values = [topic_names_short[k] for k in dom]
        color_label = "Dominant topic"
    elif color_by == "group":
        try:
            groups, skipped = mta.extract_groups_from_filenames(
                labels, position=int(args.group_position),
                separator=args.group_separator,
            )
            if skipped:
                print(f"  ⚠ {len(skipped)} file(s) skipped at "
                      f"position {args.group_position}")
            if groups:
                color_values = [groups.get(fn, "(no group)")
                                for fn in labels]
                color_label = f"Group at position {args.group_position}"
            else:
                print("  ⚠ No groups for coloring — "
                      "using dominant topic")
                dom = np.argmax(doctopic, axis=1)
                color_values = [f"T{k+1}" for k in dom]
                color_label = "Dominant topic"
        except Exception as e:
            print(f"  ⚠ Could not derive groups for coloring: {e}")

    # Save coords (small file, useful for quick inspection)
    coord_cols = [f"axis_{'XYZ'[j].lower()}" for j in range(len(axes))]
    df_coords = pd.DataFrame(coords, columns=coord_cols)
    df_coords.insert(0, "document", labels)
    save_dataframe(df_coords, f"axis_projection_{method}_coords",
                   output_dir, csv_json_formats)

    # Project plot
    print(f"  Rendering axis projection ({len(axes)}D)…")
    fig = mta.plot_axis_projection(
        coords=coords,
        labels=labels,
        axis_titles=axis_titles,
        color_values=color_values,
        color_label=color_label,
        endpoint_words=endpoint_words,
        n_top_endpoint_words=n_endpoint,
        title=f"Axis projection ({method.upper()}, K={n_topics})",
    )
    if fig is not None:
        save_figure(fig, f"axis_projection_{method}",
                    output_dir, plot_formats)
        plt.close(fig)
    print(f"  ✓ Projection done")

    # ------------------------------------------------------------------
    # PART B — STATISTICS (enriched export + ANOVA + boxplots)
    # ------------------------------------------------------------------
    print("\n  ── Statistics ──")

    # Group factor for ANOVA. Defaults to --group-position (same as
    # compare-groups), can be overridden with --axis-stats-group-position.
    stats_position = getattr(args, "axis_stats_group_position", None)
    if stats_position is None:
        stats_position = int(args.group_position)
    sep = args.group_separator

    metadata = {}
    groups = {}
    try:
        groups, skipped = mta.extract_groups_from_filenames(
            labels, position=stats_position, separator=sep,
        )
        if skipped:
            print(f"  ⚠ {len(skipped)} file(s) have no part at "
                  f"position {stats_position}: {', '.join(skipped[:5])}"
                  + (f" (and {len(skipped)-5} more)"
                     if len(skipped) > 5 else ""))
        if groups:
            metadata[f"group_pos{stats_position}"] = groups
        else:
            print("  ⚠ No groups derived — ANOVA will be skipped.")
    except Exception as e:
        print(f"  ⚠ Could not extract groups: {e}")
        groups = {}

    # Enriched export (always saved, even without ANOVA)
    df_export = mta.build_axis_export_dataframe(
        labels=labels, coords=coords,
        axis_titles=axis_titles,
        doctopic=doctopic, topicwords=topicwords, vocab=vocab,
        metadata=metadata,
    )
    save_dataframe(df_export, f"axis_export_{method}",
                   output_dir, csv_json_formats)
    print(f"  ✓ Enriched export saved "
          f"({df_export.shape[0]} rows × "
          f"{df_export.shape[1]} columns)")

    # ANOVA, if we have groups
    if not groups:
        print("  → ANOVA skipped (no group factor)")
        print(f"\n  ✓ Axis analysis complete (projection only)")
        return

    group_aligned = [groups.get(fn, "") for fn in labels]
    min_size = int(getattr(args, "axis_stats_min_group_size", 3))

    summary_rows, welch_rows, tukey_rows, group_summary_rows = [], [], [], []
    for j, letter in enumerate("XYZ"[:len(axes)]):
        print(f"\n  ANOVA on axis {letter} — {axis_titles[j]}")
        result = mta.axis_anova_one_way(
            coord_values=coords[:, j],
            group_labels=group_aligned,
            min_group_size=min_size,
        )
        if "error" in result:
            print(f"    ⚠ {result['error']}")
            continue
        clas = result["classical_anova"]
        welch = result["welch_anova"]
        print(f"    {result['n_groups_used']} groups, "
              f"{len(result['dropped_groups'])} dropped")
        print(f"    Classical F = {clas['F']:.3f}  "
              f"p = {clas['p_value']:.4g}  "
              f"η² = {clas['eta_squared']:.3f}")
        print(f"    Welch F     = {welch['F']:.3f}  "
              f"p = {welch['p_value']:.4g}")

        summary_rows.append({
            "axis": letter, "axis_title": axis_titles[j],
            "n_groups": result["n_groups_used"],
            "n_dropped": len(result["dropped_groups"]),
            "F_classical": clas["F"], "df_num_classical": clas["df_num"],
            "df_den_classical": clas["df_den"],
            "p_classical": clas["p_value"],
            "eta_squared": clas["eta_squared"],
            "F_welch": welch["F"], "df_num_welch": welch["df_num"],
            "df_den_welch": welch["df_den"], "p_welch": welch["p_value"],
        })
        for src, dst in [("welch_pairwise", welch_rows),
                          ("tukey_pairwise", tukey_rows),
                          ("group_summary", group_summary_rows)]:
            df = result[src].copy()
            if not df.empty:
                df.insert(0, "axis", letter)
                dst.append(df)

    if summary_rows:
        save_dataframe(pd.DataFrame(summary_rows),
                       f"axis_anova_summary_{method}",
                       output_dir, csv_json_formats)
    if welch_rows:
        save_dataframe(pd.concat(welch_rows, ignore_index=True),
                       f"axis_anova_welch_pairwise_{method}",
                       output_dir, csv_json_formats)
    if tukey_rows:
        save_dataframe(pd.concat(tukey_rows, ignore_index=True),
                       f"axis_anova_tukey_pairwise_{method}",
                       output_dir, csv_json_formats)
    if group_summary_rows:
        save_dataframe(pd.concat(group_summary_rows, ignore_index=True),
                       f"axis_anova_group_summary_{method}",
                       output_dir, csv_json_formats)

    # Boxplots
    fig = mta.plot_axis_anova_boxplots(
        axis_values={"XYZ"[j]: coords[:, j] for j in range(len(axes))},
        group_labels=group_aligned,
        axis_titles={"XYZ"[j]: axis_titles[j] for j in range(len(axes))},
        title=f"ANOVA: axis coordinates by group at position "
              f"{stats_position} ({method.upper()}, K={n_topics})",
        min_group_size=min_size,
    )
    if fig is not None:
        save_figure(fig, f"axis_anova_boxplots_{method}",
                    output_dir, plot_formats)
        plt.close(fig)

    print(f"\n  ✓ Axis analysis complete "
          f"({len(summary_rows)} axes analyzed)")




# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="MTA_v3.py",
        description="Multi-Text Analyser — CLI (batch + interactive).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="If no arguments are passed, the interactive menu starts.",
    )
    p.add_argument("--corpus", type=str,
                   help="Path to a folder containing .txt files.")
    p.add_argument("--stopwords", type=str,
                   help="Path to a stopwords file (one word per line).")
    p.add_argument("--output", type=str, default=None,
                   help="Output directory (default: MTA-Results_<timestamp>).")
    p.add_argument("--language", choices=["en", "fr", "de"], default="en",
                   help="Chart label language (default: en).")
    p.add_argument("--format", choices=["csv", "plots", "both"], default="both",
                   help="What kind of output to produce (default: both).")
    p.add_argument("--plot-format", choices=["pdf", "png", "both"],
                   default="both",
                   help="Plot file format (default: both).")
    p.add_argument("--json", action="store_true",
                   help="Also save tables as JSON (for Stata/R pipelines).")
    p.add_argument("--action",
                   choices=["nmf", "lda", "evolution", "word-weights",
                            "semantic", "compare-groups", "network",
                            "axis-analysis",
                            # Aliases kept for backward compatibility
                            # (3.2/3.3 → 3.4 transition). They run the
                            # same axis-analysis action and emit a
                            # deprecation notice on stderr.
                            "axis-projection", "axis-stats",
                            "all"],
                   help="Which analysis to run (required in batch mode). "
                        "'axis-projection' and 'axis-stats' are kept as "
                        "deprecated aliases of 'axis-analysis'.")
    p.add_argument("--n-topics", type=int, default=5,
                   help="Number of topics for NMF/LDA (default: 5).")
    p.add_argument("--max-topics", type=int, default=None,
                   help="If set, run cross-validation up to this k.")
    p.add_argument("--words", type=str, default=None,
                   help="Comma- or space-separated words for word-weights "
                        "or semantic actions.")
    p.add_argument("--embedding-method",
                   choices=["cooccurrence", "word2vec"],
                   default="cooccurrence",
                   help="Embedding method for semantic action "
                        "(default: cooccurrence — no extra dependency).")
    p.add_argument("--max-labels-per-cluster", type=int, default=15,
                   help="Maximum number of word labels shown around each "
                        "seed in the 2D semantic cloud (default: 15). "
                        "Labels go to the closest neighbours of the seed; "
                        "other points are still drawn but unlabeled. Raise "
                        "this to label more words on sparse corpora; the "
                        "full word list is always saved to the CSV/JSON.")
    p.add_argument("--window", type=int, default=2,
                   help="Rolling-mean window for evolution action (default: 2).")
    p.add_argument("--min-word-length", type=int, default=3,
                   help="Minimum word length to keep (default: 3).")
    p.add_argument("--min-df", type=lambda s: int(s) if s.isdigit() else float(s),
                   default=2,
                   help="Minimum document frequency for vectorizer "
                        "(default: 2 = integer count of docs; values < 1 "
                        "are interpreted as a proportion).")
    p.add_argument("--max-df", type=float, default=0.95,
                   help="Maximum document frequency for vectorizer "
                        "(default: 0.95).")
    # Group-comparison action
    p.add_argument("--groups-from", choices=["filenames", "csv"],
                   default="filenames",
                   help="How to define groups (default: filenames).")
    p.add_argument("--group-position", type=int, default=2,
                   help="Position of the group code in the filename, "
                        "1-indexed (default: 2). Used if --groups-from filenames.")
    p.add_argument("--group-separator", type=str, default="_",
                   help="Separator character for filename splitting "
                        "(default: _). Used if --groups-from filenames.")
    p.add_argument("--groups-csv", type=str, default=None,
                   help="Path to a CSV with `filename` column + group "
                        "columns. Used if --groups-from csv.")
    # Network-views action
    p.add_argument("--network-method",
                   choices=["nmf", "lda"],
                   default="nmf",
                   help="For --action network: which topic model to plot "
                        "(default: nmf).")
    p.add_argument("--network-kind",
                   choices=["doc", "word", "combined", "all"],
                   default="all",
                   help="For --action network: which graph(s) to render "
                        "(default: all = three graphs).")
    p.add_argument("--network-top-n", type=int, default=50,
                   help="For --action network: top-N words per topic in "
                        "the topic↔word graph (default: 50).")
    p.add_argument("--network-min-edge", type=float, default=0.10,
                   help="For --action network: minimum document→topic "
                        "edge weight, relative to the document's max "
                        "topic weight, to display in the topic↔document "
                        "graph (default: 0.10 = 10%%).")
    p.add_argument("--emphasize-differences", action="store_true",
                   help="For --action network: amplify visual size "
                        "differences between topic nodes. Useful on "
                        "balanced corpora where masses are similar.")
    # Axis-projection action
    p.add_argument("--axis-method", choices=["nmf", "lda"], default="nmf",
                   help="For --action axis-projection: which topic model "
                        "to project documents from (default: nmf).")
    p.add_argument("--axis-x", type=str, default=None,
                   help="For --action axis-projection: definition of "
                        "the X axis as `LEFT / RIGHT` where each side is "
                        "a comma-separated list of topic indices "
                        "(0-based). Example: \"0,1 / 2,3\" means "
                        "axis X opposes topics {0,1} (negative pole) to "
                        "topics {2,3} (positive pole). A pole may be "
                        "empty, e.g. \"/ 2,3\".")
    p.add_argument("--axis-y", type=str, default=None,
                   help="Same syntax as --axis-x, for the Y axis "
                        "(optional). If absent, plot is 1D.")
    p.add_argument("--axis-z", type=str, default=None,
                   help="Same syntax as --axis-x, for the Z axis "
                        "(optional, 3D plot).")
    p.add_argument("--axis-x-label", type=str, default=None,
                   help="Display label for the X axis (default: "
                        "auto-generated from the pole topic indices).")
    p.add_argument("--axis-y-label", type=str, default=None,
                   help="Display label for the Y axis.")
    p.add_argument("--axis-z-label", type=str, default=None,
                   help="Display label for the Z axis.")
    p.add_argument("--axis-color-by",
                   choices=["dominant-topic", "group", "none"],
                   default="dominant-topic",
                   help="How to color the document dots: by their "
                        "dominant topic (default), by group (requires "
                        "--groups-from), or no coloring.")
    p.add_argument("--axis-endpoint-words", type=int, default=5,
                   help="Number of characteristic words shown at each "
                        "axis extremity (default: 5).")
    # Axis-stats action
    p.add_argument("--axis-stats-group-position", type=int, default=None,
                   help="For --action axis-stats: position of the group "
                        "code in the filename, 1-indexed. Defaults to "
                        "--group-position (same as compare-groups). "
                        "Use this to test the ANOVA against a different "
                        "factor than the one used for plot coloring.")
    p.add_argument("--axis-stats-min-group-size", type=int, default=3,
                   help="For --action axis-stats: minimum group size; "
                        "groups smaller than this are dropped (default: 3).")
    return p


# =============================================================================
# INTERACTIVE MODE — kept for backward-compatibility with original MTA.py
# =============================================================================

def run_interactive() -> int:
    """
    Menu-driven mode, faithful in spirit to the original MTA.py but
    delegating all the science to mta_core.
    """
    print("=" * 60)
    print("  MTA — Multi-Text Analyser (interactive mode)")
    print("=" * 60)
    print()
    print("This mode mirrors the original MTA.py menu but uses the new")
    print("mta_core engine. For batch / scripted use, run with --help.")
    print()

    # Step 1: get paths
    while True:
        corpus_path = input("Corpus folder (path to .txt files): ").strip()
        if os.path.isdir(corpus_path):
            break
        print(f"  ✗ Not a directory: {corpus_path}")
    while True:
        stopwords_path = input("Stopwords file (.txt): ").strip()
        if os.path.isfile(stopwords_path):
            break
        print(f"  ✗ Not a file: {stopwords_path}")

    language = input("Language for chart labels (en/fr/de) [en]: ").strip() or "en"
    if language not in ("en", "fr", "de"):
        language = "en"

    print()
    raw_texts, labels = load_corpus(corpus_path, verbose=True)
    stopwords = load_stopwords(stopwords_path)
    print(f"  ✓ {len(raw_texts):,} documents, {len(stopwords):,} stopwords")

    # Preprocess + matrices
    min_word_length = int(input("Min word length [3]: ").strip() or "3")
    print("Cleaning corpus + building TF-IDF / Count matrices…")
    corpus_wo, corpus_re = mta.preprocess_corpus(
        raw_texts, stopwords, min_word_length=min_word_length,
    )
    matrices = mta.build_matrices(corpus_wo, stopwords,
                                  min_df=2, max_df=0.95)
    print(f"  ✓ Vocabulary: {len(matrices['tf_names']):,} words")

    output_dir = prepare_output_dir(None)
    print(f"  Outputs will be written to {output_dir.resolve()}")
    save_dataframe(matrices["df_tfidf"], "tfidf_matrix",
                   output_dir, {"csv", "json"})

    # Session state for the interactive menu (replaces original globals)
    session = {
        "matrices": matrices,
        "corpus_wo": corpus_wo,
        "corpus_re": corpus_re,
        "labels": labels,
        "language": language,
        "output_dir": output_dir,
        "nmf": None,
        "lda": None,
    }

    # Main menu loop
    while True:
        print()
        print("─" * 60)
        print(" MAIN MENU")
        print("─" * 60)
        print("  1. NMF and/or LDA topic modelling (with cross-validation)")
        print("  2. Topic evolution through texts (rolling mean, yearly)")
        print("  3. Weight of given words in topics and documents")
        print("  4. Semantic context (similar words + 2D cloud)")
        print("  5. Group comparison (significance tests)")
        print("  6. Network views (topic↔doc, topic↔words, combined)")
        print("  7. Axis analysis (projection + ANOVA on user-defined axes)")
        print("  0. Quit")
        print()
        choice = input("Your choice [0-7]: ").strip()

        if choice == "0":
            print("\nGoodbye!")
            return 0
        elif choice == "1":
            _interactive_menu_1(session)
        elif choice == "2":
            _interactive_menu_2(session)
        elif choice == "3":
            _interactive_menu_3(session)
        elif choice == "4":
            _interactive_menu_4(session)
        elif choice == "5":
            _interactive_menu_5(session)
        elif choice == "6":
            _interactive_menu_6(session)
        elif choice == "7":
            _interactive_menu_7(session)
        else:
            print(f"  ✗ Invalid choice: {choice!r}")


def _interactive_menu_1(session):
    """Menu 1 — NMF / LDA with optional cross-validation."""
    do_cv = input("Run cross-validation first? (y/N) ").strip().lower() == "y"
    if do_cv:
        max_k = int(input("  Max topics to test [8]: ").strip() or "8")
        n_total = (max_k - 1) * 3
        cb = lambda i, t, lbl: progress_bar(i, t, prefix=f"  {lbl}", length=20)
        metrics = mta.compute_topic_metrics(
            session["matrices"]["tf_matrix"],
            session["matrices"]["lda_matrix"],
            session["matrices"]["dense_a"],
            max_topics=max_k, progress_callback=cb,
        )
        print("\n  Suggested numbers of topics:")
        for name, vals in metrics["suggestions"].items():
            print(f"    {name:20s} : {vals[0] if vals else '—'}")

    n_topics = int(input("Number of topics [5]: ").strip() or "5")
    do_nmf = input("Run NMF? (Y/n) ").strip().lower() != "n"
    do_lda = input("Run LDA? (Y/n) ").strip().lower() != "n"

    out = session["output_dir"]
    if do_nmf:
        print(f"\n  Running NMF with k={n_topics}…")
        res = mta.run_nmf(session["matrices"]["tf_matrix"], n_topics)
        words = mta.top_words_per_topic(res["topicwords"],
                                        session["matrices"]["tf_names"])
        dist = mta.topic_distribution_per_doc(res["doctopic"], session["labels"])
        save_dataframe(words, "nmf_top_words", out, {"csv", "json"})
        save_dataframe(dist,  "nmf_distribution", out, {"csv", "json"})
        save_dataframe(mta.dominant_topic_per_doc(dist),
                       "nmf_dominant_topics", out, {"csv", "json"})
        sentences = mta.best_sentences_per_topic(words, session["corpus_re"])
        save_dataframe(sentences, "nmf_best_sentences", out, {"csv", "json"})
        fig = mta.plot_topic_distribution(dist, language=session["language"])
        save_figure(fig, "nmf_distribution", out, {"pdf", "png"})
        session["nmf"] = res
        print(f"  ✓ NMF done (cophenet={res['cophenet']:.3f})")

    if do_lda:
        print(f"\n  Running LDA with k={n_topics}…")
        res = mta.run_lda(session["matrices"]["lda_matrix"], n_topics)
        words = mta.top_words_per_topic(res["topicwords"],
                                        session["matrices"]["lda_names"])
        dist = mta.topic_distribution_per_doc(res["doctopic"], session["labels"])
        save_dataframe(words, "lda_top_words", out, {"csv", "json"})
        save_dataframe(dist,  "lda_distribution", out, {"csv", "json"})
        save_dataframe(mta.dominant_topic_per_doc(dist),
                       "lda_dominant_topics", out, {"csv", "json"})
        fig = mta.plot_topic_distribution(dist, language=session["language"])
        save_figure(fig, "lda_distribution", out, {"pdf", "png"})
        session["lda"] = res
        print(f"  ✓ LDA done (cophenet={res['cophenet']:.3f})")


def _interactive_menu_2(session):
    """Menu 2 — Topic evolution (CSV/JSON + rolling-mean & yearly plots)."""
    if session["nmf"] is None and session["lda"] is None:
        print("  ⚠ Run NMF or LDA first (menu 1).")
        return
    which = "nmf" if session["nmf"] else "lda"
    window = int(input("Rolling-mean window [2]: ").strip() or "2")
    res = session[which]
    rm = mta.rolling_mean_distribution(res["doctopic"], session["labels"],
                                       window=window)
    save_dataframe(rm, f"{which}_rolling_mean",
                   session["output_dir"], {"csv", "json"})
    # Rolling-mean plot (PDF + PNG)
    fig = _plot_rolling_mean(rm, language=session["language"])
    save_figure(fig, f"{which}_rolling_mean",
                session["output_dir"], {"pdf", "png"})

    yearly, bad = mta.yearly_topic_evolution(rm)
    if not yearly.empty:
        save_dataframe(yearly, f"{which}_yearly_evolution",
                       session["output_dir"], {"csv", "json"})
        # Yearly evolution plot (PDF + PNG)
        fig = _plot_yearly_evolution(yearly, language=session["language"])
        save_figure(fig, f"{which}_yearly_evolution",
                    session["output_dir"], {"pdf", "png"})
        print(f"  ✓ Evolution: {len(yearly)} years detected "
              f"(CSV/JSON + rolling-mean & yearly plots PDF/PNG)")
    else:
        print("  ⚠ No filename starts with YYYY — yearly aggregation skipped")
        print(f"  ✓ Rolling-mean saved (CSV/JSON + plot PDF/PNG)")


def _interactive_menu_3(session):
    """Menu 3 — Word weights (CSV/JSON + heatmap plots)."""
    if session["nmf"] is None:
        print("  ⚠ Run NMF first (menu 1).")
        return
    words_str = input("Words to analyze (comma- or space-separated): ").strip()
    if not words_str:
        return
    word_list = [w.strip().lower() for chunk in words_str.split(",")
                 for w in chunk.split() if w.strip()]

    df_t, missing = mta.words_weight_per_topic(
        session["nmf"]["topicwords"], session["matrices"]["tf_names"],
        word_list,
    )
    if missing:
        print(f"  ⚠ Not in vocabulary: {missing}")
    df_d, _ = mta.words_weight_per_document(
        session["matrices"]["tf_matrix"], session["matrices"]["tf_names"],
        session["labels"], word_list,
    )
    save_dataframe(df_t, "word_weights_topics",
                   session["output_dir"], {"csv", "json"})
    save_dataframe(df_d, "word_weights_documents",
                   session["output_dir"], {"csv", "json"})

    # Heatmaps
    lbl = mta.get_labels(session["language"])
    if not df_t.empty:
        fig = _plot_word_weights_heatmap(
            df_t, title=f"{lbl['weight']} — {lbl['topic']}",
            xlabel=lbl["topic"], ylabel=lbl["word"],
            is_documents_axis=False,
        )
        save_figure(fig, "word_weights_topics",
                    session["output_dir"], {"pdf", "png"})
    if not df_d.empty:
        df_view = df_d
        if len(df_d) > 50:
            df_view = (
                df_d.assign(_t=df_d.sum(axis=1))
                .sort_values("_t", ascending=False).head(50).drop(columns="_t")
            )
            print(f"  ℹ Heatmap shows top-50 documents (out of {len(df_d)}).")
        fig = _plot_word_weights_heatmap(
            df_view, title=f"{lbl['weight']} — {lbl['documents']}",
            xlabel=lbl["word"], ylabel=lbl["documents"],
            is_documents_axis=True,
        )
        save_figure(fig, "word_weights_documents",
                    session["output_dir"], {"pdf", "png"})
    print(f"  ✓ Word weights saved (CSV/JSON + heatmap PDF/PNG)")


def _interactive_menu_4(session):
    """Menu 4 — Semantic context."""
    words_str = input("Seed words (comma- or space-separated): ").strip()
    if not words_str:
        return
    word_list = [w.strip().lower() for chunk in words_str.split(",")
                 for w in chunk.split() if w.strip()]
    method = (input("Embedding method [cooccurrence/word2vec] "
                    "(default: cooccurrence): ").strip().lower()
              or "cooccurrence")

    if method == "word2vec":
        try:
            from gensim.models import Word2Vec
        except ImportError:
            print("  ✗ gensim not installed; falling back to cooccurrence")
            method = "cooccurrence"

    if method == "word2vec":
        tokenized = mta.tokenize_for_cooccurrence(session["corpus_wo"])
        model = Word2Vec(tokenized, vector_size=100, window=5,
                         min_count=2, workers=2, sample=1e-5, epochs=5)
        embeddings = {w: model.wv[w].copy() for w in model.wv.index_to_key}
    else:
        embeddings, _ = mta.build_cooccurrence_embeddings(
            session["corpus_wo"], window=5, min_count=2, n_dims=100,
        )

    rows = []
    similar_map = {}
    for w in word_list:
        sims = mta.most_similar_words(embeddings, w, topn=10)
        similar_map[w] = [s[0] for s in sims]
        for rank, (sw, sc) in enumerate(sims, start=1):
            rows.append({"Query word": w, "Rank": rank,
                         "Similar word": sw, "Similarity": sc})
    df_sim = pd.DataFrame(rows)
    save_dataframe(df_sim, "semantic_similar_words",
                   session["output_dir"], {"csv", "json"})

    df_cloud = mta.pca_project_word_clusters(
        embeddings, word_list, neighbours_per_seed=50,
    )
    save_dataframe(df_cloud, "semantic_cloud_coordinates",
                   session["output_dir"], {"csv", "json"})

    # 2D scatter plot (same logic as batch mode)
    try:
        max_labels = int(input(
            "Max labels per seed in the 2D cloud [15]: "
        ).strip() or "15")
    except ValueError:
        max_labels = 15
    fig = _plot_semantic_cloud(df_cloud,
                               max_annotations_per_cluster=max_labels)
    if fig is not None:
        save_figure(fig, "semantic_cloud",
                    session["output_dir"], {"pdf", "png"})

    df_best, _ = mta.best_documents_for_words(
        session["matrices"]["tf_matrix"],
        session["matrices"]["tf_names"],
        session["labels"], word_list, similar_map,
    )
    save_dataframe(df_best, "semantic_best_documents",
                   session["output_dir"], {"csv", "json"})
    print(f"  ✓ Semantic context saved (CSV/JSON + 2D cloud PDF/PNG)")


def _interactive_menu_5(session):
    """Menu 5 — Group comparison (significance tests)."""
    if session["nmf"] is None and session["lda"] is None:
        print("  ⚠ Run NMF or LDA first (menu 1).")
        return

    # Pick a model
    if session["nmf"] is not None and session["lda"] is not None:
        which = input("Use NMF or LDA distribution? [nmf/lda, default nmf]: "
                      ).strip().lower() or "nmf"
        if which not in ("nmf", "lda"):
            which = "nmf"
    else:
        which = "nmf" if session["nmf"] else "lda"

    res = session[which]
    distribution = mta.topic_distribution_per_doc(
        res["doctopic"], session["labels"]
    )

    # Define groups
    method = input(
        "Define groups from: filenames or csv? [filenames/csv, default filenames]: "
    ).strip().lower() or "filenames"

    if method == "csv":
        csv_path = input("Path to groups CSV (must have 'filename' column): "
                         ).strip()
        if not csv_path or not os.path.isfile(csv_path):
            print(f"  ✗ File not found: {csv_path}")
            return
        try:
            groupings_to_run, skipped = mta.extract_groups_from_csv(
                csv_path, session["labels"],
            )
        except ValueError as e:
            print(f"  ✗ CSV error: {e}")
            return
        if skipped:
            print(f"  ⚠ {len(skipped)} file(s) not in CSV: "
                  f"{', '.join(skipped[:3])}"
                  + (" …" if len(skipped) > 3 else ""))
    else:
        try:
            position = int(input(
                "Position of group code in filename (1-indexed) [2]: "
            ).strip() or "2")
        except ValueError:
            position = 2
        separator = input("Separator [_]: ").strip() or "_"
        groups, skipped = mta.extract_groups_from_filenames(
            session["labels"], position=position, separator=separator,
        )
        if skipped:
            print(f"  ⚠ {len(skipped)} file(s) skipped (no part at "
                  f"position {position}): {', '.join(skipped[:3])}"
                  + (" …" if len(skipped) > 3 else ""))
        groupings_to_run = {f"position_{position}": groups}

    # Run for each grouping
    for grouping_name, groups in groupings_to_run.items():
        print(f"\n  Grouping '{grouping_name}':")
        if not groups:
            print(f"    ⚠ No groups — skipping.")
            continue
        gc = pd.Series(list(groups.values())).value_counts()
        print(f"    Counts: " + ", ".join(f"{g}: {n}" for g, n in gc.items()))
        if len(gc) < 2:
            print(f"    ⚠ Need ≥ 2 groups — skipping.")
            continue
        small = gc[gc < 30]
        if len(small) > 0:
            print(f"    ⚠ Small samples: "
                  + ", ".join(f"{g}: {n}" for g, n in small.items()))

        stats_df = mta.compute_group_statistics(distribution, groups)
        save_dataframe(stats_df, f"group_stats_{grouping_name}",
                       session["output_dir"], {"csv", "json"})

        tests_df = mta.compare_groups_pairwise(distribution, groups)
        save_dataframe(tests_df, f"group_tests_{grouping_name}",
                       session["output_dir"], {"csv", "json"})

        # Quick summary
        if not tests_df.empty:
            sig = (tests_df["p_welch"] < 0.05).sum()
            sig_bh = (tests_df["p_welch_BH"] < 0.05).sum()
            n_total = len(tests_df)
            print(f"    Significant (Welch α=0.05): {sig}/{n_total} "
                  f"(BH-corrected: {sig_bh})")

            # Box-plots for topics where at least one pairwise BH-corrected
            # Welch test is significant. Same logic as batch mode.
            sig_topics = tests_df.loc[
                tests_df["p_welch_BH"] < 0.05, "topic"
            ].unique().tolist()
            if sig_topics:
                print(f"    Generating box-plots for {len(sig_topics)} "
                      f"significant topic(s): {', '.join(sig_topics)}")
                for topic in sig_topics:
                    tests_for_topic = tests_df[tests_df["topic"] == topic]
                    fig = _plot_group_boxplot(
                        distribution, groups, topic, tests_for_topic,
                        language=session["language"],
                    )
                    save_figure(
                        fig,
                        f"boxplot_{grouping_name}_{topic}",
                        session["output_dir"], {"pdf", "png"},
                    )
            else:
                print(f"    No topic survives BH correction "
                      f"— no boxplots generated.")

    print("  ✓ Group comparison saved (CSV/JSON + PDF/PNG plots)")


def _interactive_menu_6(session):
    """Menu 6 — Network views (topic↔doc, topic↔words, combined)."""
    if session["nmf"] is None and session["lda"] is None:
        print("  ✗ Run NMF and/or LDA first (menu 1).")
        return

    # Method choice
    if session["nmf"] is not None and session["lda"] is not None:
        method = input("  Use NMF or LDA? (nmf/lda) [nmf]: ").strip().lower() or "nmf"
        if method not in ("nmf", "lda"):
            method = "nmf"
    elif session["nmf"] is not None:
        method = "nmf"
        print("  Using NMF (only model available).")
    else:
        method = "lda"
        print("  Using LDA (only model available).")

    res = session[method]
    doctopic = res["doctopic"]
    topicwords = res["topicwords"]
    vocab = (session["matrices"]["tf_names"] if method == "nmf"
             else session["matrices"]["lda_names"])

    # Kind of graph
    print()
    print("  Which graph(s)?")
    print("    1. Topic ↔ Documents")
    print("    2. Topic ↔ Top-N words")
    print("    3. Combined (topics + docs + top words)")
    print("    4. All three")
    kc = input("  Choice [4]: ").strip() or "4"
    kind_map = {"1": ["doc"], "2": ["word"], "3": ["combined"],
                "4": ["doc", "word", "combined"]}
    kinds = kind_map.get(kc, ["doc", "word", "combined"])

    # Top-N words (only relevant if word or combined chosen)
    top_n = 50
    if "word" in kinds or "combined" in kinds:
        try:
            top_n = int(input("  Top-N words per topic [50]: ").strip() or "50")
        except ValueError:
            top_n = 50

    # Min edge weight
    try:
        min_edge = float(input("  Min edge weight as %% of doc max [10]: ").strip()
                         or "10") / 100.0
    except ValueError:
        min_edge = 0.10

    # Emphasis
    emph = input("  Emphasize topic-size differences? (y/N) ").strip().lower() == "y"

    # Auto-name topics
    topic_names = []
    for k in range(topicwords.shape[0]):
        top_idx = np.argsort(topicwords[k])[::-1][:3]
        topic_names.append(" / ".join([vocab[i] for i in top_idx]))

    out = session["output_dir"]
    n_topics = topicwords.shape[0]
    title_method = method.upper()

    if "doc" in kinds:
        print(f"  Rendering topic↔document network…")
        fig = mtanet.plot_topic_document_network(
            doctopic=doctopic, labels=session["labels"],
            topic_names=topic_names,
            min_weight_pct=min_edge,
            title=f"Topics ↔ Documents ({title_method}, K={n_topics})",
            emphasize_differences=emph,
        )
        save_figure(fig, f"network_{method}_topic_document",
                    out, {"pdf", "png"})
        plt.close(fig)

    if "word" in kinds:
        print(f"  Rendering topic↔word network (top {top_n})…")
        fig = mtanet.plot_topic_word_network(
            topicwords=topicwords, vocab=vocab,
            topic_names=topic_names, top_n=top_n,
            title=f"Topics ↔ Top-{top_n} words "
                  f"({title_method}, K={n_topics})",
            emphasize_differences=emph,
        )
        save_figure(fig, f"network_{method}_topic_words",
                    out, {"pdf", "png"})
        plt.close(fig)

    if "combined" in kinds:
        comb_top_n = min(25, top_n)
        print(f"  Rendering combined network…")
        fig = mtanet.plot_combined_network(
            doctopic=doctopic, topicwords=topicwords,
            labels=session["labels"], vocab=vocab,
            topic_names=topic_names,
            top_n_words=comb_top_n,
            min_doc_weight_pct=max(min_edge, 0.20),
            title=f"Topics + Documents + Top-words "
                  f"({title_method}, K={n_topics})",
            emphasize_differences=emph,
        )
        save_figure(fig, f"network_{method}_combined",
                    out, {"pdf", "png"})
        plt.close(fig)

    print(f"  ✓ Network views saved (PDF + PNG)")


def _interactive_menu_7(session):
    """
    Menu 7 — Axis analysis (projection + statistics, unified).

    Asks the user for 1–3 axes (oppositions between topic pools),
    then for an analysis mode: projection only (visual scatter),
    statistics only (enriched CSV + ANOVA), or both. The two parts share
    the same axis definition — no need to redefine the axes between
    them.
    """
    if session["nmf"] is None and session["lda"] is None:
        print("  ✗ Run NMF and/or LDA first (menu 1).")
        return

    # Method choice
    if session["nmf"] is not None and session["lda"] is not None:
        method = input("  Use NMF or LDA? (nmf/lda) [nmf]: ").strip().lower() or "nmf"
        if method not in ("nmf", "lda"):
            method = "nmf"
    elif session["nmf"] is not None:
        method = "nmf"
        print("  Using NMF (only model available).")
    else:
        method = "lda"
        print("  Using LDA (only model available).")

    res = session[method]
    doctopic = res["doctopic"]
    topicwords = res["topicwords"]
    vocab = (session["matrices"]["tf_names"] if method == "nmf"
             else session["matrices"]["lda_names"])
    n_topics = topicwords.shape[0]

    # Show topic summaries
    print("\n  Available topics:")
    for k in range(n_topics):
        top_idx = np.argsort(topicwords[k])[::-1][:5]
        top_words = ", ".join(vocab[i] for i in top_idx)
        print(f"    {k}: {top_words}")

    print("\n  Define axes as oppositions between pools of topics.")
    print("  Format: \"LEFT / RIGHT\"; example: \"0,1 / 2,3\"; "
          "either pole may be empty.")

    # Collect axis specs
    axes = []
    axis_titles = []
    for j, name in enumerate(["X", "Y", "Z"]):
        spec = input(f"\n  Axis {name} (empty to stop): ").strip()
        if not spec:
            break
        try:
            axis = _parse_axis_spec(spec, n_topics)
        except ValueError as e:
            print(f"    ✗ {e}")
            return
        axes.append(axis)
        custom = input(f"    Custom label for axis {name} "
                       "(empty for auto): ").strip()
        axis_titles.append(custom if custom else None)

    if not axes:
        print("  ✗ No axes defined, aborting.")
        return

    # Endpoint words + auto-fill titles
    endpoint_words = []
    for j, (left, right) in enumerate(axes):
        ew = mta.axis_endpoint_words(topicwords, vocab, left, right, top_n=15)
        endpoint_words.append(ew)
        if axis_titles[j] is None:
            left_words = [w for w, _ in ew.get("left", [])[:3]]
            right_words = [w for w, _ in ew.get("right", [])[:3]]
            l = "/".join(left_words) if left_words else ""
            r = "/".join(right_words) if right_words else ""
            if l and r:
                axis_titles[j] = f"{l} ↔ {r}"
            elif r:
                axis_titles[j] = f"→ {r}"
            elif l:
                axis_titles[j] = f"→ {l}"
            else:
                axis_titles[j] = f"Axis {'XYZ'[j]}"

    # Project documents (needed for both projection and stats)
    coords = mta.project_documents_on_axes(doctopic, axes)
    out = session["output_dir"]

    # Ask the user what they want to do with the axes
    print("\n  What do you want to produce?")
    print("    1. Projection only (visual scatter + PDF/PNG)")
    print("    2. Statistics only (enriched CSV export + ANOVA + boxplots)")
    print("    3. Both (recommended)")
    mode_choice = input("  Your choice [3]: ").strip() or "3"
    do_projection = mode_choice in ("1", "3")
    do_statistics = mode_choice in ("2", "3")

    # ------------------------------------------------------------------
    # PROJECTION
    # ------------------------------------------------------------------
    if do_projection:
        print("\n  ── Projection ──")

        # Coloring choice (only for the projection)
        color_choice = input(
            "  Color dots by: (1) dominant topic, "
            "(2) group from filename, (3) none [1]: "
        ).strip() or "1"
        color_values = None
        color_label = "Group"
        if color_choice == "1":
            dom = np.argmax(doctopic, axis=1)
            names = []
            for k in range(n_topics):
                top_idx = np.argsort(topicwords[k])[::-1][:2]
                names.append(f"T{k+1}: " + "/".join(vocab[i] for i in top_idx))
            color_values = [names[k] for k in dom]
            color_label = "Dominant topic"
        elif color_choice == "2":
            try:
                pos = int(input("  Group position in filename [2]: ").strip() or "2")
                sep = input("  Separator [_]: ").strip() or "_"
                groups, skipped = mta.extract_groups_from_filenames(
                    session["labels"], position=pos, separator=sep)
                if skipped:
                    print(f"  ⚠ {len(skipped)} file(s) skipped")
                if groups:
                    color_values = [groups.get(fn, "(no group)")
                                    for fn in session["labels"]]
                    color_label = f"Group at position {pos}"
                else:
                    print("  ⚠ No groups derived, using dominant topic")
                    dom = np.argmax(doctopic, axis=1)
                    color_values = [f"T{k+1}" for k in dom]
                    color_label = "Dominant topic"
            except Exception as e:
                print(f"  ⚠ Could not derive groups: {e}")

        # Save coords
        coord_cols = [f"axis_{'xyz'[j]}" for j in range(len(axes))]
        df_coords = pd.DataFrame(coords, columns=coord_cols)
        df_coords.insert(0, "document", session["labels"])
        save_dataframe(df_coords, f"axis_projection_{method}_coords",
                       out, {"csv", "json"})

        fig = mta.plot_axis_projection(
            coords=coords,
            labels=session["labels"],
            axis_titles=axis_titles,
            color_values=color_values,
            color_label=color_label,
            endpoint_words=endpoint_words,
            n_top_endpoint_words=5,
            title=f"Axis projection ({method.upper()}, K={n_topics})",
        )
        if fig is not None:
            save_figure(fig, f"axis_projection_{method}",
                        out, {"pdf", "png"})
            plt.close(fig)
        print(f"  ✓ Projection saved (PDF + PNG + CSV/JSON)")

    # ------------------------------------------------------------------
    # STATISTICS
    # ------------------------------------------------------------------
    if do_statistics:
        print("\n  ── Statistics ──")
        try:
            pos = int(input("  Group position in filename "
                            "(for ANOVA factor) [2]: ").strip() or "2")
            sep = input("  Separator [_]: ").strip() or "_"
            min_size = int(input("  Minimum group size [3]: ").strip() or "3")
        except ValueError:
            pos, sep, min_size = 2, "_", 3

        metadata = {}
        groups = {}
        try:
            groups, skipped = mta.extract_groups_from_filenames(
                session["labels"], position=pos, separator=sep,
            )
            if skipped:
                print(f"  ⚠ {len(skipped)} file(s) skipped: "
                      f"{', '.join(skipped[:3])}"
                      + (f" (and {len(skipped)-3} more)"
                         if len(skipped) > 3 else ""))
            if groups:
                metadata[f"group_pos{pos}"] = groups
        except Exception as e:
            print(f"  ⚠ Could not extract groups: {e}")
            groups = {}

        # Enriched export
        df_export = mta.build_axis_export_dataframe(
            labels=session["labels"], coords=coords,
            axis_titles=axis_titles,
            doctopic=doctopic, topicwords=topicwords, vocab=vocab,
            metadata=metadata,
        )
        save_dataframe(df_export, f"axis_export_{method}",
                       out, {"csv", "json"})
        print(f"  ✓ Enriched export: {df_export.shape[0]} rows × "
              f"{df_export.shape[1]} cols (CSV + JSON)")

        if not groups:
            print("  → ANOVA skipped (no groups derived).")
        else:
            group_aligned = [groups.get(fn, "") for fn in session["labels"]]

            summary_rows, welch_rows, tukey_rows, group_summary_rows = [], [], [], []
            for j, letter in enumerate("XYZ"[:len(axes)]):
                print(f"\n  ANOVA on axis {letter} — {axis_titles[j]}")
                result = mta.axis_anova_one_way(
                    coord_values=coords[:, j],
                    group_labels=group_aligned,
                    min_group_size=min_size,
                )
                if "error" in result:
                    print(f"    ⚠ {result['error']}")
                    continue
                clas = result["classical_anova"]
                welch = result["welch_anova"]
                print(f"    {result['n_groups_used']} groups, "
                      f"{len(result['dropped_groups'])} dropped")
                print(f"    Classical F = {clas['F']:.3f}  "
                      f"p = {clas['p_value']:.4g}  "
                      f"η² = {clas['eta_squared']:.3f}")
                print(f"    Welch F     = {welch['F']:.3f}  "
                      f"p = {welch['p_value']:.4g}")
                summary_rows.append({
                    "axis": letter, "axis_title": axis_titles[j],
                    "n_groups": result["n_groups_used"],
                    "n_dropped": len(result["dropped_groups"]),
                    "F_classical": clas["F"], "df_num_classical": clas["df_num"],
                    "df_den_classical": clas["df_den"],
                    "p_classical": clas["p_value"],
                    "eta_squared": clas["eta_squared"],
                    "F_welch": welch["F"], "df_num_welch": welch["df_num"],
                    "df_den_welch": welch["df_den"], "p_welch": welch["p_value"],
                })
                for src, dst in [("welch_pairwise", welch_rows),
                                  ("tukey_pairwise", tukey_rows),
                                  ("group_summary", group_summary_rows)]:
                    df = result[src].copy()
                    if not df.empty:
                        df.insert(0, "axis", letter)
                        dst.append(df)

            if summary_rows:
                save_dataframe(pd.DataFrame(summary_rows),
                               f"axis_anova_summary_{method}",
                               out, {"csv", "json"})
            if welch_rows:
                save_dataframe(pd.concat(welch_rows, ignore_index=True),
                               f"axis_anova_welch_pairwise_{method}",
                               out, {"csv", "json"})
            if tukey_rows:
                save_dataframe(pd.concat(tukey_rows, ignore_index=True),
                               f"axis_anova_tukey_pairwise_{method}",
                               out, {"csv", "json"})
            if group_summary_rows:
                save_dataframe(pd.concat(group_summary_rows, ignore_index=True),
                               f"axis_anova_group_summary_{method}",
                               out, {"csv", "json"})

            fig = mta.plot_axis_anova_boxplots(
                axis_values={"XYZ"[j]: coords[:, j] for j in range(len(axes))},
                group_labels=group_aligned,
                axis_titles={"XYZ"[j]: axis_titles[j] for j in range(len(axes))},
                title=f"ANOVA by group at position {pos} ({method.upper()})",
                min_group_size=min_size,
            )
            if fig is not None:
                save_figure(fig, f"axis_anova_boxplots_{method}",
                            out, {"pdf", "png"})
                plt.close(fig)
            print(f"\n  ✓ Statistics complete "
                  f"({len(summary_rows)} axes analyzed)")



# =============================================================================
# ENTRY POINT
# =============================================================================

def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # No arguments → interactive mode
    if not any([args.corpus, args.stopwords, args.action]):
        return run_interactive()

    # Validate batch arguments
    if not args.corpus or not args.stopwords or not args.action:
        parser.error(
            "Batch mode requires --corpus, --stopwords AND --action. "
            "Run without arguments for interactive mode, or use --help."
        )

    if not os.path.isdir(args.corpus):
        parser.error(f"--corpus is not a directory: {args.corpus}")
    if not os.path.isfile(args.stopwords):
        parser.error(f"--stopwords is not a file: {args.stopwords}")

    if args.action in ("word-weights", "semantic") and not args.words:
        parser.error(f"--words is required for action '{args.action}'")

    return run_batch(args)


if __name__ == "__main__":
    sys.exit(main())
