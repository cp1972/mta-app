#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Page 5 — Semantic context (similar words, 2D clouds, sub-corpus).

Replicates the functionality of menu entry 4 of the original MTA.py
(Word2Vec / TSNE clouds + best-files selection), with two key changes:

1. The user can pick between two embedding methods:
   - Co-Occurrence + PCA (no extra dependency, transparent for teaching)
   - Word2Vec via gensim (closer to the original MTA.py)

2. The 2D layout uses PCA (deterministic, fast, already available)
   instead of TSNE (slow, stochastic, perplexity tuning).
"""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import mta_core as mta
from shared import (
    init_session_state,
    page_header,
    require_matrices,
    download_csv,
    get_chart_language,
)

init_session_state()
_LANG = get_chart_language()
_LBL = mta.get_labels(_LANG)

page_header(
    "🧠 Semantic context",
    "Find words that surround a chosen word in the corpus, see their "
    "semantic cloud in 2D, and select the documents where these words "
    "are most present.",
)

if not require_matrices():
    st.stop()


# =============================================================================
# Method explainer (collapsed by default — keep page clean)
# =============================================================================

with st.expander("ℹ️ How does this work? (method explainer)"):
    st.markdown(
        """
        MTA computes a **word embedding** — a numerical vector for each
        word — from your corpus, then uses these vectors to find words
        with similar meanings (those with similar vectors).

        Two methods are available:

        **1. Co-Occurrence + PCA (default, transparent)**
        - For each pair of words, count how often they appear close to
          each other (within a window of ±5 words by default) across the
          corpus.
        - Apply a log smoothing so that very frequent words don't dominate.
        - Reduce this word-word matrix to 100 dimensions via Truncated SVD.
        - Result: each word gets a 100-dim vector summarising its
          linguistic neighbourhood in *your* corpus.

        **2. Word2Vec (closer to original MTA.py)**
        - Trains a shallow neural network on your corpus to predict the
          context of each word.
        - More sensitive to hyperparameters (window size, dimensions,
          epochs); needs larger corpora to give stable results.

        Both methods produce vectors which we compare with **cosine
        similarity** to find similar words. For visualisation, we use
        **PCA** to project these vectors into 2D — a deterministic,
        fast method (TSNE would be slower and stochastic).
        """
    )


# =============================================================================
# Method selector
# =============================================================================

st.subheader("Embedding method")

method = st.radio(
    "Choose how MTA learns word vectors from your corpus:",
    options=["Co-Occurrence + PCA (recommended)", "Word2Vec (gensim)"],
    horizontal=True,
    help="Co-Occurrence is fast, deterministic, and transparent. "
         "Word2Vec is closer to the original MTA.py but slower and "
         "needs a larger corpus to be stable.",
)
use_w2v = method.startswith("Word2Vec")

# Method-specific parameters
col_a, col_b, col_c = st.columns(3)
with col_a:
    window = st.slider(
        "Context window size",
        min_value=2, max_value=20,
        value=5,
        help="Number of words on each side considered as 'neighbours' "
             "(both methods).",
    )
with col_b:
    min_count = st.slider(
        "Minimum word frequency",
        min_value=1, max_value=20,
        value=2 if not use_w2v else 5,
        help="Words appearing in fewer documents (Co-Occurrence) or fewer "
             "times total (Word2Vec) are dropped.",
    )
with col_c:
    n_dims = st.slider(
        "Embedding dimensions",
        min_value=20, max_value=300,
        value=100,
        help="Higher = finer-grained vectors, but needs more data. "
             "Default 100 works well for most corpora.",
    )


# =============================================================================
# Train / load embeddings — cached on parameters
# =============================================================================

@st.cache_data(show_spinner=False)
def _compute_cooccurrence(corpus_wo_tuple, window, min_count, n_dims):
    """Cache wrapper: takes a tuple so it's hashable."""
    return mta.build_cooccurrence_embeddings(
        list(corpus_wo_tuple), window=window,
        min_count=min_count, n_dims=n_dims,
    )


@st.cache_data(show_spinner=False)
def _compute_word2vec(corpus_wo_tuple, window, min_count, n_dims):
    """Cache wrapper for Word2Vec. Returns (embeddings dict, vocab list)."""
    # Lazy import — only loaded if user picks Word2Vec
    try:
        from gensim.models import Word2Vec
    except ImportError:
        return None, "gensim is not installed"

    tokenized = mta.tokenize_for_cooccurrence(list(corpus_wo_tuple))
    if not tokenized:
        return {}, []
    model = Word2Vec(
        tokenized,
        vector_size=n_dims, window=window,
        min_count=min_count, workers=2,
        sample=1e-5, epochs=5,
    )
    vocab = sorted(model.wv.index_to_key)
    embeddings = {w: model.wv[w].copy() for w in vocab}
    return embeddings, vocab


corpus_wo_tuple = tuple(st.session_state.corpus_wo)

if use_w2v:
    with st.spinner("Training Word2Vec model on your corpus…"):
        result = _compute_word2vec(corpus_wo_tuple, window, min_count, n_dims)
    if result[0] is None and result[1] == "gensim is not installed":
        st.error(
            "**gensim** is not installed in this environment. "
            "Either install it (`pip install gensim`) or switch to "
            "**Co-Occurrence + PCA** above (no extra dependency needed).",
            icon="📦",
        )
        st.stop()
    embeddings, vocab = result
else:
    with st.spinner("Computing co-occurrence embeddings…"):
        embeddings, vocab = _compute_cooccurrence(
            corpus_wo_tuple, window, min_count, n_dims,
        )

if not embeddings:
    st.error(
        f"Could not compute embeddings — your corpus may be too small or "
        f"`min_count` too strict. Vocabulary size after filtering: "
        f"**{len(vocab)}**. Try lowering `min_count` or `embedding dimensions`.",
        icon="🚫",
    )
    st.stop()

st.success(
    f"✓ Embeddings ready: **{len(embeddings):,} words** × "
    f"**{n_dims} dimensions** ({'Word2Vec' if use_w2v else 'Co-Occurrence + PCA'})."
)

st.divider()


# =============================================================================
# Section 1 — Similar words
# =============================================================================

st.subheader("1. Similar words")

st.markdown(
    "Enter one or more words to find the words that surround them most "
    "often in your corpus. Words must appear in the vocabulary above."
)

words_input = st.text_input(
    "Words to analyze (comma- or space-separated)",
    placeholder="e.g. urban, markets, digital",
    help="Search is case-insensitive. Words not in the vocabulary will "
         "be listed but skipped.",
)

if not words_input.strip():
    st.info("Enter at least one word above to start.")
    st.stop()

raw_words = [p.strip().lower() for chunk in words_input.split(",")
             for p in chunk.split()]
words = [w for w in raw_words if w]

# Resolve found vs missing
found_words = [w for w in words if w in embeddings]
missing_words = [w for w in words if w not in embeddings]

if missing_words:
    st.warning(
        f"Words not in the vocabulary: **{', '.join(missing_words)}**",
        icon="🔍",
    )

if not found_words:
    st.error(
        "None of the requested words is in the vocabulary. Try other "
        "words, or lower the `Minimum word frequency` above.",
        icon="🚫",
    )
    st.stop()

st.caption(f"Analyzing **{len(found_words)} word(s)**: "
           f"{', '.join(found_words)}")

# Compute similar words for each
n_neighbours = st.slider(
    "Number of similar words to show per query word",
    min_value=5, max_value=30, value=10,
)

similar_map = {}
for w in found_words:
    sims = mta.most_similar_words(embeddings, w, topn=n_neighbours)
    similar_map[w] = [s[0] for s in sims]

# Build a long-format DataFrame for display
rows = []
for w in found_words:
    sims = mta.most_similar_words(embeddings, w, topn=n_neighbours)
    for rank, (sw, score) in enumerate(sims, start=1):
        rows.append({"Query word": w, "Rank": rank,
                     "Similar word": sw, "Similarity": score})
sims_df = pd.DataFrame(rows)

st.markdown("**Top similar words**")
st.dataframe(sims_df.style.format({"Similarity": "{:.3f}"}),
             use_container_width=True, hide_index=True)
download_csv(sims_df, "similar_words")

# Single word → bar chart; multiple words → grouped chart
if len(found_words) == 1:
    qw = found_words[0]
    sub = sims_df[sims_df["Query word"] == qw]
    bar = alt.Chart(sub).mark_bar(color="#1f78b4").encode(
        x=alt.X("Similar word:N",
                sort=alt.SortField("Rank", order="ascending"),
                title=_LBL["similar_word"]),
        y=alt.Y("Similarity:Q",
                title=f"{_LBL['similarity']} — '{qw}'"),
        tooltip=["Similar word",
                 alt.Tooltip("Similarity:Q", format=".3f")],
    ).properties(height=300)
    st.altair_chart(bar, use_container_width=True)
else:
    # Use .facet() pattern (stable across Altair versions) rather than
    # passing column= inside encode().
    base = alt.Chart(sims_df).mark_bar().encode(
        x=alt.X("Rank:O", title=_LBL["rank"]),
        y=alt.Y("Similarity:Q", title=_LBL["similarity"]),
        color=alt.Color("Query word:N",
                        scale=alt.Scale(scheme="category10")),
        tooltip=["Query word", "Similar word",
                 alt.Tooltip("Similarity:Q", format=".3f")],
    ).properties(height=240, width=140)
    grouped = base.facet(column=alt.Column("Query word:N", title=None))
    st.altair_chart(grouped, use_container_width=False)

st.divider()


# =============================================================================
# Section 2 — Semantic cloud (2D PCA visualisation)
# =============================================================================

st.subheader("2. Semantic cloud (2D)")

st.markdown(
    "MTA places each query word and its neighbours on a 2D plot using PCA. "
    "Words close together in the cloud are semantically similar in your "
    "corpus. With multiple query words, each cluster has its own colour — "
    "you can see whether their semantic neighbourhoods overlap or stay "
    "separate."
)

cloud_neighbours = st.slider(
    "Neighbours per query word in the cloud",
    min_value=10, max_value=80, value=50 if len(found_words) > 1 else 15,
    help="With multiple words, more neighbours = denser clusters but "
         "harder to read individual labels.",
)

with st.spinner("Projecting words into 2D with PCA…"):
    df_cloud = mta.pca_project_word_clusters(
        embeddings, found_words,
        neighbours_per_seed=cloud_neighbours,
    )

if df_cloud.empty:
    st.warning("Could not build the 2D cloud — try different query words.",
               icon="⚠️")
else:
    # Two layers: small dots for neighbours, larger dots + bold labels for seeds.
    base = alt.Chart(df_cloud)

    # Neighbour points (small, semi-transparent)
    neighbours = base.transform_filter(
        alt.datum.IsSeed == False
    ).mark_circle(size=60, opacity=0.55).encode(
        x=alt.X("x:Q", title=None,
                axis=alt.Axis(labels=False, ticks=False)),
        y=alt.Y("y:Q", title=None,
                axis=alt.Axis(labels=False, ticks=False)),
        color=alt.Color("Cluster:N",
                        scale=alt.Scale(scheme="category10"),
                        legend=alt.Legend(title=_LBL["query_word"])),
        tooltip=["Word", "Cluster"],
    )

    neighbour_labels = base.transform_filter(
        alt.datum.IsSeed == False
    ).mark_text(
        align="left", baseline="middle", dx=6, fontSize=10, opacity=0.85,
    ).encode(
        x="x:Q", y="y:Q",
        text="Word:N",
        color=alt.Color("Cluster:N",
                        scale=alt.Scale(scheme="category10"),
                        legend=None),
    )

    # Seed points (larger, bold black labels)
    seeds = base.transform_filter(
        alt.datum.IsSeed == True
    ).mark_point(size=200, filled=True, color="black",
                 shape="diamond").encode(
        x="x:Q", y="y:Q",
        tooltip=["Word", "Cluster"],
    )

    seed_labels = base.transform_filter(
        alt.datum.IsSeed == True
    ).mark_text(
        align="left", baseline="middle", dx=10,
        fontSize=14, fontWeight="bold", color="black",
    ).encode(
        x="x:Q", y="y:Q",
        text="Word:N",
    )

    cloud_chart = (neighbours + neighbour_labels + seeds + seed_labels)\
        .properties(height=500)\
        .interactive()

    st.altair_chart(cloud_chart, use_container_width=True)
    st.caption(
        "💡 Drag to pan, scroll to zoom. To download as PNG: click the "
        "**⋯** menu at the top-right of the chart, then **Save as PNG**."
    )

    with st.expander("Raw coordinates (downloadable)"):
        st.dataframe(df_cloud.round(3), use_container_width=True,
                     hide_index=True)
        download_csv(df_cloud, "semantic_cloud_coordinates")

st.divider()


# =============================================================================
# Section 3 — Best documents for the chosen words (sub-corpus selection)
# =============================================================================

st.subheader("3. Best documents for the chosen words")

st.markdown(
    "MTA scores each document by the **mean TF-IDF weight** of your query "
    "words *and* their neighbours. Documents above a threshold can be "
    "saved as a list — useful to build a sub-corpus for a focused analysis."
)

# Compute the means once (no threshold) to show stats
df_all, full_means = mta.best_documents_for_words(
    st.session_state.matrices["tf_matrix"],
    st.session_state.matrices["tf_names"],
    st.session_state.doc_labels,
    found_words,
    similar_map,
    min_mean_weight=None,
)

if df_all.empty or full_means.empty:
    st.warning(
        "None of the query words (or their neighbours) appear in the "
        "TF-IDF vocabulary. This can happen if `min_df` in step 2 was "
        "more restrictive than `min_count` here.",
        icon="⚠️",
    )
else:
    stats = full_means.describe()

    st.markdown("**Statistics of the mean weight across all documents**")
    stats_df = pd.DataFrame({"Value": stats}).round(4)
    st.dataframe(stats_df, use_container_width=False)

    # Slider for the threshold, with reasonable default
    default_thr = float(stats["50%"])  # median
    max_thr = float(stats["max"])
    min_thr = float(stats["min"])
    if max_thr <= min_thr:
        st.info("All documents have the same mean weight — no filtering "
                "possible.")
    else:
        threshold = st.slider(
            "Keep documents whose mean weight exceeds this value:",
            min_value=float(min_thr),
            max_value=float(max_thr),
            value=float(default_thr),
            step=float((max_thr - min_thr) / 100),
            format="%.4f",
            help="A value around the median keeps about half the documents. "
                 "Higher = fewer but more relevant documents.",
        )

        df_selected, _ = mta.best_documents_for_words(
            st.session_state.matrices["tf_matrix"],
            st.session_state.matrices["tf_names"],
            st.session_state.doc_labels,
            found_words,
            similar_map,
            min_mean_weight=threshold,
        )

        st.success(
            f"**{len(df_selected):,} of {len(full_means):,} documents** "
            f"have a mean weight > {threshold:.4f}."
        )

        if len(df_selected) > 0:
            # Display: mean column + word-level columns
            display_cols = ["mean"] + [c for c in df_selected.columns
                                       if c != "mean"]
            st.dataframe(
                df_selected[display_cols].style.format("{:.4f}"),
                use_container_width=True,
            )

            # Two CSV downloads: full table OR just the filenames
            col_csv1, col_csv2 = st.columns(2)
            with col_csv1:
                download_csv(df_selected, "best_documents_full")
            with col_csv2:
                names_only = pd.DataFrame({"Document": df_selected.index})
                download_csv(names_only, "best_documents_names_only")
            st.caption(
                "💡 The 'names_only' CSV is the list of selected filenames — "
                "use it to copy that subset of files into a separate folder "
                "and run MTA again on the sub-corpus."
            )
