#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Page 7 — Network views.

Bipartite network visualizations of the topic model:
  • Topic ↔ Document : how documents distribute across topics
  • Topic ↔ Top-N words : the lexical content of each topic
  • Combined : topics, documents (circles) and words (squares) together

The graphs use ForceAtlas2 layout (same algorithm as Gephi), a Solarized
palette, and curved edges. Node sizes encode the cumulated edge weight
attached to each topic.
"""

import numpy as np
import streamlit as st

import mta_core as mta
import mta_network as mtanet
from shared import (
    init_session_state,
    page_header,
    require_model,
    download_figure,
    get_chart_language,
)

init_session_state()
_LANG = get_chart_language()

page_header(
    "🕸 Network views",
    "Bipartite network graphs of the topic model: documents, words and "
    "their connections to topics, on one canvas.",
)

if not require_model():
    st.stop()


# =============================================================================
# Method explainer
# =============================================================================

with st.expander("ℹ️ How does this work?"):
    st.markdown(
        """
        A network graph represents the topic model as nodes and links:

        - **Topic nodes** (colored circles, large): one per topic. Their
          size reflects the cumulated weight attached to them (how much
          of the corpus the topic accounts for).
        - **Document nodes** (small circles): one per document. A
          document is linked to a topic whenever its weight on that
          topic is at least *X%* of the document's strongest topic
          weight (the threshold is adjustable).
        - **Word nodes** (small squares): the top-N most representative
          words of each topic. A word may be linked to several topics
          if it ranks high in more than one.
        - **Edges**: their thickness encodes the strength of the
          document↔topic or word↔topic relation.

        The layout uses **ForceAtlas2**, the same algorithm Gephi uses,
        producing organic placements where strongly connected nodes
        cluster together. Colors come from the **Solarized** palette
        for a warm, publication-ready look.
        """
    )


# =============================================================================
# Method choice (NMF / LDA) and parameters
# =============================================================================

st.subheader("7.1 — Settings")

# Which model do we have?
nmf_res = st.session_state.get("nmf_results")
lda_res = st.session_state.get("lda_results")
matrices = st.session_state["matrices"]
labels = st.session_state["doc_labels"]

available = []
if nmf_res is not None:
    available.append("NMF")
if lda_res is not None:
    available.append("LDA")

col1, col2 = st.columns(2)
with col1:
    if len(available) > 1:
        method = st.radio(
            "Topic model to visualize",
            available, horizontal=True,
            help="Both NMF and LDA models are available; pick one.",
        )
    else:
        method = available[0]
        st.markdown(f"**Model:** {method} (only model available)")

with col2:
    emphasize = st.toggle(
        "Emphasize size differences",
        value=False,
        help=(
            "When OFF (default), topic node sizes faithfully reflect "
            "their weight in the data. On a balanced corpus, nodes may "
            "look similar — that's honest. When ON, the smallest and "
            "largest topics are stretched apart, so even small "
            "differences become visible (at the cost of proportionality)."
        ),
    )

# Retrieve doctopic / topicwords / vocab for the chosen model
if method == "NMF":
    doctopic = nmf_res["doctopic"]
    topicwords = nmf_res["topicwords"]
    vocab = list(matrices["tf_names"])
else:
    doctopic = lda_res["doctopic"]
    topicwords = lda_res["topicwords"]
    vocab = list(matrices["lda_names"])

n_topics = topicwords.shape[0]

# Build topic names from top-3 words of each topic
topic_names = []
for k in range(n_topics):
    top_idx = np.argsort(topicwords[k])[::-1][:3]
    topic_names.append(" / ".join([vocab[i] for i in top_idx]))


# =============================================================================
# Tabs — one per graph kind
# =============================================================================

tab_doc, tab_word, tab_comb = st.tabs([
    "📄 Topic ↔ Document",
    "🔤 Topic ↔ Top-words",
    "🌐 Combined",
])


# ----------------- Tab 1: topic ↔ document ----------------------------------

with tab_doc:
    st.markdown(
        "Each document is connected to the topic(s) it weighs strongly on. "
        "Use the slider below to keep only the strongest links."
    )
    min_edge_doc = st.slider(
        "Minimum edge weight "
        "(as % of the document's strongest topic)",
        min_value=5, max_value=80, value=10, step=5,
        key="doc_min_edge",
        help=(
            "Lower values keep more (weaker) links; the graph becomes "
            "dense and harder to read. Higher values keep only the "
            "documents that strongly align with their dominant topic."
        ),
    ) / 100.0

    if st.button("🖼️ Generate topic↔document graph", type="primary",
                 key="gen_doc"):
        with st.spinner("Computing layout with ForceAtlas2…"):
            fig = mtanet.plot_topic_document_network(
                doctopic=doctopic,
                labels=labels,
                topic_names=topic_names,
                min_weight_pct=min_edge_doc,
                title=f"Topics ↔ Documents ({method}, K={n_topics})",
                emphasize_differences=emphasize,
            )
        st.pyplot(fig, use_container_width=True)
        download_figure(fig,
                        name=f"network_{method.lower()}_topic_document")
        st.session_state["_last_fig_doc"] = fig


# ----------------- Tab 2: topic ↔ words --------------------------------------

with tab_word:
    st.markdown(
        "Each topic is connected to its top-N most representative words. "
        "A word linked to several topics has nuance between them."
    )
    top_n_word = st.slider(
        "Top-N words per topic",
        min_value=10, max_value=80, value=50, step=5,
        key="word_top_n",
        help=(
            "How many top-weighted words to display for each topic. "
            "Larger values reveal long tails but may clutter the graph."
        ),
    )

    if st.button("🖼️ Generate topic↔word graph", type="primary",
                 key="gen_word"):
        with st.spinner("Computing layout with ForceAtlas2…"):
            fig = mtanet.plot_topic_word_network(
                topicwords=topicwords,
                vocab=vocab,
                topic_names=topic_names,
                top_n=top_n_word,
                title=f"Topics ↔ Top-{top_n_word} words "
                      f"({method}, K={n_topics})",
                emphasize_differences=emphasize,
            )
        st.pyplot(fig, use_container_width=True)
        download_figure(fig,
                        name=f"network_{method.lower()}_topic_words")
        st.session_state["_last_fig_word"] = fig


# ----------------- Tab 3: combined -------------------------------------------

with tab_comb:
    st.markdown(
        "All three node kinds in one view: topics, documents (circles) "
        "and top-words (squares). For readability, defaults are stricter "
        "than in the previous two tabs."
    )
    col_a, col_b = st.columns(2)
    with col_a:
        top_n_comb = st.slider(
            "Top-N words per topic", min_value=5, max_value=40,
            value=25, step=5, key="comb_top_n",
        )
    with col_b:
        min_edge_comb = st.slider(
            "Minimum doc→topic edge (%)",
            min_value=10, max_value=80, value=20, step=5,
            key="comb_min_edge",
        ) / 100.0

    if st.button("🖼️ Generate combined graph", type="primary",
                 key="gen_comb"):
        with st.spinner("Computing layout with ForceAtlas2…"):
            fig = mtanet.plot_combined_network(
                doctopic=doctopic,
                topicwords=topicwords,
                labels=labels,
                vocab=vocab,
                topic_names=topic_names,
                top_n_words=top_n_comb,
                min_doc_weight_pct=min_edge_comb,
                title=f"Topics + Documents + Top-words "
                      f"({method}, K={n_topics})",
                emphasize_differences=emphasize,
            )
        st.pyplot(fig, use_container_width=True)
        download_figure(fig,
                        name=f"network_{method.lower()}_combined")
        st.session_state["_last_fig_comb"] = fig


# =============================================================================
# Footer note
# =============================================================================

st.divider()
st.caption(
    "💡 **Tip.** The graphs are deterministic (same seed every run). "
    "Save them as PDF for publication: vector graphics scale to any "
    "size without quality loss."
)
