#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Page 3 — Word weights.

For one or more user-supplied words, show their weight in every topic
(NMF and LDA) and in every document (TF-IDF / Count).
"""

import altair as alt
import streamlit as st

import mta_core as mta
from shared import (
    init_session_state,
    page_header,
    require_model,
    download_csv,
    get_chart_language,
)

# Page config is handled centrally by streamlit_app.py + st.navigation.
init_session_state()
_LANG = get_chart_language()
_LBL = mta.get_labels(_LANG)

page_header("🔍 Word weight analysis",
            "How much weight does a given word carry in each topic and "
            "each document?")

if not require_model():
    st.stop()

st.markdown(
    "Enter one or more words to see **how much weight each word carries "
    "in each topic** and **in which documents it appears most strongly**. "
    "This is useful for tracing specific keywords (e.g. *Maskenpflicht*, "
    "*Impfung*) across your corpus."
)

words_input = st.text_input(
    "Words to analyze (comma- or space-separated)",
    placeholder="e.g. urban, markets, digital",
    help="Search is case-insensitive. Words not found in the vocabulary "
         "will be listed but not analyzed.",
)

if not words_input.strip():
    st.info("Enter at least one word above to start the analysis.")
    st.stop()

# Parse user input
raw_words = [piece.strip()
             for chunk in words_input.split(",")
             for piece in chunk.split()]
words = [w for w in raw_words if w]

if not words:
    st.warning("Please enter at least one valid word.", icon="⚠️")
    st.stop()

st.caption(f"Analyzing **{len(words)} word(s)**: {', '.join(words)}")


# =============================================================================
# Renderer (used for both NMF and LDA)
# =============================================================================

def _render(model_name: str, topicwords, feature_names, term_doc_matrix):
    """Render the full Word-weight UI for one model (NMF or LDA)."""

    # ---- Weights per topic ----
    df_topics, missing = mta.words_weight_per_topic(
        topicwords, feature_names, words,
    )
    if missing:
        st.warning(
            f"Words not in the {model_name} vocabulary: "
            f"**{', '.join(missing)}**",
            icon="🔍",
        )

    if df_topics.empty:
        st.info(
            f"None of the requested words are present in the "
            f"{model_name} vocabulary."
        )
        return

    st.markdown(f"**Weight in {model_name} topics**")
    st.dataframe(df_topics.style.format("{:.4f}"),
                 use_container_width=True)
    download_csv(df_topics, f"{model_name.lower()}_word_weights_topics")

    if df_topics.shape[0] == 1:
        # Single word → bar chart
        word_label = df_topics.index[0]
        bar_long = (
            df_topics.T.reset_index()
            .rename(columns={"index": "Topic", word_label: "Weight"})
        )
        bar = alt.Chart(bar_long).mark_bar(color="#1f78b4").encode(
            x=alt.X("Topic:N", title=_LBL["topic"]),
            y=alt.Y("Weight:Q", title=f"{_LBL['weight']} — '{word_label}'"),
            tooltip=["Topic", alt.Tooltip("Weight:Q", format=".4f")],
        ).properties(height=300)
        st.altair_chart(bar, use_container_width=True)
    else:
        # Multiple words → heatmap words × topics
        long_df = (
            df_topics.reset_index()
            .melt(id_vars="Word", var_name="Topic", value_name="Weight")
        )
        vmax = float(df_topics.values.max())
        heatmap = alt.Chart(long_df).mark_rect().encode(
            x=alt.X("Topic:N", title=_LBL["topic"]),
            y=alt.Y("Word:N", title=_LBL["word"],
                    sort=list(df_topics.index)),
            color=alt.Color(
                "Weight:Q",
                scale=alt.Scale(scheme="pinkyellowgreen",
                                domain=[0, vmax]),
                legend=alt.Legend(title=_LBL["weight"]),
            ),
            tooltip=["Word", "Topic",
                     alt.Tooltip("Weight:Q", format=".4f")],
        ).properties(height=max(200, 30 * df_topics.shape[0] + 80))
        text = alt.Chart(long_df).mark_text(
            baseline="middle", fontSize=11, fontWeight="bold",
            color="black",
        ).encode(
            x="Topic:N", y="Word:N",
            text=alt.Text("Weight:Q", format=".3f"),
        )
        st.altair_chart(heatmap + text, use_container_width=True)
        st.caption(
            "💡 To download as PNG: click the **⋯** menu at the top-right "
            "of the chart, then **Save as PNG**."
        )

    # ---- Weights per document ----
    st.markdown(
        f"**Weight in documents** "
        f"({'TF-IDF' if model_name == 'NMF' else 'Count'})"
    )
    df_docs, _ = mta.words_weight_per_document(
        term_doc_matrix, feature_names,
        st.session_state.doc_labels, words,
    )

    n_docs_total = df_docs.shape[0]
    if n_docs_total > 30:
        top_n = st.slider(
            f"Show top N documents by total weight "
            f"({model_name}, out of {n_docs_total})",
            min_value=10, max_value=min(200, n_docs_total),
            value=min(30, n_docs_total),
            key=f"topn_{model_name}",
        )
        df_docs_view = (
            df_docs.assign(_total=df_docs.sum(axis=1))
            .sort_values("_total", ascending=False)
            .head(top_n)
            .drop(columns="_total")
        )
    else:
        df_docs_view = df_docs

    st.dataframe(df_docs_view.style.format("{:.4f}"),
                 use_container_width=True)
    download_csv(df_docs, f"{model_name.lower()}_word_weights_documents")

    if df_docs.shape[1] == 1:
        # Single word → bar chart over docs
        word_label = df_docs.columns[0]
        bar_long = (
            df_docs_view.reset_index()
            .rename(columns={word_label: "Weight"})
        )
        bar = alt.Chart(bar_long).mark_bar(color="#33a02c").encode(
            x=alt.X("Document:N", title=_LBL["documents"],
                    sort=list(df_docs_view.index),
                    axis=alt.Axis(labelAngle=-90, labelLimit=200)),
            y=alt.Y("Weight:Q", title=f"{_LBL['weight']} — '{word_label}'"),
            tooltip=["Document", alt.Tooltip("Weight:Q", format=".4f")],
        ).properties(height=400)
        st.altair_chart(bar, use_container_width=True)
    else:
        # Multi-word heatmap
        long_df = (
            df_docs_view.reset_index()
            .melt(id_vars="Document", var_name="Word", value_name="Weight")
        )
        vmax_d = float(df_docs_view.values.max()) if not df_docs_view.empty else 1.0
        if vmax_d == 0:
            vmax_d = 1.0
        heatmap = alt.Chart(long_df).mark_rect().encode(
            x=alt.X("Word:N", title=_LBL["word"],
                    sort=list(df_docs_view.columns)),
            y=alt.Y("Document:N", title=_LBL["documents"],
                    sort=list(df_docs_view.index),
                    axis=alt.Axis(labelLimit=200)),
            color=alt.Color(
                "Weight:Q",
                scale=alt.Scale(scheme="pinkyellowgreen",
                                domain=[0, vmax_d]),
                legend=alt.Legend(title=_LBL["weight"]),
            ),
            tooltip=["Document", "Word",
                     alt.Tooltip("Weight:Q", format=".4f")],
        ).properties(height=max(300, 22 * df_docs_view.shape[0] + 60))
        text = alt.Chart(long_df).mark_text(
            baseline="middle", fontSize=10, fontWeight="bold",
            color="black",
        ).encode(
            x="Word:N", y="Document:N",
            text=alt.Text("Weight:Q", format=".3f"),
        )
        st.altair_chart(heatmap + text, use_container_width=True)
        st.caption(
            "💡 To download as PNG: click the **⋯** menu at the top-right "
            "of the chart, then **Save as PNG**."
        )


# =============================================================================
# Render NMF and LDA in separate tabs (only those that have been run)
# =============================================================================

available_tabs = []
if st.session_state.nmf_results is not None:
    available_tabs.append("NMF")
if st.session_state.lda_results is not None:
    available_tabs.append("LDA")
tabs = st.tabs(available_tabs)

tab_idx = 0
if st.session_state.nmf_results is not None:
    with tabs[tab_idx]:
        _render(
            "NMF",
            st.session_state.nmf_results["topicwords"],
            st.session_state.matrices["tf_names"],
            st.session_state.matrices["tf_matrix"],
        )
    tab_idx += 1
if st.session_state.lda_results is not None:
    with tabs[tab_idx]:
        _render(
            "LDA",
            st.session_state.lda_results["topicwords"],
            st.session_state.matrices["lda_names"],
            st.session_state.matrices["lda_matrix"],
        )
