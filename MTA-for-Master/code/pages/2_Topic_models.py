#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Page 2 — Topic models.

Cross-validation to suggest a number of topics (optional), then NMF and
LDA modelling. Includes the persistent Cophenet scoreboard, topic
distribution charts with selector, and NMF↔LDA comparison heatmap.
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
    download_png_via_matplotlib,
    paired_color_range,
    get_chart_language,
)

# Page config is handled centrally by streamlit_app.py + st.navigation.
init_session_state()
_LANG = get_chart_language()
_LBL = mta.get_labels(_LANG)

page_header("📊 Topic models",
            "Cross-validation, NMF and LDA modelling.")

# Hard gate: this page needs corpus + matrices
if not require_matrices():
    st.stop()


# =============================================================================
# 2.1 — CROSS-VALIDATION
# =============================================================================

st.subheader("2.1 — How many topics? (optional but recommended)")

n_docs = len(st.session_state.doc_labels)
max_allowed = max(3, min(50, n_docs - 1))
default_max = min(8, max_allowed)

st.markdown(
    "MTA tests several values of k (number of topics) and suggests "
    "those most consistent with your corpus structure. "
    "**For large corpora, try a wider range** — they tend to contain "
    "more distinct themes."
)
if n_docs <= 10:
    st.warning(
        f"Your corpus only contains **{n_docs} documents**, so the slider "
        f"is capped at {max_allowed}. With so few texts, the metrics will "
        "be unstable — interpret with caution.",
        icon="⚠️",
    )

max_topics = st.slider(
    "Maximum number of topics to test",
    min_value=3, max_value=max_allowed, value=default_max,
    help=f"Cannot exceed {max_allowed} (corpus has {n_docs} documents). "
         "Note: testing many values is the slowest step — each k requires "
         "training KMeans, NMF and LDA models.",
)

if st.button("📊 Compute cross-validation metrics"):
    progress = st.progress(0.0, text="Starting…")

    def cb(i, total, label):
        progress.progress(i / total, text=f"{label} ({i}/{total})")

    with st.spinner("Cross-validation in progress…"):
        metrics = mta.compute_topic_metrics(
            st.session_state.matrices["tf_matrix"],
            st.session_state.matrices["lda_matrix"],
            st.session_state.matrices["dense_a"],
            max_topics=max_topics,
            progress_callback=cb,
        )
        st.session_state.metrics = metrics
    progress.empty()

if st.session_state.metrics is not None:
    metrics = st.session_state.metrics

    st.markdown("**Suggested numbers of topics**")
    suggestions = metrics["suggestions"]
    cols = st.columns(3)
    for i, (metric_name, values) in enumerate(suggestions.items()):
        with cols[i % 3]:
            if values:
                st.metric(
                    label=metric_name,
                    value=values[0],
                    help=f"Other candidates: "
                         f"{values[1:] if len(values) > 1 else '—'}",
                )
            else:
                st.metric(label=metric_name, value="—",
                          help="No clear candidate")

    st.markdown("**Metrics visualization**")
    st.caption(
        "How to read: a local valley or peak on each curve indicates a "
        "candidate number of topics worth trying. Hover the lines to see "
        "exact values."
    )

    metrics_df = pd.DataFrame({
        "Elbow":             list(metrics["elbow"].values()),
        "Silhouette":        list(metrics["silhouette"].values()),
        "Calinski-Harabasz": list(metrics["calinski"].values()),
        "Davies-Bouldin":    list(metrics["bouldin"].values()),
        "Cophenet NMF":      list(metrics["cophenet_nmf"].values()),
        "Cophenet LDA":      list(metrics["cophenet_lda"].values()),
    }, index=metrics["ks"])
    metrics_df.index.name = "Number of topics"

    chart_layout = [
        ["Elbow",             "Silhouette",       "Cophenet NMF"],
        ["Calinski-Harabasz", "Davies-Bouldin",   "Cophenet LDA"],
    ]
    for row in chart_layout:
        row_cols = st.columns(3)
        for col, metric_name in zip(row_cols, row):
            with col:
                st.markdown(f"**{metric_name}**")
                st.line_chart(
                    metrics_df[[metric_name]],
                    height=200,
                    use_container_width=True,
                )

    with st.expander("Download metrics as PNG / CSV"):
        download_png_via_matplotlib(
            metrics_df, kind="line", name="cv_metrics",
            xlabel=_LBL["number_of_topics"], ylabel=_LBL["score"],
        )
        download_csv(metrics_df, "cv_metrics")

st.divider()


# =============================================================================
# 2.2 — RUN NMF / LDA
# =============================================================================

st.subheader("2.2 — Model the topics")

max_topics_model = max(2, min(50, n_docs))

col_n, col_l = st.columns(2)
with col_n:
    n_topics_nmf = st.number_input(
        "Number of topics — NMF",
        min_value=2, max_value=max_topics_model,
        value=min(5, max_topics_model),
    )
with col_l:
    n_topics_lda = st.number_input(
        "Number of topics — LDA",
        min_value=2, max_value=max_topics_model,
        value=min(5, max_topics_model),
    )

col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    run_nmf_btn = st.button("🟢 Run NMF", type="primary",
                            use_container_width=True)
with col_btn2:
    run_lda_btn = st.button("🔵 Run LDA", type="primary",
                            use_container_width=True)

if run_nmf_btn:
    with st.spinner(f"NMF with {n_topics_nmf} topics…"):
        res = mta.run_nmf(st.session_state.matrices["tf_matrix"], n_topics_nmf)
        st.session_state.nmf_results = res
        st.session_state.nmf_words = mta.top_words_per_topic(
            res["topicwords"], st.session_state.matrices["tf_names"]
        )

if run_lda_btn:
    with st.spinner(f"LDA with {n_topics_lda} topics…"):
        res = mta.run_lda(st.session_state.matrices["lda_matrix"], n_topics_lda)
        st.session_state.lda_results = res
        st.session_state.lda_words = mta.top_words_per_topic(
            res["topicwords"], st.session_state.matrices["lda_names"]
        )

# Persistent Cophenet scoreboard
if st.session_state.nmf_results or st.session_state.lda_results:
    st.markdown("**Model quality — Cophenet correlations**")
    c_nmf, c_lda = st.columns(2)
    with c_nmf:
        if st.session_state.nmf_results is not None:
            st.metric("NMF Cophenet",
                      f"{st.session_state.nmf_results['cophenet']:.3f}",
                      help="Closer to 1.0 = more stable clustering structure.")
        else:
            st.metric("NMF Cophenet", "—", help="Run NMF to compute.")
    with c_lda:
        if st.session_state.lda_results is not None:
            st.metric("LDA Cophenet",
                      f"{st.session_state.lda_results['cophenet']:.3f}",
                      help="Closer to 1.0 = more stable clustering structure.")
        else:
            st.metric("LDA Cophenet", "—", help="Run LDA to compute.")


# =============================================================================
# DISTRIBUTION CHART WITH TOPIC SELECTOR
# =============================================================================

def _render_distribution_with_selector(dist: pd.DataFrame, model_name: str):
    """Stacked bars by default; in focus mode, 'Other topics' as grey segment."""
    all_topics = list(dist.columns)

    st.markdown("**Chart: topic distribution per document**")
    st.caption(
        "💡 Use the selector below to focus on the topics that interest "
        "you most. The 'Other topics' grey segment shows the combined "
        "weight of the topics you didn't select."
    )

    selected = st.multiselect(
        "Topics to display",
        options=all_topics,
        default=all_topics,
        key=f"selector_{model_name}",
    )

    if not selected:
        st.warning("Select at least one topic to display the chart.",
                   icon="⚠️")
        return

    if len(selected) == len(all_topics):
        chart_data = dist
        grey_col = None
        png_name = f"{model_name.lower()}_topic_weights_all"
    else:
        focus = dist[selected].copy()
        other_cols = [c for c in all_topics if c not in selected]
        other_label = _LBL["other_topics"].format(n=len(other_cols))
        chart_data = pd.concat(
            [pd.DataFrame({other_label: dist[other_cols].sum(axis=1)},
                          index=dist.index),
             focus],
            axis=1,
        )
        grey_col = other_label
        png_name = (f"{model_name.lower()}_topic_weights_"
                    f"{len(selected)}of{len(all_topics)}")

    chart_long = chart_data.reset_index().melt(
        id_vars=chart_data.index.name or "index",
        var_name="Topic", value_name="Weight",
    )
    x_field = chart_data.index.name or "index"

    color_domain = list(chart_data.columns)
    color_range = paired_color_range(color_domain, grey_col=grey_col)

    bar_chart = alt.Chart(chart_long).mark_bar().encode(
        x=alt.X(f"{x_field}:N", title=_LBL["documents"],
                sort=list(chart_data.index),
                axis=alt.Axis(labelLimit=120, labelAngle=-90)),
        y=alt.Y("Weight:Q", title=_LBL["weight_of_topics"],
                stack="normalize" if (chart_data.sum(axis=1) - 1).abs().max() > 0.01
                else "zero"),
        color=alt.Color("Topic:N",
                        scale=alt.Scale(domain=color_domain,
                                        range=color_range),
                        sort=color_domain,
                        legend=alt.Legend(title=_LBL["topic"])),
        order=alt.Order("Topic:N", sort="ascending"),
        tooltip=[x_field, alt.Tooltip("Topic:N"),
                 alt.Tooltip("Weight:Q", format=".3f")],
    ).properties(height=400)

    st.altair_chart(bar_chart, use_container_width=True)

    if len(selected) < len(all_topics):
        st.markdown("**Filtered distribution table** (selected topics + Other)")
        st.dataframe(chart_data.style.format("{:.3f}"),
                     use_container_width=True)

    with st.expander(f"High-resolution PNG / CSV export — {model_name}"):
        download_png_via_matplotlib(
            chart_data, kind="bar", name=png_name,
            xlabel=_LBL["documents"], ylabel=_LBL["weight_of_topics"],
            stacked=True, grey_col=grey_col,
        )
        download_csv(chart_data, png_name)


def _render_model_results(model_name: str, model_key: str, words_key: str):
    """Render the result section for one model (NMF or LDA)."""
    results = st.session_state[model_key]
    words = st.session_state[words_key]

    if results is None:
        st.info(f"Run {model_name} above to see results.")
        return

    st.markdown("**20 most representative words per topic**")
    st.dataframe(words, use_container_width=True)
    download_csv(words, f"{model_name.lower()}_top_words")

    dist = mta.topic_distribution_per_doc(
        results["doctopic"], st.session_state.doc_labels,
    )
    st.markdown("**Topic distribution per document — full table**")
    st.dataframe(dist.style.format("{:.3f}"), use_container_width=True)
    download_csv(dist, f"{model_name.lower()}_distribution")

    st.markdown("**Dominant topic per document**")
    dom = mta.dominant_topic_per_doc(dist)
    st.dataframe(dom, use_container_width=True)
    download_csv(dom, f"{model_name.lower()}_dominant_topics")

    _render_distribution_with_selector(dist, model_name)

    if model_name == "NMF":
        st.markdown("**Representative sentences per topic**")
        sentences = mta.best_sentences_per_topic(
            words, st.session_state.corpus_re
        )
        st.dataframe(sentences.head(50), use_container_width=True)
        download_csv(sentences, "nmf_best_sentences")


if st.session_state.nmf_results or st.session_state.lda_results:
    st.markdown("### Results")
    tab_nmf, tab_lda, tab_compare = st.tabs(
        ["NMF", "LDA", "NMF↔LDA comparison"]
    )

    with tab_nmf:
        _render_model_results("NMF", "nmf_results", "nmf_words")

    with tab_lda:
        _render_model_results("LDA", "lda_results", "lda_words")

    with tab_compare:
        if (st.session_state.nmf_results is None
                or st.session_state.lda_results is None):
            st.info(
                "Run NMF **and** LDA with the **same number of topics** "
                "to enable the comparison."
            )
        else:
            nmf_words_df = st.session_state.nmf_words
            lda_words_df = st.session_state.lda_words
            n_nmf = nmf_words_df.shape[1]
            n_lda = lda_words_df.shape[1]

            if n_nmf != n_lda:
                st.warning(
                    "The comparison requires the same number of topics "
                    "for NMF and LDA."
                )
            else:
                top_n = 20
                sim = np.zeros((n_nmf, n_lda))
                for i in range(n_nmf):
                    wi = set(nmf_words_df.iloc[:top_n, i])
                    for j in range(n_lda):
                        wj = set(lda_words_df.iloc[:top_n, j])
                        sim[i, j] = len(wi & wj) / float(top_n)

                sim_long = pd.DataFrame([
                    {"NMF": f"NMF_{i}", "LDA": f"LDA_{j}",
                     "Similarity": sim[i, j]}
                    for i in range(n_nmf) for j in range(n_lda)
                ])

                st.markdown(
                    "**Jaccard similarity between NMF and LDA topics** — "
                    "1.00 = identical topics, 0.00 = entirely distinct."
                )

                heatmap = alt.Chart(sim_long).mark_rect().encode(
                    x=alt.X("LDA:N", title=_LBL["lda_topics"]),
                    y=alt.Y("NMF:N", title=_LBL["nmf_topics"]),
                    color=alt.Color(
                        "Similarity:Q",
                        scale=alt.Scale(scheme="pinkyellowgreen",
                                        domain=[0, 1]),
                        legend=alt.Legend(title=_LBL["jaccard"]),
                    ),
                    tooltip=["NMF", "LDA",
                             alt.Tooltip("Similarity:Q", format=".2f")],
                ).properties(height=400)

                text = alt.Chart(sim_long).mark_text(
                    baseline="middle", fontSize=12, fontWeight="bold",
                    color="black",
                ).encode(
                    x="LDA:N", y="NMF:N",
                    text=alt.Text("Similarity:Q", format=".2f"),
                )

                st.altair_chart(heatmap + text, use_container_width=True)
                st.caption(
                    "💡 To download as PNG: click the **⋯** menu at the "
                    "top-right of the chart, then **Save as PNG**."
                )
                download_csv(
                    sim_long.pivot(index="NMF", columns="LDA",
                                   values="Similarity"),
                    "nmf_lda_comparison",
                )

    st.success(
        "✓ Models computed! Continue with **🔍 Word weights** in the "
        "left sidebar.",
        icon="➡️",
    )
