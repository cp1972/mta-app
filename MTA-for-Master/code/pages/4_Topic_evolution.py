#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Page 4 — Topic evolution through texts.

For each model (NMF, LDA) that has been computed, show:
1. A rolling mean of topic weights along the alphabetically-sorted
   sequence of documents (the natural ordering for files named like
   'YYYY-MM-DD_article.txt').
2. Optionally, a yearly aggregation if filenames start with 'YYYY'.

Mirrors menu entry 2 of the original MTA.py, with two improvements:
- The rolling-mean chart is always shown (no 40-document cap).
- Topic selector for focus mode, consistent with page 2.
"""

import altair as alt
import pandas as pd
import streamlit as st

import mta_core as mta
from shared import (
    init_session_state,
    page_header,
    require_model,
    download_csv,
    download_png_via_matplotlib,
    paired_color_range,
    get_chart_language,
)

init_session_state()
_LANG = get_chart_language()
_LBL = mta.get_labels(_LANG)

page_header(
    "📈 Topic evolution through texts",
    "Rolling mean of topic weights across documents, plus optional "
    "yearly aggregation if filenames start with a date (YYYY-...).",
)

if not require_model():
    st.stop()


# =============================================================================
# Helper: render one model's evolution section (NMF or LDA)
# =============================================================================

def _render_evolution(model_name: str, doctopic):
    """Full UI for one model's evolution: window slider, RM chart, yearly opt."""

    st.markdown(
        f"### {model_name} — Rolling mean of topic weights"
    )

    n_docs = len(st.session_state.doc_labels)
    max_window = max(2, n_docs)
    default_window = min(2, max_window)

    window = st.slider(
        "Window size (number of consecutive documents averaged)",
        min_value=1, max_value=max_window, value=default_window,
        key=f"window_{model_name}",
        help="Window = 1 means no smoothing (raw distribution). Larger "
             "windows produce a smoother evolution curve. The first rows "
             "use a smaller window (it grows from 1 up to your choice).",
    )

    rm = mta.rolling_mean_distribution(
        doctopic, st.session_state.doc_labels, window=window,
    )

    # --- Table ---
    st.markdown("**Rolling-mean table** (sorted documents × topics)")
    st.dataframe(rm.style.format("{:.3f}"), use_container_width=True)
    download_csv(rm, f"{model_name.lower()}_rolling_mean")

    # --- Chart with topic selector (consistent with page 2) ---
    st.markdown("**Chart: rolling mean of topic weights per document**")
    if n_docs > 40:
        st.caption(
            f"💡 You have **{n_docs} documents** — the chart will be dense. "
            "Use the topic selector below to focus on specific topics. "
            "Use the mouse wheel inside the chart to zoom horizontally."
        )

    all_topics = list(rm.columns)
    selected = st.multiselect(
        "Topics to display",
        options=all_topics,
        default=all_topics,
        key=f"selector_evo_{model_name}",
        help="Pick the topics to focus on. The 'Other topics' grey segment "
             "shows the combined weight of unselected topics.",
    )

    if not selected:
        st.warning("Select at least one topic to display the chart.",
                   icon="⚠️")
        return

    # Build chart data (stack 'Other' first at the bottom, focus on top)
    if len(selected) == len(all_topics):
        chart_data = rm
        grey_col = None
        png_name = f"{model_name.lower()}_rolling_mean_all"
    else:
        focus = rm[selected].copy()
        other_cols = [c for c in all_topics if c not in selected]
        other_label = _LBL["other_topics"].format(n=len(other_cols))
        chart_data = pd.concat(
            [pd.DataFrame({other_label: rm[other_cols].sum(axis=1)},
                          index=rm.index),
             focus],
            axis=1,
        )
        grey_col = other_label
        png_name = (f"{model_name.lower()}_rolling_mean_"
                    f"{len(selected)}of{len(all_topics)}")

    chart_long = chart_data.reset_index().melt(
        id_vars="Document", var_name="Topic", value_name="Weight",
    )

    color_domain = list(chart_data.columns)
    color_range = paired_color_range(color_domain, grey_col=grey_col)

    bar_chart = alt.Chart(chart_long).mark_bar().encode(
        x=alt.X("Document:N", title=_LBL["documents_sorted"],
                sort=list(chart_data.index),
                axis=alt.Axis(labelLimit=120, labelAngle=-90)),
        y=alt.Y("Weight:Q", title=_LBL["weight_rm"],
                stack="zero"),
        color=alt.Color("Topic:N",
                        scale=alt.Scale(domain=color_domain,
                                        range=color_range),
                        sort=color_domain,
                        legend=alt.Legend(title=_LBL["topic"])),
        order=alt.Order("Topic:N", sort="ascending"),
        tooltip=["Document", alt.Tooltip("Topic:N"),
                 alt.Tooltip("Weight:Q", format=".3f")],
    ).properties(height=420)

    st.altair_chart(bar_chart, use_container_width=True)

    with st.expander(f"High-resolution PNG / CSV export — {model_name}"):
        download_png_via_matplotlib(
            chart_data, kind="bar", name=png_name,
            xlabel=_LBL["documents_sorted"],
            ylabel=_LBL["weight_rm"],
            stacked=True, grey_col=grey_col,
        )
        download_csv(chart_data, png_name)

    # -----------------------------------------------------------------
    # Optional: yearly aggregation
    # -----------------------------------------------------------------
    st.markdown(f"### {model_name} — Yearly aggregation (optional)")
    st.markdown(
        "If your filenames start with a 4-digit year (e.g. "
        "`2020-03-15_article.txt`), MTA can group the rolling mean by "
        "year and show one curve per topic across years."
    )

    do_yearly = st.checkbox(
        "Group by year (filenames must start with YYYY)",
        key=f"yearly_{model_name}",
    )

    if not do_yearly:
        return

    yearly_df, bad_labels = mta.yearly_topic_evolution(rm)

    if bad_labels:
        st.warning(
            f"**{len(bad_labels)} filename(s)** do not start with a "
            "4-digit year and were ignored. First few examples: "
            f"`{', '.join(bad_labels[:5])}`"
            + (" …" if len(bad_labels) > 5 else ""),
            icon="🔍",
        )

    if yearly_df.empty:
        st.error(
            "No filename starts with a 4-digit year — cannot perform "
            "yearly aggregation. Check your filenames or uncheck the box "
            "above.",
            icon="🚫",
        )
        return

    n_years = len(yearly_df)
    st.success(
        f"✓ Aggregated **{n_years} year(s)**: "
        f"{', '.join(yearly_df.index.tolist())}"
    )

    st.markdown("**Yearly mean table** (years × topics)")
    st.dataframe(yearly_df.style.format("{:.3f}"),
                 use_container_width=True)
    download_csv(yearly_df,
                 f"{model_name.lower()}_yearly_evolution")

    # --- Interactive line chart with Altair ---
    st.markdown("**Yearly evolution of topics**")
    if n_years < 2:
        st.info(
            "Only one year detected — a line chart needs at least two "
            "points. The table above shows the single-year averages."
        )
        return

    yearly_long = yearly_df.reset_index().melt(
        id_vars="Year", var_name="Topic", value_name="Weight",
    )

    line_chart = alt.Chart(yearly_long).mark_line(
        point=alt.OverlayMarkDef(size=60),
        strokeWidth=2.5,
    ).encode(
        x=alt.X("Year:O", title=_LBL["year"],
                axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Weight:Q", title=_LBL["yearly_weight"]),
        color=alt.Color("Topic:N",
                        scale=alt.Scale(scheme="category10"),
                        legend=alt.Legend(title=_LBL["topic"])),
        tooltip=["Year", "Topic",
                 alt.Tooltip("Weight:Q", format=".3f")],
    ).properties(height=380)

    st.altair_chart(line_chart, use_container_width=True)
    st.caption(
        "💡 To download as PNG: click the **⋯** menu at the top-right "
        "of the chart, then **Save as PNG**. "
        "For a high-resolution version, use the button below."
    )

    with st.expander(f"High-resolution PNG / CSV export — yearly {model_name}"):
        download_png_via_matplotlib(
            yearly_df, kind="line",
            name=f"{model_name.lower()}_yearly_evolution",
            xlabel=_LBL["year"], ylabel=_LBL["yearly_weight"],
        )
        download_csv(yearly_df,
                     f"{model_name.lower()}_yearly_evolution")


# =============================================================================
# Render NMF and LDA in separate tabs
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
        _render_evolution(
            "NMF", st.session_state.nmf_results["doctopic"],
        )
    tab_idx += 1
if st.session_state.lda_results is not None:
    with tabs[tab_idx]:
        _render_evolution(
            "LDA", st.session_state.lda_results["doctopic"],
        )
