#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Page 6 — Group comparison.

For each topic, test whether documents differ significantly between
groups (e.g. interviews labelled F vs M, or low vs high income).
Two tests are computed in parallel (Welch's t-test + Mann-Whitney U),
with Benjamini-Hochberg correction for multiple comparisons.

Groups can be defined either by position in the filename (e.g. 'F' in
'interview_F_25-34_001.txt' is at position 2) or by an uploaded CSV
mapping filenames to group codes.
"""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import mta_core as mta
from shared import (
    init_session_state,
    page_header,
    require_model,
    download_csv,
    get_chart_language,
)

init_session_state()
_LANG = get_chart_language()
_LBL = mta.get_labels(_LANG)

page_header(
    "⚖️ Group comparison",
    "Test whether documents differ significantly between groups on "
    "each topic.",
)

if not require_model():
    st.stop()


# =============================================================================
# Method explainer
# =============================================================================

with st.expander("ℹ️ How does this work? (method explainer)"):
    st.markdown(
        """
        Once you have a topic model, you can ask whether different
        **groups** of documents (e.g. interviews with women vs men, or
        with different age groups) score differently on each topic.

        **Step 1 — Define groups.** Either:
        - Encode the group in the filename: `interview_F_25-34_001.txt`
          places `F` at position 2 (counting from 1, splitting on `_`).
        - Or upload a CSV with a `filename` column and one or more
          group columns: `filename,gender,age_band,income`.

        **Step 2 — Statistical tests.** For each topic and each pair of
        groups, MTA computes:
        - **Welch's t-test** — classical, sensitive to mean differences,
          assumes roughly normal data.
        - **Mann-Whitney U** — non-parametric, more robust, compares
          rank distributions.

        We show both because topic weights are bounded (0–1) and often
        skewed, so the non-parametric test is generally more honest.

        **Step 3 — Correction for multiple testing.** If you test 5
        topics, the chance of a false positive grows. We apply
        **Benjamini-Hochberg (FDR)** correction, which controls the
        proportion of false discoveries among the "significant" results.

        **A small sample warning (n < 30 per group)** flags tests that
        may be unreliable. We don't block them — your students can
        learn to judge fragility.
        """
    )


# =============================================================================
# Step 1 — Define groups
# =============================================================================

st.subheader("1. Define groups")

method = st.radio(
    "How are groups defined?",
    options=["From filenames (simple)", "From uploaded CSV (advanced)"],
    horizontal=True,
)

groups: dict = {}
grouping_name = ""

if method.startswith("From filenames"):
    col1, col2 = st.columns(2)
    with col1:
        position = st.number_input(
            "Position of the group code in the filename",
            min_value=1, max_value=10, value=2,
            help="Position 1 = first chunk before the first separator. "
                 "Example: `interview_F_25-34_001.txt` → position 2 is `F`.",
        )
    with col2:
        separator = st.text_input(
            "Separator", value="_",
            help="Character that separates the chunks. Usually `_` or `-`.",
        )

    groups, skipped = mta.extract_groups_from_filenames(
        st.session_state.doc_labels, position=int(position),
        separator=separator,
    )
    grouping_name = f"position_{position}"

    if skipped:
        st.warning(
            f"**{len(skipped)} file(s)** don't have a part at position "
            f"{position}: {', '.join(skipped[:5])}"
            + (f" (and {len(skipped)-5} more…)" if len(skipped) > 5 else ""),
            icon="⚠️",
        )

    if groups:
        # Show a preview of the extraction
        preview = pd.DataFrame({
            "Filename": list(groups.keys())[:5],
            "Extracted group": list(groups.values())[:5],
        })
        st.caption(f"Preview — first 5 files mapped:")
        st.dataframe(preview, hide_index=True, use_container_width=True)

else:
    csv_file = st.file_uploader(
        "Upload a CSV with a `filename` column + group columns",
        type=["csv"],
        help="Example columns: `filename,gender,age_band,income`. "
             "Each non-filename column becomes a separate grouping you "
             "can analyse.",
    )

    if csv_file is None:
        st.info("Please upload a CSV to continue.")
        st.stop()

    # Save to temp path so mta_core can read it
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="wb",
                                     delete=False) as f:
        f.write(csv_file.read())
        tmp_path = f.name

    try:
        groupings, skipped = mta.extract_groups_from_csv(
            tmp_path, st.session_state.doc_labels,
        )
    except ValueError as e:
        st.error(f"CSV error: {e}", icon="🚫")
        st.stop()
    finally:
        os.unlink(tmp_path)

    if skipped:
        st.warning(
            f"**{len(skipped)} file(s)** in the corpus are not listed "
            f"in the CSV: {', '.join(skipped[:5])}"
            + (f" (and {len(skipped)-5} more…)" if len(skipped) > 5 else ""),
            icon="⚠️",
        )

    if not groupings:
        st.error("No usable groupings in the CSV.", icon="🚫")
        st.stop()

    # Let user pick which grouping to analyse
    grouping_name = st.selectbox(
        "Which grouping do you want to analyse?",
        options=list(groupings.keys()),
    )
    groups = groupings[grouping_name]

if not groups:
    st.error("No groups could be extracted. Check your settings above.",
             icon="🚫")
    st.stop()


# =============================================================================
# Step 2 — Pick a model
# =============================================================================

st.subheader("2. Choose a model")

available = []
if st.session_state.nmf_results is not None:
    available.append("NMF")
if st.session_state.lda_results is not None:
    available.append("LDA")

model_choice = st.radio(
    "Which model do you want to compare on?",
    options=available, horizontal=True,
)

if model_choice == "NMF":
    distribution = mta.topic_distribution_per_doc(
        st.session_state.nmf_results["doctopic"],
        st.session_state.doc_labels,
    )
else:
    distribution = mta.topic_distribution_per_doc(
        st.session_state.lda_results["doctopic"],
        st.session_state.doc_labels,
    )

# Group composition
group_counts = pd.Series(list(groups.values())).value_counts().sort_index()
st.caption(
    f"**Grouping `{grouping_name}`** — "
    + ", ".join(f"{g}: {n}" for g, n in group_counts.items())
    + f" — total {sum(group_counts)} files."
)

if len(group_counts) < 2:
    st.error(
        "Need at least 2 different groups to compare. "
        "Currently only one group exists.",
        icon="🚫",
    )
    st.stop()

# Warn about small samples
small_groups = group_counts[group_counts < 30]
if len(small_groups) > 0:
    st.warning(
        f"⚠️ Some groups have **fewer than 30 documents** "
        f"({', '.join(f'{g}: {n}' for g, n in small_groups.items())}). "
        f"Tests will run but results may be unreliable.",
        icon="⚠️",
    )

st.divider()


# =============================================================================
# Step 3 — Group statistics
# =============================================================================

st.subheader("3. Group statistics (mean ± std per topic)")

stats_df = mta.compute_group_statistics(distribution, groups)
if stats_df.empty:
    st.error("Could not compute statistics — group mapping may be empty.")
    st.stop()

# Display as one row per topic, with columns (group, statistic)
st.dataframe(stats_df.round(4), use_container_width=True)
download_csv(stats_df, f"group_stats_{grouping_name}_{model_choice.lower()}")

# Bar chart with error bars (mean ± std)
mean_df = stats_df.xs("mean", level=0)
std_df = stats_df.xs("std", level=0)
long_data = []
for topic in mean_df.index:
    for group in mean_df.columns:
        long_data.append({
            "Topic": topic,
            "Group": str(group),
            "Mean": mean_df.loc[topic, group],
            "Std":  std_df.loc[topic, group],
        })
long_df = pd.DataFrame(long_data)
long_df["ymin"] = long_df["Mean"] - long_df["Std"]
long_df["ymax"] = long_df["Mean"] + long_df["Std"]

# Build bars + errorbars as a single layered chart, then facet by Topic.
# This is the stable Altair pattern (it avoids passing column= inside
# encode(), which has changed signature across Altair versions).
bars = alt.Chart(long_df).mark_bar().encode(
    x=alt.X("Group:N", title=None),
    y=alt.Y("Mean:Q", title=_LBL["weight"]),
    color=alt.Color("Group:N", scale=alt.Scale(scheme="category10")),
    tooltip=["Topic", "Group",
             alt.Tooltip("Mean:Q", format=".3f"),
             alt.Tooltip("Std:Q", format=".3f")],
)

err = alt.Chart(long_df).mark_errorbar(color="black").encode(
    x=alt.X("Group:N"),
    y=alt.Y("ymin:Q", title=None),
    y2=alt.Y2("ymax:Q"),
)

combined = (bars + err).properties(width=80, height=240)
faceted = combined.facet(column=alt.Column("Topic:N", title=None))

st.altair_chart(faceted, use_container_width=False)


st.divider()


# =============================================================================
# Step 4 — Pairwise tests
# =============================================================================

st.subheader("4. Pairwise statistical tests")

st.markdown(
    "For each topic and each pair of groups, we run **Welch's t-test** "
    "and **Mann-Whitney U**, then apply **Benjamini-Hochberg** "
    "correction on the full set of p-values."
)

tests_df = mta.compare_groups_pairwise(distribution, groups)

if tests_df.empty:
    st.info("Not enough data to run tests.")
    st.stop()

# Configurable significance threshold
alpha = st.slider(
    "Significance threshold α",
    min_value=0.001, max_value=0.10, value=0.05, step=0.005,
    format="%.3f",
    help="Standard convention is 0.05; some fields prefer 0.01.",
)

# Highlight significant cells
def _highlight_sig(val):
    if pd.isna(val):
        return ""
    if val < alpha:
        return "background-color: #c6efce; color: #006100; font-weight: bold;"
    return ""

display_df = tests_df.copy()
# Reorder columns for readability
display_df = display_df[[
    "topic", "group_A", "group_B", "n_A", "n_B",
    "mean_A", "mean_B",
    "p_welch", "p_welch_BH",
    "p_mannwhitney", "p_mannwhitney_BH",
    "small_sample",
]]

styled = display_df.style.format({
    "mean_A": "{:.4f}", "mean_B": "{:.4f}",
    "p_welch": "{:.4f}", "p_welch_BH": "{:.4f}",
    "p_mannwhitney": "{:.4f}", "p_mannwhitney_BH": "{:.4f}",
}).map(_highlight_sig, subset=[
    "p_welch", "p_welch_BH", "p_mannwhitney", "p_mannwhitney_BH"
])

st.dataframe(styled, use_container_width=True, hide_index=True)
st.caption(
    f"🟢 Green cells indicate p < α = {alpha:.3f}. "
    "Compare raw vs BH-corrected: differences that survive BH "
    "correction are the most robust findings."
)
download_csv(tests_df, f"group_tests_{grouping_name}_{model_choice.lower()}")


# =============================================================================
# Step 5 — Visual summary: boxplot per topic
# =============================================================================

st.divider()
st.subheader("5. Distribution per group for one topic")

topic_choice = st.selectbox(
    "Pick a topic to visualise the full distributions",
    options=list(distribution.columns),
)

# Build long-format data for the selected topic
viz_df = distribution[[topic_choice]].copy()
viz_df["Group"] = viz_df.index.map(groups)
viz_df = viz_df.dropna(subset=["Group"])
viz_df = viz_df.rename(columns={topic_choice: "Weight"})

# Combined boxplot + jittered points
box = alt.Chart(viz_df).mark_boxplot(extent="min-max", size=40).encode(
    x=alt.X("Group:N", title=None),
    y=alt.Y("Weight:Q", title=f"{_LBL['weight']} — {topic_choice}"),
    color=alt.Color("Group:N", scale=alt.Scale(scheme="category10"),
                    legend=None),
).properties(height=400)

points = alt.Chart(viz_df).mark_circle(size=30, opacity=0.5).encode(
    x=alt.X("Group:N",
            scale=alt.Scale(padding=0.5)),
    y="Weight:Q",
    xOffset=alt.X("jitter:Q",
                  scale=alt.Scale(domain=[-0.5, 0.5])),
    color=alt.Color("Group:N", legend=None),
    tooltip=[alt.Tooltip("Weight:Q", format=".3f")],
).transform_calculate(
    jitter="random()-0.5"
).properties(height=400)

st.altair_chart(box + points, use_container_width=True)
st.caption(
    "💡 Boxes show median + interquartile range; whiskers show min/max; "
    "circles are individual documents. To download as PNG: click the "
    "**⋯** menu at the top-right of the chart."
)
