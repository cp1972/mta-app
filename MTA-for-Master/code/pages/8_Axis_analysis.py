#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Page 8 — Axis analysis.

User-defined semantic axes on the doctopic matrix, with two
complementary modes accessible via tabs after the axes are defined:

  - Projection (visual) — scatter plot of documents on the axes, with
    endpoint words/topics annotated. Interactive Altair view + static
    matplotlib export.
  - Statistics — enriched CSV export ready for Stata/R, plus one-way
    ANOVA on each axis testing whether groups derived from filenames
    differ significantly in their axis positions. Reports BOTH the
    classical F-test (with Tukey HSD post-hoc) and Welch's F-test
    (with BH-corrected pairwise t-tests).

The axes are defined ONCE in the common header section above the tabs,
so the user picks the topic poles once and then chooses what to do
with them: explore visually, test statistically, or both.

Replaces former separate pages 8 (projection-only) and 9 (stats-only)
in MTA 3.4: the strong overlap between the two pages was confusing
students, who could define an axis on one page and have to redefine
it on the other.
"""

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

import mta_core as mta
from shared import (
    init_session_state,
    page_header,
    require_model,
    download_figure,
    download_csv,
    get_chart_language,
)

init_session_state()
_LANG = get_chart_language()

page_header(
    "🎯 Axis analysis",
    "Project documents onto 1, 2 or 3 user-defined semantic axes, "
    "then explore them visually or test statistically whether groups "
    "differ.",
)

if not require_model():
    st.stop()


# =============================================================================
# Method explainer
# =============================================================================

with st.expander("ℹ️ How does this work?"):
    st.markdown(
        r"""
        Each axis is defined by an **opposition between two pools of
        topics** that you choose: a *left pole* (negative direction) and
        a *right pole* (positive direction).

        The axis direction $v$ in topic space is the contrast vector:

        $$
        v[k] = \begin{cases}
            +1 / |R| & \text{if topic } k \in \text{right pole} \\
            -1 / |L| & \text{if topic } k \in \text{left pole} \\
            \phantom{+}0 & \text{otherwise}
        \end{cases}
        $$

        The position of document $d$ on the axis is the inner product
        $\langle \text{doctopic}[d], v \rangle$ — i.e., a weighted
        contrast of how much $d$ leans toward $R$ versus $L$.

        Documents pulled equally toward both poles score near 0;
        documents weighted strongly on the right pole score positive;
        on the left pole, negative. Either pole may be empty (then the
        axis measures intensity on a single set of topics).

        **A word on "opposition".** It need not mean *antagonism*.
        Two topics may be complementary in content (e.g. *health* and
        *prevention*) and yet usefully contrasted on an axis: some
        documents will weigh more on one than the other, and that
        *relative leaning* is what the axis measures. You can also
        leave one pole empty (e.g. `"/ 3,5"`) — then the axis simply
        measures how strongly documents express a single set of topics,
        without any opposition at all.

        **Why this matters.** Unlike automatic PCA, the axes are
        *interpretable by design* — they encode hypotheses you bring
        to the corpus. This brings into MTA the spirit of Bourdieu's
        correspondence analysis and of Slapin & Proksch's text scaling
        (Wordfish / Wordscores).

        **Two ways to use the axes (tabs below).** Once you've defined
        your axes, you can either *explore them visually* (Projection
        tab — scatter plot, endpoint words, interactive zoom) or
        *test them statistically* (Statistics tab — enriched CSV export
        ready for Stata/R, one-way ANOVA on each axis).
        """
    )


# =============================================================================
# Settings: which topic model, number of axes
# =============================================================================

st.subheader("8.1 — Settings")

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
            "Topic model to project from",
            available, horizontal=True,
        )
    else:
        method = available[0]
        st.markdown(f"**Model:** {method} (only model available)")

# Retrieve the model
if method == "NMF":
    doctopic = nmf_res["doctopic"]
    topicwords = nmf_res["topicwords"]
    vocab = list(matrices["tf_names"])
else:
    doctopic = lda_res["doctopic"]
    topicwords = lda_res["topicwords"]
    vocab = list(matrices["lda_names"])

n_topics = topicwords.shape[0]

# Build short topic descriptions for the multiselects
topic_options = {}
for k in range(n_topics):
    top_idx = np.argsort(topicwords[k])[::-1][:4]
    short = " / ".join(vocab[i] for i in top_idx)
    topic_options[k] = f"T{k+1}: {short}"

with col2:
    n_axes = st.radio(
        "Number of axes",
        [1, 2, 3], index=1, horizontal=True,
        help=(
            "1 axis = 1D line. 2 axes = 2D scatter. 3 axes = 3D plot "
            "(static export only — Altair is 2D)."
        ),
    )

# Topic browser
with st.expander("📋 Topic browser (top-words for each topic)"):
    for k in range(n_topics):
        top_idx = np.argsort(topicwords[k])[::-1][:8]
        words = ", ".join(vocab[i] for i in top_idx)
        st.markdown(f"**T{k+1}** — {words}")


# =============================================================================
# Axis definition (shared between both tabs)
# =============================================================================

st.subheader("8.2 — Define your axes")

st.markdown(
    "For each axis, pick the topics that form its **left pole** "
    "(negative side) and its **right pole** (positive side). A pole may "
    "be left empty (one-sided axis). The two poles of an axis must be "
    "disjoint."
)

axes = []
axis_labels = []

for j in range(int(n_axes)):
    letter = "XYZ"[j]
    st.markdown(f"#### Axis {letter}")
    cl, cr = st.columns(2)
    with cl:
        left = st.multiselect(
            f"Left pole (negative) — Axis {letter}",
            options=list(range(n_topics)),
            format_func=lambda k: topic_options[k],
            key=f"axis_{letter}_left",
        )
    with cr:
        right = st.multiselect(
            f"Right pole (positive) — Axis {letter}",
            options=list(range(n_topics)),
            format_func=lambda k: topic_options[k],
            key=f"axis_{letter}_right",
        )
    custom = st.text_input(
        f"Custom label for axis {letter} (leave empty for auto)",
        key=f"axis_{letter}_label",
        placeholder=f"e.g. Critique ↔ Innovation",
    )
    axes.append((left, right))
    axis_labels.append(custom.strip() if custom.strip() else None)


# =============================================================================
# Validation (shared)
# =============================================================================

validation_errors = []
for j, (left, right) in enumerate(axes):
    letter = "XYZ"[j]
    if not left and not right:
        validation_errors.append(
            f"Axis {letter}: both poles are empty — define at least one."
        )
    if set(left) & set(right):
        overlap = sorted(set(left) & set(right))
        validation_errors.append(
            f"Axis {letter}: topics {overlap} appear in both poles."
        )

if validation_errors:
    for err in validation_errors:
        st.warning(err)
    st.stop()


# =============================================================================
# Endpoint words and axis title resolution (shared)
# =============================================================================

# Endpoint words for each axis (used for plot annotation, sanitized
# column names in the export, and auto-generated labels).
endpoint_words = []
endpoint_topic_names = []
for j, (left, right) in enumerate(axes):
    ew = mta.axis_endpoint_words(
        topicwords, vocab, left, right,
        top_n=30,  # generous; downstream slicing limits actual display
    )
    endpoint_words.append(ew)
    endpoint_topic_names.append({
        "left": [(topic_options[k], 1.0) for k in left],
        "right": [(topic_options[k], 1.0) for k in right],
    })

# Auto-fill missing axis labels
for j, (left, right) in enumerate(axes):
    if axis_labels[j] is None:
        left_words = [w for w, _ in endpoint_words[j].get("left", [])[:3]]
        right_words = [w for w, _ in endpoint_words[j].get("right", [])[:3]]
        l = "/".join(left_words) if left_words else ""
        r = "/".join(right_words) if right_words else ""
        if l and r:
            axis_labels[j] = f"{l} ↔ {r}"
        elif r:
            axis_labels[j] = f"→ {r}"
        elif l:
            axis_labels[j] = f"→ {l}"
        else:
            axis_labels[j] = f"Axis {'XYZ'[j]}"


# Project documents
coords = mta.project_documents_on_axes(doctopic, axes)


# =============================================================================
# TABS — Projection (visual) | Statistics (ANOVA + enriched export)
# =============================================================================

tab_proj, tab_stats = st.tabs([
    "📍 Projection (visual)",
    "📊 Statistics (ANOVA + enriched export)",
])


# -----------------------------------------------------------------------------
# Tab 1 — Projection
# -----------------------------------------------------------------------------

with tab_proj:
    st.markdown(
        "Visualise documents in the user-defined axis space. The "
        "interactive chart supports zoom, pan and hover; a static "
        "publication-ready version is available below for PDF/PNG export."
    )

    # Display options (specific to this tab)
    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        endpoint_mode = st.radio(
            "Show at axis extremities",
            ["Top words", "Topic names", "Both"],
            index=0,
            key="proj_endpoint_mode",
            help=(
                "Top words: most representative words at each end of "
                "each axis. Topic names: the indices/labels of the "
                "topics that compose each pole."
            ),
        )
    with col_d2:
        n_endpoint = st.slider(
            "Number of words / topics shown per extremity",
            min_value=5, max_value=30, value=15,
            key="proj_n_endpoint",
        )
    with col_d3:
        color_choice = st.selectbox(
            "Color documents by",
            ["Dominant topic", "Group (from filename)", "None"],
            index=0,
            key="proj_color_choice",
        )

    # If group coloring chosen, ask for the group position
    proj_group_position = None
    proj_group_separator = "_"
    if color_choice == "Group (from filename)":
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            proj_group_position = st.number_input(
                "Group code position in filename (1-indexed)",
                min_value=1, max_value=10, value=2,
                key="proj_group_position",
            )
        with col_g2:
            proj_group_separator = st.text_input(
                "Separator in filename", value="_",
                key="proj_group_separator",
            )

    # Resolve coloring
    proj_color_values = None
    proj_color_label = "Group"
    if color_choice == "Dominant topic":
        dom = np.argmax(doctopic, axis=1)
        proj_color_values = [topic_options[k] for k in dom]
        proj_color_label = "Dominant topic"
    elif color_choice == "Group (from filename)":
        try:
            groups, skipped = mta.extract_groups_from_filenames(
                labels, position=int(proj_group_position),
                separator=proj_group_separator,
            )
            if skipped:
                st.warning(
                    f"⚠️ {len(skipped)} file(s) have no part at position "
                    f"{proj_group_position}: {', '.join(skipped[:5])}"
                    + (f" (and {len(skipped)-5} more…)"
                       if len(skipped) > 5 else "")
                    + "  — these documents will appear with an empty group."
                )
            if groups:
                proj_color_values = [groups.get(fn, "(no group)")
                                      for fn in labels]
                proj_color_label = f"Group at position {proj_group_position}"
            else:
                st.info(
                    "No group could be derived from filenames at "
                    f"position {proj_group_position}. Falling back to "
                    "no coloring."
                )
        except Exception as e:
            st.warning(f"Could not derive groups: {e}")

    # Build extremity_info for the plot
    if endpoint_mode == "Top words":
        extremity_info = [
            {pole: ew.get(pole, [])[:int(n_endpoint)]
             for pole in ("left", "right")}
            for ew in endpoint_words
        ]
    elif endpoint_mode == "Topic names":
        extremity_info = endpoint_topic_names
    else:  # Both
        extremity_info = []
        for ew, et in zip(endpoint_words, endpoint_topic_names):
            combined = {}
            for pole in ("left", "right"):
                words = ew.get(pole, [])[:int(n_endpoint)]
                tnames = et.get(pole, [])
                sep = [("──────", 0.0)] if words and tnames else []
                combined[pole] = (
                    words
                    + sep
                    + [(f"[{t}]", 0.0) for t, _ in tnames]
                )
            extremity_info.append(combined)

    # Interactive Altair view (1D / 2D)
    st.markdown("#### Interactive view")
    if int(n_axes) <= 2:
        df_plot = pd.DataFrame({
            "document": labels,
            "x": coords[:, 0],
        })
        if int(n_axes) == 2:
            df_plot["y"] = coords[:, 1]
        else:
            rng = np.random.RandomState(42)
            df_plot["y"] = rng.uniform(-0.05, 0.05, size=len(labels))

        if proj_color_values is not None:
            df_plot["color"] = proj_color_values

        base_chart = alt.Chart(df_plot).mark_circle(size=80, opacity=0.7)
        enc = {
            "x": alt.X("x:Q", title=axis_labels[0]),
            "y": alt.Y("y:Q",
                       title=axis_labels[1] if int(n_axes) == 2 else ""),
            "tooltip": (["document", "x"]
                        + (["y"] if int(n_axes) == 2 else [])),
        }
        if proj_color_values is not None:
            enc["color"] = alt.Color("color:N", title=proj_color_label)

        chart = base_chart.encode(**enc).interactive().properties(
            width="container", height=550,
        )
        zero_x = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(
            color="gray", strokeDash=[3, 3]
        ).encode(x="x:Q")
        zero_y = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
            color="gray", strokeDash=[3, 3]
        ).encode(y="y:Q")

        # Endpoint annotations — bounds computed from data
        x_min, x_max = float(df_plot["x"].min()), float(df_plot["x"].max())
        y_min, y_max = float(df_plot["y"].min()), float(df_plot["y"].max())
        x_pad = (x_max - x_min) * 0.01 if x_max > x_min else 0.0
        y_pad = (y_max - y_min) * 0.01 if y_max > y_min else 0.0

        def _build_annotation_df(items, x_pos, y_center, vertical):
            """Stack items vertically at horizontal position x_pos."""
            if not items:
                return None
            rows = []
            n = len(items)
            available = (y_max - y_min) * 0.70 if y_max > y_min else 1.0
            spacing = min(
                (y_max - y_min) * 0.04 if y_max > y_min else 0.05,
                available / max(n - 1, 1),
            )
            for i, (text, _) in enumerate(items):
                offset = (i - (n - 1) / 2) * spacing
                rows.append({
                    "x": x_pos,
                    "y": y_center - offset,
                    "label": text,
                })
            return pd.DataFrame(rows)

        annotation_layers = []

        if int(n_axes) == 2 and extremity_info:
            y_center = (y_min + y_max) / 2
            x_center = (x_min + x_max) / 2

            ew_x = extremity_info[0] if len(extremity_info) > 0 else {}
            df_xleft = _build_annotation_df(
                ew_x.get("left", []),
                x_pos=x_min + x_pad, y_center=y_center, vertical=False,
            )
            df_xright = _build_annotation_df(
                ew_x.get("right", []),
                x_pos=x_max - x_pad, y_center=y_center, vertical=False,
            )

            if df_xleft is not None:
                annotation_layers.append(
                    alt.Chart(df_xleft).mark_text(
                        align="left", baseline="middle",
                        fontSize=10, fontStyle="italic",
                        color="#7a5500",
                    ).encode(
                        x=alt.X("x:Q"), y=alt.Y("y:Q"),
                        text=alt.Text("label:N"),
                    )
                )
            if df_xright is not None:
                annotation_layers.append(
                    alt.Chart(df_xright).mark_text(
                        align="right", baseline="middle",
                        fontSize=10, fontStyle="italic",
                        color="#7a5500",
                    ).encode(
                        x=alt.X("x:Q"), y=alt.Y("y:Q"),
                        text=alt.Text("label:N"),
                    )
                )

            # Y axis: horizontally tiled bands at the top and bottom
            ew_y = extremity_info[1] if len(extremity_info) > 1 else {}

            def _build_y_annotation_df(items, y_pos, x_center,
                                        max_per_row=5):
                if not items:
                    return None
                n = len(items)
                row_spacing = (y_max - y_min) * 0.04
                col_spacing = (x_max - x_min) * 0.15
                direction = 1 if y_pos < y_center else -1
                rows = []
                for i, (text, _) in enumerate(items):
                    row = i // max_per_row
                    col = i % max_per_row
                    n_in_row = min(max_per_row, n - row * max_per_row)
                    x = x_center + (col - (n_in_row - 1) / 2) * col_spacing
                    y = y_pos + direction * row * row_spacing
                    rows.append({"x": x, "y": y, "label": text})
                return pd.DataFrame(rows)

            df_ybot = _build_y_annotation_df(
                ew_y.get("left", []),
                y_pos=y_min + y_pad, x_center=x_center,
            )
            df_ytop = _build_y_annotation_df(
                ew_y.get("right", []),
                y_pos=y_max - y_pad, x_center=x_center,
            )
            if df_ybot is not None:
                annotation_layers.append(
                    alt.Chart(df_ybot).mark_text(
                        align="center", baseline="bottom",
                        fontSize=10, fontStyle="italic",
                        color="#7a5500",
                    ).encode(
                        x=alt.X("x:Q"), y=alt.Y("y:Q"),
                        text=alt.Text("label:N"),
                    )
                )
            if df_ytop is not None:
                annotation_layers.append(
                    alt.Chart(df_ytop).mark_text(
                        align="center", baseline="top",
                        fontSize=10, fontStyle="italic",
                        color="#7a5500",
                    ).encode(
                        x=alt.X("x:Q"), y=alt.Y("y:Q"),
                        text=alt.Text("label:N"),
                    )
                )

        elif int(n_axes) == 1 and extremity_info:
            ew = extremity_info[0]
            df_left = _build_annotation_df(
                ew.get("left", []),
                x_pos=x_min + x_pad, y_center=0.2, vertical=False,
            )
            df_right = _build_annotation_df(
                ew.get("right", []),
                x_pos=x_max - x_pad, y_center=0.2, vertical=False,
            )
            if df_left is not None:
                annotation_layers.append(
                    alt.Chart(df_left).mark_text(
                        align="left", baseline="middle",
                        fontSize=10, fontStyle="italic",
                        color="#7a5500",
                    ).encode(
                        x=alt.X("x:Q"), y=alt.Y("y:Q"),
                        text=alt.Text("label:N"),
                    )
                )
            if df_right is not None:
                annotation_layers.append(
                    alt.Chart(df_right).mark_text(
                        align="right", baseline="middle",
                        fontSize=10, fontStyle="italic",
                        color="#7a5500",
                    ).encode(
                        x=alt.X("x:Q"), y=alt.Y("y:Q"),
                        text=alt.Text("label:N"),
                    )
                )

        # Combine all layers
        layers = [zero_x]
        if int(n_axes) == 2:
            layers.append(zero_y)
        layers.extend(annotation_layers)
        layers.append(chart)

        combined = layers[0]
        for layer in layers[1:]:
            combined = combined + layer

        st.altair_chart(combined, use_container_width=True)

        st.caption(
            "💡 Drag to pan, scroll to zoom. Hover dots to see document "
            "names. Annotations at the axis extremities show the top "
            "words and/or topic names you selected above."
        )
    else:
        st.info(
            "🔍 3D projection — interactive view is not available in "
            "Altair. Use the static export below for a 3D scatter."
        )

    # Coordinates table
    with st.expander("Raw coordinates"):
        coord_cols = [f"axis_{'xyz'[j]}" for j in range(int(n_axes))]
        df_coords = pd.DataFrame(coords, columns=coord_cols)
        df_coords.insert(0, "document", labels)
        if proj_color_values is not None:
            df_coords[proj_color_label] = proj_color_values
        st.dataframe(df_coords.round(4), use_container_width=True,
                     hide_index=True)
        st.caption(
            "📌 For a fuller export (with metadata, dominant topic, all "
            "K topic weights), see the **Statistics** tab."
        )

    # Static publication-ready export
    with st.expander("📐 Publication-ready export (PDF/PNG)"):
        st.markdown(
            "The static figure below uses matplotlib (same look as the "
            "CLI batch export). Extremity annotations and document "
            "labels for the most extreme points are included."
        )
        if st.button("🖼️ Build static figure for download",
                     key="build_axis_static"):
            with st.spinner("Rendering with matplotlib…"):
                fig = mta.plot_axis_projection(
                    coords=coords,
                    labels=labels,
                    axis_titles=axis_labels,
                    color_values=proj_color_values,
                    color_label=proj_color_label,
                    endpoint_words=extremity_info,
                    n_top_endpoint_words=int(n_endpoint),
                    title=f"Axis projection ({method}, K={n_topics})",
                )
            if fig is not None:
                st.pyplot(fig, use_container_width=False)
                download_figure(fig,
                                name=f"axis_projection_{method.lower()}")


# -----------------------------------------------------------------------------
# Tab 2 — Statistics
# -----------------------------------------------------------------------------

with tab_stats:
    st.markdown(
        "Test statistically whether documents from different groups "
        "(extracted from filenames) have significantly different "
        "positions on each axis you defined above. Also produces an "
        "enriched CSV export ready for Stata, R, SPSS or Excel."
    )

    st.markdown("#### Group factor (for ANOVA)")
    st.markdown(
        "The ANOVA tests whether documents from different groups have "
        "different positions on each axis. The groups are derived from a "
        "position in the filename (after splitting by the separator)."
    )

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        stats_group_position = st.number_input(
            "Position of group code (1-indexed)",
            min_value=1, max_value=10, value=2,
            key="stats_group_position",
        )
    with col_s2:
        stats_group_separator = st.text_input(
            "Separator", value="_",
            key="stats_group_separator",
        )
    with col_s3:
        min_group_size = st.number_input(
            "Minimum group size",
            min_value=2, max_value=20, value=3,
            key="stats_min_group_size",
            help="Groups smaller than this are dropped from the ANOVA.",
        )

    # Extract groups for the statistics
    metadata = {}
    groups_dict = {}
    try:
        groups_dict, skipped = mta.extract_groups_from_filenames(
            labels, position=int(stats_group_position),
            separator=stats_group_separator,
        )
        if skipped:
            st.warning(
                f"⚠️ {len(skipped)} file(s) have no part at position "
                f"{stats_group_position}: {', '.join(skipped[:5])}"
                + (f" (and {len(skipped)-5} more…)"
                   if len(skipped) > 5 else "")
                + "  — these documents are dropped from the ANOVA but "
                "kept in the export with an empty group label."
            )
        if groups_dict:
            metadata[f"group_pos{stats_group_position}"] = groups_dict
    except Exception as e:
        st.error(f"Could not extract groups: {e}")
        st.stop()

    # Enriched export (always shown — useful even without group factor)
    st.markdown("#### Enriched export")
    df_export = mta.build_axis_export_dataframe(
        labels=labels, coords=coords,
        axis_titles=axis_labels,
        doctopic=doctopic, topicwords=topicwords, vocab=vocab,
        metadata=metadata,
    )
    st.markdown(
        f"**{df_export.shape[0]} documents × {df_export.shape[1]} "
        "columns.** This table combines all the variables you need for "
        "further analysis in Stata, R, SPSS or Excel."
    )
    st.dataframe(df_export.head(20).round(4), use_container_width=True,
                 hide_index=True)
    download_csv(df_export, f"axis_export_{method.lower()}")

    # ANOVA (only if we have groups)
    if not groups_dict:
        st.info(
            "No group could be derived from the filenames at "
            f"position {stats_group_position} with separator "
            f"'{stats_group_separator}'. ANOVA skipped."
        )
        st.stop()

    st.markdown("---")
    st.markdown("#### One-way ANOVA on each axis")

    group_aligned = [groups_dict.get(fn, "") for fn in labels]

    # Run all ANOVAs
    results_per_axis = {}
    for j, letter in enumerate("XYZ"[:int(n_axes)]):
        results_per_axis[letter] = mta.axis_anova_one_way(
            coord_values=coords[:, j],
            group_labels=group_aligned,
            min_group_size=int(min_group_size),
        )

    # Summary table
    summary_rows = []
    for letter, result in results_per_axis.items():
        if "error" in result:
            continue
        j = "XYZ".index(letter)
        clas = result["classical_anova"]
        welch = result["welch_anova"]
        summary_rows.append({
            "Axis": letter,
            "Title": axis_labels[j],
            "n groups": result["n_groups_used"],
            "n dropped": len(result["dropped_groups"]),
            "F (classical)": round(clas["F"], 3),
            "p (classical)": clas["p_value"],
            "η²": round(clas["eta_squared"], 4),
            "F (Welch)": round(welch["F"], 3),
            "p (Welch)": welch["p_value"],
        })

    if not summary_rows:
        for letter, result in results_per_axis.items():
            st.warning(f"Axis {letter}: "
                       f"{result.get('error', 'unknown error')}")
        st.stop()

    df_summary = pd.DataFrame(summary_rows)
    st.markdown(
        "**ANOVA summary** — F-statistic, p-value and effect size for "
        "each axis, both tests side by side:"
    )
    df_summary_display = df_summary.copy()
    for col in ["p (classical)", "p (Welch)"]:
        df_summary_display[col] = df_summary_display[col].apply(
            lambda p: f"{p:.4g}"
        )
    st.dataframe(df_summary_display, use_container_width=True,
                 hide_index=True)
    download_csv(df_summary, f"axis_anova_summary_{method.lower()}")

    # Per-axis details
    for letter, result in results_per_axis.items():
        if "error" in result:
            continue
        j = "XYZ".index(letter)
        clas = result["classical_anova"]
        welch = result["welch_anova"]

        st.markdown("---")
        st.markdown(f"### Axis {letter} — {axis_labels[j]}")

        with st.expander(f"Group summary (n, mean, std) — Axis {letter}"):
            gs = result["group_summary"].copy()
            gs_display = gs.copy()
            for col in ["mean", "std", "min", "max"]:
                gs_display[col] = gs_display[col].round(4)
            st.dataframe(gs_display, use_container_width=True,
                         hide_index=True)

        # Convergence indicator
        p_cls = clas["p_value"]
        p_wel = welch["p_value"]
        both_sig = (p_cls < 0.05) and (p_wel < 0.05)
        both_not = (p_cls >= 0.05) and (p_wel >= 0.05)
        if both_sig:
            eta = clas["eta_squared"]
            effect = ("small" if eta < 0.06 else
                       "medium" if eta < 0.14 else "large")
            st.success(
                f"✓ **Convergence**: both tests significant (p < 0.05). "
                f"Robust conclusion: groups differ on axis {letter}. "
                f"η² = {eta:.3f} ({effect} effect)."
            )
        elif both_not:
            st.info(
                f"○ **Convergence**: both tests non-significant. "
                f"Robust conclusion: no detectable difference between "
                f"groups on axis {letter}."
            )
        else:
            st.warning(
                f"⚠️ **Divergence**: classical F p = {p_cls:.4g}, "
                f"Welch p = {p_wel:.4g}. Check the boxplots below: if "
                f"box widths differ strongly between groups, variance "
                f"heterogeneity is at play and the Welch result is more "
                f"trustworthy."
            )

        # Pairwise comparisons
        col_t, col_w = st.columns(2)
        with col_t:
            st.markdown("**Tukey HSD (post-hoc, classical)**")
            tk = result["tukey_pairwise"].copy()
            if not tk.empty:
                tk_display = tk.copy()
                for col in ["mean_diff", "ci_low", "ci_high"]:
                    tk_display[col] = tk_display[col].round(4)
                tk_display["p_tukey"] = tk_display["p_tukey"].apply(
                    lambda p: f"{p:.4g}"
                )
                st.dataframe(tk_display, use_container_width=True,
                             hide_index=True)
            else:
                st.markdown("_(no pairs)_")
        with col_w:
            st.markdown("**Pairwise Welch t-tests with BH correction**")
            wp = result["welch_pairwise"].copy()
            if not wp.empty:
                wp_display = wp.copy()
                wp_display["mean_diff"] = wp_display["mean_diff"].round(4)
                wp_display["t"] = wp_display["t"].round(3)
                for col in ["p_welch", "p_welch_BH"]:
                    wp_display[col] = wp_display[col].apply(
                        lambda p: f"{p:.4g}"
                    )
                st.dataframe(wp_display, use_container_width=True,
                             hide_index=True)
            else:
                st.markdown("_(no pairs)_")

    # Boxplots
    st.markdown("---")
    st.markdown("#### Visual summary (boxplots)")

    axis_values = {letter: coords[:, "XYZ".index(letter)]
                   for letter in results_per_axis
                   if "error" not in results_per_axis[letter]}
    axis_titles_dict = {letter: axis_labels["XYZ".index(letter)]
                        for letter in axis_values}

    if axis_values:
        fig = mta.plot_axis_anova_boxplots(
            axis_values=axis_values,
            group_labels=group_aligned,
            axis_titles=axis_titles_dict,
            title=f"ANOVA: axis coordinates by group at position "
                  f"{stats_group_position} ({method}, K={n_topics})",
            min_group_size=int(min_group_size),
        )
        if fig is not None:
            st.pyplot(fig, use_container_width=False)
            download_figure(
                fig, name=f"axis_anova_boxplots_{method.lower()}",
            )

    # Downloadable companion tables
    with st.expander("📥 Download all ANOVA tables (CSV)"):
        st.markdown(
            "In addition to the summary above, you can download the "
            "detailed pairwise tables (one row per pair, with the axis "
            "letter in the first column):"
        )
        welch_rows, tukey_rows, group_rows = [], [], []
        for letter, result in results_per_axis.items():
            if "error" in result:
                continue
            for src, dst in [("welch_pairwise", welch_rows),
                              ("tukey_pairwise", tukey_rows),
                              ("group_summary", group_rows)]:
                df = result[src].copy()
                if not df.empty:
                    df.insert(0, "axis", letter)
                    dst.append(df)

        if welch_rows:
            df_wp = pd.concat(welch_rows, ignore_index=True)
            st.markdown("**Welch pairwise (with BH correction)**")
            download_csv(df_wp,
                         f"axis_anova_welch_pairwise_{method.lower()}")
        if tukey_rows:
            df_tp = pd.concat(tukey_rows, ignore_index=True)
            st.markdown("**Tukey HSD pairwise**")
            download_csv(df_tp,
                         f"axis_anova_tukey_pairwise_{method.lower()}")
        if group_rows:
            df_gs = pd.concat(group_rows, ignore_index=True)
            st.markdown("**Group summary**")
            download_csv(df_gs,
                         f"axis_anova_group_summary_{method.lower()}")
