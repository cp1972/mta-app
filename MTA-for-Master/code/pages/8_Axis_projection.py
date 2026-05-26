#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Page 8 — Axis projection.

User-defined semantic axes on the doctopic matrix. Instead of running
an automatic PCA (which picks the directions of maximum variance, often
hard to interpret), the user defines each axis as an *opposition*
between two pools of topics. The K-dimensional doctopic matrix is then
projected onto these 1, 2 or 3 user-chosen axes.

This is the spirit of Bourdieu's correspondence analysis (axes as
interpretable oppositions, not as variance directions) and of Slapin &
Proksch's text scaling. The researcher's hypotheses drive the
dimensions, not the algorithm.
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
    get_chart_language,
)

init_session_state()
_LANG = get_chart_language()

page_header(
    "🎯 Axis projection",
    "Project documents onto 1, 2 or 3 user-defined semantic axes, "
    "each formed by an opposition between two pools of topics.",
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
        """
    )


# =============================================================================
# Method choice (NMF / LDA) and basic settings
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

# =============================================================================
# Topic browser (so users see what they're picking)
# =============================================================================

with st.expander("📋 Topic browser (top-words for each topic)"):
    for k in range(n_topics):
        top_idx = np.argsort(topicwords[k])[::-1][:8]
        words = ", ".join(vocab[i] for i in top_idx)
        st.markdown(f"**T{k+1}** — {words}")


# =============================================================================
# Axis definition
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
# Display options
# =============================================================================

st.subheader("8.3 — Display options")

col_d1, col_d2, col_d3 = st.columns(3)
with col_d1:
    endpoint_mode = st.radio(
        "Show at axis extremities",
        ["Top words", "Topic names", "Both"],
        index=0,
        help=(
            "Top words: most representative words at each end of each "
            "axis (computed from the topicwords matrix). Topic names: "
            "the indices/labels of the topics that compose each pole."
        ),
    )
with col_d2:
    n_endpoint = st.slider(
        "Number of words / topics shown per extremity",
        min_value=5, max_value=30, value=15,
    )
with col_d3:
    color_choice = st.selectbox(
        "Color documents by",
        ["Dominant topic", "Group (from filename)", "None"],
        index=0,
    )

# If group coloring chosen, ask for the group position
group_position = None
group_separator = "_"
if color_choice == "Group (from filename)":
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        group_position = st.number_input(
            "Group code position in filename (1-indexed)",
            min_value=1, max_value=10, value=2,
        )
    with col_g2:
        group_separator = st.text_input(
            "Separator in filename", value="_",
        )


# =============================================================================
# Validation and computation
# =============================================================================

# Validate axes
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


# Compute endpoint information for each axis.
# axis_endpoint_words returns weights sorted by descending value; we
# slice to n_endpoint downstream depending on the chosen display mode.
endpoint_words = []
endpoint_topic_names = []
for j, (left, right) in enumerate(axes):
    ew = mta.axis_endpoint_words(
        topicwords, vocab, left, right,
        top_n=max(30, int(n_endpoint)),
    )
    endpoint_words.append(ew)
    # Topic names per pole: just the short topic descriptions
    endpoint_topic_names.append({
        "left": [(topic_options[k], 1.0) for k in left],
        "right": [(topic_options[k], 1.0) for k in right],
    })

# Build `extremity_info` — what actually goes onto the figures —
# according to the user's display mode. Slicing happens HERE (not in
# the plotting helper), so that all three modes show the right thing:
#   - Top words: top-N mots only
#   - Topic names: all topic names (typically 1–3, no truncation needed)
#   - Both: top-N mots, then the topic names underneath
if endpoint_mode == "Top words":
    extremity_info = [
        {pole: ew.get(pole, [])[:int(n_endpoint)]
         for pole in ("left", "right")}
        for ew in endpoint_words
    ]
elif endpoint_mode == "Topic names":
    # Topic names are typically few; show them all
    extremity_info = endpoint_topic_names
else:  # Both
    extremity_info = []
    for ew, et in zip(endpoint_words, endpoint_topic_names):
        combined = {}
        for pole in ("left", "right"):
            words = ew.get(pole, [])[:int(n_endpoint)]
            tnames = et.get(pole, [])
            # Words first, then a visual separator and topic names
            sep = [("──────", 0.0)] if words and tnames else []
            combined[pole] = (
                words
                + sep
                + [(f"[{t}]", 0.0) for t, _ in tnames]
            )
        extremity_info.append(combined)


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


# Color values
color_values = None
color_label = "Group"
if color_choice == "Dominant topic":
    dom = np.argmax(doctopic, axis=1)
    color_values = [topic_options[k] for k in dom]
    color_label = "Dominant topic"
elif color_choice == "Group (from filename)":
    try:
        groups, skipped = mta.extract_groups_from_filenames(
            labels, position=int(group_position),
            separator=group_separator,
        )
        if skipped:
            st.warning(
                f"⚠️ {len(skipped)} file(s) have no part at position "
                f"{group_position}: {', '.join(skipped[:5])}"
                + (f" (and {len(skipped)-5} more…)" if len(skipped) > 5
                   else "")
                + "  — these documents will appear with an empty group."
            )
        if groups:
            # Build color_values in the same order as `labels`
            color_values = [groups.get(fn, "(no group)") for fn in labels]
            color_label = f"Group at position {group_position}"
        else:
            st.info(
                "No group could be derived from filenames at "
                f"position {group_position} with separator "
                f"'{group_separator}'. Falling back to no coloring."
            )
    except Exception as e:
        st.warning(f"Could not derive groups: {e}")


# =============================================================================
# Interactive view (Altair) for 1D / 2D
# =============================================================================

st.subheader("8.4 — Interactive view")

if int(n_axes) <= 2:
    # Build DataFrame for Altair
    df_plot = pd.DataFrame({
        "document": labels,
        "x": coords[:, 0],
    })
    if int(n_axes) == 2:
        df_plot["y"] = coords[:, 1]
    else:
        # 1D: jitter for visibility
        rng = np.random.RandomState(42)
        df_plot["y"] = rng.uniform(-0.05, 0.05, size=len(labels))

    if color_values is not None:
        df_plot["color"] = color_values

    base_chart = alt.Chart(df_plot).mark_circle(size=80, opacity=0.7)

    enc = {
        "x": alt.X("x:Q", title=axis_labels[0]),
        "y": alt.Y("y:Q",
                   title=axis_labels[1] if int(n_axes) == 2 else ""),
        "tooltip": ["document", "x"] + (["y"] if int(n_axes) == 2 else []),
    }
    if color_values is not None:
        enc["color"] = alt.Color("color:N", title=color_label)

    chart = base_chart.encode(**enc).interactive().properties(
        width="container",
        height=550,
    )

    # Zero lines
    zero_x = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(
        color="gray", strokeDash=[3, 3]
    ).encode(x="x:Q")
    zero_y = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
        color="gray", strokeDash=[3, 3]
    ).encode(y="y:Q")

    # -- Endpoint annotations -------------------------------------------------
    # Build per-axis text rows: one row = one word/topic to display at
    # an extremity, with coordinates set to the data's min/max so the
    # text anchors to the visible bounds.
    x_min, x_max = float(df_plot["x"].min()), float(df_plot["x"].max())
    y_min, y_max = float(df_plot["y"].min()), float(df_plot["y"].max())
    # Small padding inward so the text doesn't sit exactly on the edge
    x_pad = (x_max - x_min) * 0.01 if x_max > x_min else 0.0
    y_pad = (y_max - y_min) * 0.01 if y_max > y_min else 0.0

    def _build_annotation_df(items, x_pos, y_center, vertical):
        """
        Stack `items` vertically at horizontal position x_pos, centered
        around y_center. The total stack height is capped to ~70 % of
        the visible range so it doesn't overflow into the scatter cloud.
        """
        if not items:
            return None
        rows = []
        n = len(items)
        # Reserve at most 70% of the height for the stack and cap the
        # per-item spacing accordingly (so 15+ items still fit cleanly)
        available = (y_max - y_min) * 0.70 if y_max > y_min else 1.0
        spacing = min(
            (y_max - y_min) * 0.04 if y_max > y_min else 0.05,
            available / max(n - 1, 1),
        )
        for i, (text, _) in enumerate(items):
            offset = (i - (n - 1) / 2) * spacing
            rows.append({
                "x": x_pos,
                "y": y_center - offset,  # top of list above
                "label": text,
            })
        return pd.DataFrame(rows)

    annotation_layers = []

    if int(n_axes) == 2 and extremity_info:
        # X axis: words/topics at left (x=x_min) and right (x=x_max),
        # vertically centered around y_center
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

        # Y axis: words/topics at bottom (y=y_min) and top (y=y_max),
        # horizontally centered around x_center
        ew_y = extremity_info[1] if len(extremity_info) > 1 else {}

        def _build_y_annotation_df(items, y_pos, x_center,
                                    max_per_row=5):
            """
            Stack items horizontally in a band near the top or bottom
            edge of the plot. Items wrap after `max_per_row` and the
            band grows inward (away from the edge). Far more readable
            than a single vertical column when there are many words,
            and never overlaps the document scatter near the center.
            """
            if not items:
                return None
            n = len(items)
            row_spacing = (y_max - y_min) * 0.04
            col_spacing = (x_max - x_min) * 0.15
            direction = 1 if y_pos < y_center else -1  # grow inward
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
        # 1D: words at the two ends, stacked vertically just above the line
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
    layers.append(chart)  # documents on top for hover visibility

    combined = layers[0]
    for layer in layers[1:]:
        combined = combined + layer

    st.altair_chart(combined, use_container_width=True)

    st.caption(
        "💡 Drag to pan, scroll to zoom. Hover dots to see document "
        "names. Annotations at the axis extremities show the top words "
        "and/or topic names you selected above (use the slider to change "
        "how many). The static export below produces the same view as "
        "a publication-ready PDF/PNG with framed annotation boxes."
    )
else:
    st.info(
        "🔍 3D projection — interactive view is not available in Altair. "
        "Use the static export below for a 3D scatter (PDF/PNG)."
    )


# =============================================================================
# Coordinates table
# =============================================================================

with st.expander("Raw coordinates (downloadable)"):
    coord_cols = [f"axis_{'xyz'[j]}" for j in range(int(n_axes))]
    df_coords = pd.DataFrame(coords, columns=coord_cols)
    df_coords.insert(0, "document", labels)
    if color_values is not None:
        df_coords[color_label] = color_values
    st.dataframe(df_coords.round(4), use_container_width=True,
                 hide_index=True)
    from shared import download_csv
    download_csv(df_coords, f"axis_projection_{method.lower()}_coords")


# =============================================================================
# Publication-ready static export
# =============================================================================

with st.expander("📐 Publication-ready export (PDF/PNG)"):
    st.markdown(
        "The static figure below uses matplotlib (same look as the CLI "
        "batch export). Extremity annotations and document labels for "
        "the most extreme points are included."
    )
    if st.button("🖼️ Build static figure for download",
                 key="build_axis_static"):
        with st.spinner("Rendering with matplotlib…"):
            fig = mta.plot_axis_projection(
                coords=coords,
                labels=labels,
                axis_titles=axis_labels,
                color_values=color_values,
                color_label=color_label,
                endpoint_words=extremity_info,
                n_top_endpoint_words=int(n_endpoint),
                title=f"Axis projection ({method}, K={n_topics})",
            )
        if fig is not None:
            st.pyplot(fig, use_container_width=False)
            download_figure(fig,
                            name=f"axis_projection_{method.lower()}")
