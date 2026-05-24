#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
shared.py
=========

Utilities shared across all MTA Streamlit pages.

Why a single module?
- Streamlit pages are independent scripts; each runs from top to bottom
  on every interaction. Putting shared code here keeps pages short.
- Session state is initialized in one place, so every page is guaranteed
  to find the same keys.
- Access guards (require_corpus, require_model) live here, so each page
  starts with a one-line check.
"""

import datetime
import io

import pandas as pd
import streamlit as st


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

_DEFAULT_STATE = {
    "raw_texts": None,
    "doc_labels": None,
    "stopwords": None,
    "corpus_wo": None,
    "corpus_re": None,
    "matrices": None,
    "metrics": None,
    "nmf_results": None,
    "lda_results": None,
    "nmf_words": None,
    "lda_words": None,
    "uploader_round": 0,  # incremented to reset uploader widgets
    "chart_language": "en",  # 'en' / 'fr' / 'de' — picked on home page
}


def init_session_state() -> None:
    """Make sure every key our pages might read exists in st.session_state."""
    for k, default in _DEFAULT_STATE.items():
        if k not in st.session_state:
            st.session_state[k] = default


def reset_corpus_state() -> None:
    """Wipe everything downstream of the corpus (used by 'Clear all files')."""
    for k in ["raw_texts", "doc_labels", "stopwords",
              "corpus_wo", "corpus_re", "matrices", "metrics",
              "nmf_results", "lda_results", "nmf_words", "lda_words"]:
        st.session_state[k] = None


# =============================================================================
# CHART LANGUAGE
# =============================================================================

def get_chart_language() -> str:
    """
    Return the user's chosen chart language (set on the Home page).
    Falls back to 'en' if not set yet — so pages visited directly never
    crash and just render in English.
    """
    return st.session_state.get("chart_language", "en")


# =============================================================================
# ACCESS GUARDS (STRICT LOCKING)
# =============================================================================

def require_corpus() -> bool:
    """
    Page-level guard. Returns True if a corpus has been loaded; otherwise
    renders a lock message and returns False. Page should `return` after
    a False to avoid further rendering.
    """
    if st.session_state.get("corpus_wo") is None:
        st.error(
            "🔒 **This page is locked.** You must first load a corpus and "
            "build the term-document matrices.",
            icon="🚫",
        )
        st.info(
            "👉 Go to the **📥 Load corpus** page in the left sidebar.",
        )
        return False
    return True


def require_matrices() -> bool:
    """Like require_corpus, but also demands that matrices are built."""
    if not require_corpus():
        return False
    if st.session_state.get("matrices") is None:
        st.error(
            "🔒 **This page is locked.** You loaded a corpus but the "
            "term-document matrices have not been built yet.",
            icon="🚫",
        )
        st.info(
            "👉 Go back to **📥 Load corpus** and click *Build the matrices*.",
        )
        return False
    return True


def require_model() -> bool:
    """Like require_matrices, but also demands NMF or LDA has been run."""
    if not require_matrices():
        return False
    if (st.session_state.get("nmf_results") is None
            and st.session_state.get("lda_results") is None):
        st.error(
            "🔒 **This page is locked.** You must first run at least one "
            "topic model (NMF or LDA).",
            icon="🚫",
        )
        st.info(
            "👉 Go to the **📊 Topic models** page in the left sidebar.",
        )
        return False
    return True


# =============================================================================
# SHARED HEADER (rendered at the top of every page for consistency)
# =============================================================================

def page_header(title: str, subtitle: str = "") -> None:
    """Standard MTA header for every page."""
    st.title(title)
    if subtitle:
        st.caption(subtitle)


# =============================================================================
# DOWNLOAD BUTTONS
# =============================================================================

def download_csv(df: pd.DataFrame, name: str) -> None:
    """CSV download button for a DataFrame."""
    csv = df.to_csv(index=True).encode("utf-8")
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button(
        label=f"⬇ Download {name}.csv",
        data=csv,
        file_name=f"{name}_{ts}.csv",
        mime="text/csv",
        key=f"dl_{name}_{ts}",
    )


def download_figure(fig, name: str, formats: tuple = ("png", "pdf"),
                    dpi: int = 150) -> None:
    """
    Render side-by-side download buttons for an already-built matplotlib
    Figure. Used by pages that produce figures directly (e.g. network
    views) rather than tabular chart data.

    formats : tuple of ('png', 'pdf')
    """
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cols = st.columns(len(formats))
    for col, fmt in zip(cols, formats):
        buf = io.BytesIO()
        if fmt == "png":
            fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
            mime = "image/png"
        elif fmt == "pdf":
            fig.savefig(buf, format="pdf", bbox_inches="tight")
            mime = "application/pdf"
        else:
            continue
        buf.seek(0)
        with col:
            st.download_button(
                label=f"⬇ Download {name}.{fmt}",
                data=buf,
                file_name=f"{name}_{ts}.{fmt}",
                mime=mime,
                key=f"dl_{name}_{fmt}_{ts}",
            )


def download_png_via_matplotlib(
    chart_data: pd.DataFrame,
    kind: str,
    name: str,
    xlabel: str = "",
    ylabel: str = "",
    stacked: bool = False,
    grey_col: str = None,
) -> None:
    """
    Provide a PNG export by re-rendering the data via matplotlib.

    Long row labels (e.g. document filenames) are truncated to keep the
    figure readable; the underlying CSV export keeps the full names.
    The figure size adapts to the number of rows.

    If grey_col is given AND kind == 'bar', that column is plotted in
    light grey (used for the "Other topics (sum)" context segment).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colormaps

    # Import the helpers from mta_core for consistency with MTA_v3.py
    import mta_core as _mta

    buf = io.BytesIO()
    n = len(chart_data)
    figsize = _mta.auto_figsize(n)
    fig, ax = plt.subplots(figsize=figsize, dpi=130)

    # Truncate row labels for display only, keep underlying data intact
    display_df = chart_data.copy()
    display_df.index = _mta.truncate_labels(display_df.index)

    if kind == "bar":
        paired = colormaps["Paired"]
        cols = list(display_df.columns)
        colors = []
        normal_idx = 0
        for c in cols:
            if c == grey_col:
                colors.append("#bbbbbb")
            else:
                colors.append(paired(normal_idx % paired.N))
                normal_idx += 1
        display_df.plot(kind="bar", stacked=stacked, ax=ax,
                        color=colors, width=0.85)
        plt.xticks(rotation=90, fontsize=7 if n > 40 else 8)
    elif kind == "line":
        display_df.plot(ax=ax, marker="o", markersize=3, linewidth=1.2)
        if n > 20:
            plt.xticks(rotation=90, fontsize=7)
        else:
            plt.xticks(rotation=45, fontsize=8, ha="right")

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(_mta.wrap_ylabel(ylabel))
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5),
              prop={"size": 8}, frameon=False)
    fig.subplots_adjust(left=0.10, right=0.82, top=0.95, bottom=0.30)
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button(
        label=f"⬇ Download {name}.png",
        data=buf,
        file_name=f"{name}_{ts}.png",
        mime="image/png",
        key=f"dlpng_{name}_{ts}",
    )


# =============================================================================
# MISC HELPERS
# =============================================================================

def human_size(num_bytes: int) -> str:
    """Format a byte count into a human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


# Color palette for Altair stacked bars (Paired-like).
PAIRED_PALETTE = [
    "#a6cee3", "#1f78b4", "#b2df8a", "#33a02c",
    "#fb9a99", "#e31a1c", "#fdbf6f", "#ff7f00",
    "#cab2d6", "#6a3d9a", "#ffff99", "#b15928",
]


def paired_color_range(columns, grey_col: str = None):
    """Build an Altair color range for a list of columns, with optional grey."""
    color_range = []
    normal_i = 0
    for c in columns:
        if c == grey_col:
            color_range.append("#bbbbbb")
        else:
            color_range.append(PAIRED_PALETTE[normal_i % len(PAIRED_PALETTE)])
            normal_i += 1
    return color_range
