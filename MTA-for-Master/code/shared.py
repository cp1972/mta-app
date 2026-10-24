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

import mta_commands as mc


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
    "stability": None,     # 3.5: result of the multi-run stability check
    # 3.7: settings with which the results were computed (for the commands)
    "corpus_settings": None,   # min_word_length, file names, stopword file
    "matrix_settings": None,   # min_word_length, min_df, max_df at build time
    "cv_settings": None,       # max_topics of the cross-validation
    "nmf_settings": None,      # n_topics, init, seed of the NMF run
    "lda_settings": None,      # n_topics, seed of the LDA run
    "command_log": None,       # list of logged commands (see command_box)
    "cmd_paths": None,         # CORPUS / STOPWORDS / GROUPS for the commands
    "cmd_paths_edited": None,  # names of the paths the user typed himself
    "uploader_round": 0,  # incremented to reset uploader widgets
    "chart_language": "en",  # 'en' / 'fr' / 'de' — picked on home page
}


def init_session_state() -> None:
    """Make sure every key our pages might read exists in st.session_state."""
    for k, default in _DEFAULT_STATE.items():
        if k not in st.session_state:
            st.session_state[k] = default
    if st.session_state.command_log is None:
        st.session_state.command_log = []
    if st.session_state.cmd_paths is None:
        st.session_state.cmd_paths = {"CORPUS": "corpus", "STOPWORDS": "", "GROUPS": ""}
    if st.session_state.cmd_paths_edited is None:
        st.session_state.cmd_paths_edited = set()


def reset_corpus_state() -> None:
    """Wipe everything downstream of the corpus (used by 'Clear all files')."""
    for k in ["raw_texts", "doc_labels", "stopwords",
              "corpus_wo", "corpus_re", "matrices", "metrics",
              "nmf_results", "lda_results", "nmf_words", "lda_words",
              "stability", "corpus_settings", "matrix_settings",
              "cv_settings", "nmf_settings", "lda_settings"]:
        st.session_state[k] = None
    # the command log is kept: it documents what was done before


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
    """Standard MTA header for every page (with the session log in the
    sidebar, 3.7)."""
    st.title(title)
    if subtitle:
        st.caption(subtitle)
    command_log_sidebar()


# =============================================================================
# COMMANDS AND SESSION LOG (3.7)
# =============================================================================
# Every analysis shows the batch command that reproduces it (expander "As
# command") and is written to the session log when it is computed. The log
# can be downloaded as a script for Mac/Linux (.sh) or Windows (.bat).
# The translation itself lives in mta_commands.py.

def corpus_context():
    """Settings of corpus and matrices for the commands, as used when the
    matrices were built. Returns (dict, notes)."""
    m = st.session_state.get("matrix_settings") or {}
    c = st.session_state.get("corpus_settings") or {}
    ctx = {"min_word_length": m.get("min_word_length", c.get("min_word_length", 3)),
           "min_df": m.get("min_df", 2), "max_df": m.get("max_df", 0.95),
           "language": get_chart_language()}
    notes = []
    if c and m and c.get("min_word_length") != m.get("min_word_length"):
        notes.append(f"The minimum word length was changed to {c['min_word_length']} "
                     f"after the matrices were built with {m['min_word_length']}; the "
                     "command uses the value of the matrices.")
    return ctx, notes


def model_command_args(model: str) -> list:
    """--n-topics/--init/--seed of the NMF or LDA result in this session."""
    key = "nmf_settings" if model.lower() == "nmf" else "lda_settings"
    s = st.session_state.get(key)
    if s is None:   # no model run yet (e.g. cross-validation first): the app's defaults
        import mta_core as _mta
        n_docs = len(st.session_state.get("doc_labels") or []) or 5
        k = max(2, min(5, n_docs))
        s = ({"n_topics": k, "init": _mta.DEFAULT_NMF_INIT, "seed": _mta.DEFAULT_NMF_SEED}
             if model.lower() == "nmf" else {"n_topics": k, "seed": _mta.DEFAULT_LDA_SEED})
    return mc.model_args(model.lower(), s)


def log_command(analysis_id: str, title: str, args: list, notes=None) -> None:
    """Append a command to the session log unless the last command logged
    for this analysis is the same (pages are recomputed on every click)."""
    log = st.session_state.command_log
    for e in reversed(log):
        if e["id"] == analysis_id:
            if e["args"] == args:
                return
            break
    log.append({"id": analysis_id, "title": title, "args": list(args),
                "notes": list(notes or []),
                "time": datetime.datetime.now().strftime("%H:%M:%S")})


def command_box(analysis_id: str, title: str, label: str, action_args: list,
                notes=None, log: bool = True) -> None:
    """Show the batch command of an analysis and log it.

    analysis_id : stable id of the analysis (e.g. 'nmf', 'evolution_LDA')
    title       : what the log says about it
    label       : name of its output folder
    action_args : the action-specific arguments (mta_commands builds the rest)
    notes       : remarks for things the batch mode does differently
    """
    ctx, ctx_notes = corpus_context()
    args = mc.command(label, ctx, action_args)
    notes = list(notes or []) + ctx_notes
    if log:
        log_command(analysis_id, title, args, notes)
    values = _path_values()
    with st.expander("🧾 As command (batch mode)"):
        st.caption("This command reproduces the analysis above with the batch "
                   "mode of MTA. Run it in a terminal from the MTA-for-Master "
                   "folder. Set the paths of your corpus and stopwords in the "
                   "sidebar under **Session log → Paths and logged analyses**; "
                   "there you can also download all analyses of this session "
                   "as one script.")
        t_posix, t_win = st.tabs(["Mac / Linux", "Windows (Command Prompt)"])
        with t_posix:
            st.code(mc.render(args, "posix", values=values, multiline=True),
                    language="bash")
        with t_win:
            st.code(mc.render(args, "windows", values=values, multiline=True),
                    language="bat")
            st.caption("For the Command Prompt (cmd). In PowerShell, type `cmd` "
                       "first, or run the downloaded .bat script.")
        for n in notes:
            st.caption("ℹ️ " + n)
        # the session script, up to date including this analysis (the
        # sidebar is drawn before the page and shows it after the next click)
        st.markdown("**All analyses of this session as one script**")
        _session_downloads(f"box_{analysis_id}")


def _download_kwargs() -> dict:
    """Downloads without a rerun where Streamlit supports it (1.43+): a rerun
    would close the sidebar section after the first of the two scripts."""
    import inspect
    try:
        ann = inspect.signature(st.download_button).parameters["on_click"].annotation
        return {"on_click": "ignore"} if "ignore" in str(ann) else {}
    except (KeyError, TypeError, ValueError):
        return {}


def _set_path(name: str, key: str) -> None:
    st.session_state.cmd_paths[name] = st.session_state[key].strip()
    st.session_state.cmd_paths_edited.add(name)


def _path_values() -> dict:
    """The paths for the commands. Until the user types a path, the stopword
    file follows the uploaded list (and page 6 the uploaded CSV)."""
    paths = st.session_state.cmd_paths
    c = st.session_state.get("corpus_settings") or {}
    if "STOPWORDS" not in st.session_state.cmd_paths_edited and c.get("stopwords_name"):
        paths["STOPWORDS"] = c["stopwords_name"]
    return {"CORPUS": paths.get("CORPUS") or "corpus",
            "STOPWORDS": paths.get("STOPWORDS") or "stopwords.txt",
            "GROUPS": paths.get("GROUPS") or "groups.csv"}


def _session_downloads(key: str) -> None:
    """The two download buttons for the session script."""
    log = st.session_state.command_log
    if not log:
        return
    values = _path_values()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    header = _log_header()
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "⬇ Script for Mac / Linux (.sh)",
            mc.script(log, "posix", values, header).encode("utf-8"),
            file_name=f"MTA_session_{ts}.sh", mime="text/x-shellscript",
            use_container_width=True, key=f"dl_sh_{key}", **_download_kwargs())
    with c2:
        st.download_button(
            "⬇ Script for Windows (.bat)",
            mc.script(log, "windows", values, header).encode("utf-8"),
            file_name=f"MTA_session_{ts}.bat", mime="application/x-bat",
            use_container_width=True, key=f"dl_bat_{key}", **_download_kwargs())


def _clear_log() -> None:
    st.session_state.command_log = []


def _log_header() -> list:
    c = st.session_state.get("corpus_settings") or {}
    names = c.get("file_names") or []
    h = []
    if names:
        h += ["", f"Corpus of the session: {len(names)} documents (sorted by name, as the "
                  "batch mode reads them):"]
        shown = names if len(names) <= 30 else names[:15] + ["…"] + names[-5:]
        h += ["  " + n for n in shown]
    if c.get("stopwords_name"):
        h.append(f"Stopwords of the session: {c['stopwords_name']} "
                 f"({c.get('n_stopwords', '?')} words)")
    return h


def command_log_sidebar() -> None:
    """Session log in the sidebar: paths for the commands and download."""
    values = _path_values()
    paths = st.session_state.cmd_paths
    log = st.session_state.command_log
    # the inputs show the stored paths (set before the widgets are created)
    st.session_state["_cmd_corpus"] = paths["CORPUS"]
    st.session_state["_cmd_stop"] = values["STOPWORDS"]
    st.session_state["_cmd_groups"] = values["GROUPS"]
    with st.sidebar:
        st.markdown("**🧾 Session log**")
        st.caption(("1 analysis logged as a batch command" if len(log) == 1 else
                    f"{len(log)} analyses logged as batch commands")
                   + ". Download the log as a script to reproduce or refine "
                   "your analyses later.")
        if log:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
            header = _log_header()
            st.download_button(
                "⬇ Script for Mac / Linux (.sh)",
                mc.script(log, "posix", values, header).encode("utf-8"),
                file_name=f"MTA_session_{ts}.sh", mime="text/x-shellscript",
                use_container_width=True, key="dl_log_sh", **_download_kwargs())
            st.download_button(
                "⬇ Script for Windows (.bat)",
                mc.script(log, "windows", values, header).encode("utf-8"),
                file_name=f"MTA_session_{ts}.bat", mime="application/x-bat",
                use_container_width=True, key="dl_log_bat", **_download_kwargs())
        with st.expander("Paths and logged analyses"):
            st.text_input("Folder with your .txt files", key="_cmd_corpus",
                          on_change=_set_path, args=("CORPUS", "_cmd_corpus"),
                          help="Relative to the MTA-for-Master folder, or a full "
                               "path. It must contain the same files as the "
                               "corpus loaded here.")
            st.text_input("Stopword file", key="_cmd_stop",
                          on_change=_set_path, args=("STOPWORDS", "_cmd_stop"))
            if any(mc.GROUPS in e["args"] for e in log):
                st.text_input("CSV with groups", key="_cmd_groups",
                              on_change=_set_path, args=("GROUPS", "_cmd_groups"))
            if log:
                for i, e in enumerate(log, 1):
                    st.caption(f"{i}. {e['time']} — {e['title']}")
                st.button("🗑 Clear the log", on_click=_clear_log,
                          use_container_width=True, key="clear_log")
        st.divider()


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
