#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
home.py — MTA landing page.

Rendered when the user opens the app or clicks "Home" in the sidebar.
Session state initialization happens here, so it runs at least once
before the other pages can be visited.
"""

import streamlit as st

from shared import init_session_state, page_header

init_session_state()

page_header(
    "📚 MTA — Multi-Text Analyser",
    "Interactive topic modelling for the Master in Sociology course. "
    "No installation required: everything happens here, in your browser.",
)

st.markdown(
    """
    ## Welcome

    MTA helps you analyze a corpus of texts with **topic modelling**:
    NMF, LDA, cross-validation, word weights, and more. The whole workflow
    is split into pages — visit them **in order** using the sidebar on the left.

    ### Steps

    | # | Page | What it does |
    |---|------|--------------|
    | 1 | **📥 Load corpus** | Upload your texts and stopwords, then build the term-document matrices. |
    | 2 | **📊 Topic models** | Find the best number of topics (cross-validation), then run NMF and LDA. |
    | 3 | **🔍 Word weights** | For a given word (or several), see its weight in each topic and in each document. |
    | 4 | **📈 Topic evolution** | Rolling mean of topic weights across documents; optional yearly aggregation. |
    | 5 | **🧠 Semantic context** | Find similar words, visualise semantic clouds in 2D, select sub-corpora by keyword. |
    | 6 | **⚖️ Group comparison** | Test whether documents differ significantly between groups (e.g. F vs M, age bands…). |
    | 7 | **🕸 Network views** | Bipartite network graphs of topics, documents and top-words (publication-ready). |

    ### Tips

    - Pages are **locked** until their prerequisites are met. If a page
      shows 🔒, just follow the link it gives you to the right step.
    - Your data **never leaves your computer** — everything runs locally
      in your browser.
    - Use the **🗑 Clear all files** button on the *Load corpus* page to
      start a new analysis.
    """
)

st.divider()

# ============================================================================
# Chart language selector — applies to every chart on every other page.
# ============================================================================

st.subheader("Chart language")
st.caption(
    "Language used for chart axis labels, titles, and legends throughout "
    "the app. You can change it at any time by coming back to this page."
)

_LANG_LABELS = {"en": "English", "fr": "Français", "de": "Deutsch"}
_options = ["en", "fr", "de"]

# Pre-select the current value so the widget stays in sync with state
_current = st.session_state.get("chart_language", "en")
chosen = st.radio(
    "Chart language",
    options=_options,
    index=_options.index(_current) if _current in _options else 0,
    format_func=lambda c: _LANG_LABELS[c],
    horizontal=True,
    label_visibility="collapsed",
)
st.session_state.chart_language = chosen

st.divider()

# Quick status overview, helpful for the user to know where they are.
st.subheader("Current session status")

c1, c2, c3 = st.columns(3)

with c1:
    n_docs = (len(st.session_state.raw_texts)
              if st.session_state.raw_texts else 0)
    st.metric("📄 Documents loaded", n_docs)

with c2:
    n_vocab = (len(st.session_state.matrices["tf_names"])
               if st.session_state.matrices else 0)
    st.metric("📖 Vocabulary size", f"{n_vocab:,}" if n_vocab else "—")

with c3:
    models = []
    if st.session_state.nmf_results is not None:
        models.append("NMF")
    if st.session_state.lda_results is not None:
        models.append("LDA")
    st.metric("🧮 Models run", " + ".join(models) if models else "—")

st.divider()

st.caption(
    "MTA — Multi-Text Analyser • C. Papilloud (2017-2026) • "
    "Streamlit interface for the Master in Sociology course, "
    "Martin Luther University Halle-Wittenberg."
)
