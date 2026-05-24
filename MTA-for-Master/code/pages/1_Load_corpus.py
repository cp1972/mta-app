#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Page 1 — Load corpus and build matrices.

This page combines the original Step 1 (upload + cleaning) and Step 2
(matrix construction), since the two operations are conceptually tied
and always done together.
"""

import pandas as pd
import streamlit as st

import mta_core as mta
from shared import (
    init_session_state,
    page_header,
    reset_corpus_state,
    human_size,
)

# Page config is handled centrally by streamlit_app.py + st.navigation.
init_session_state()

page_header("📥 Load corpus and build matrices",
            "Step 1 of the MTA workflow.")


# =============================================================================
# SIDEBAR — global parameters
# =============================================================================

with st.sidebar:
    st.header("⚙️ General settings")
    min_word_length = st.slider(
        "Minimum word length (characters)",
        min_value=2, max_value=9, value=3,
        help="Shorter words are ignored. MTA standard: 3.",
    )
    st.caption(
        "Upload limit: **5 GB**. Large corpora (thousands of documents) "
        "will take longer to process — be patient during cross-validation."
    )


# =============================================================================
# CLEAR ALL FILES button (centered, top of the page)
# =============================================================================

col_left, col_center, col_right = st.columns([2, 1, 2])
with col_center:
    if st.button("🗑 Clear all files",
                 help="Reset both upload boxes at once",
                 use_container_width=True):
        st.session_state.uploader_round += 1
        reset_corpus_state()
        st.rerun()

uploader_key_texts = f"uploader_texts_{st.session_state.uploader_round}"
uploader_key_stops = f"uploader_stops_{st.session_state.uploader_round}"


# =============================================================================
# 1.1 — UPLOAD
# =============================================================================

st.subheader("1.1 — Upload corpus and stopwords")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Your texts**")
    text_files = st.file_uploader(
        "Select your .txt files (multiple selection allowed)",
        type=["txt"],
        accept_multiple_files=True,
        help="Hold Ctrl (Windows/Linux) or Cmd (Mac) to pick several files. "
             "Up to 5 GB total.",
        key=uploader_key_texts,
    )

with col2:
    st.markdown("**Your stopwords**")
    stop_file = st.file_uploader(
        "Select a stopwords file (one word per line)",
        type=["txt"],
        help="One word per line. Ready-made lists are available in the "
             "examples/ folder.",
        key=uploader_key_stops,
    )

if text_files:
    total_size = sum(getattr(f, "size", 0) for f in text_files)
    st.success(
        f"📄 **{len(text_files):,} text file(s) loaded** "
        f"— total size: {human_size(total_size)}"
    )
    if len(text_files) > 10:
        with st.expander(f"Show all {len(text_files)} filenames", expanded=False):
            names_df = pd.DataFrame({
                "#": range(1, len(text_files) + 1),
                "Filename": [f.name for f in text_files],
                "Size": [human_size(getattr(f, "size", 0)) for f in text_files],
            })
            st.dataframe(names_df, use_container_width=True, hide_index=True,
                         height=min(300, 35 * len(text_files) + 38))

# --- Preprocessing happens here, as soon as both uploads are present ---
if text_files and stop_file:
    raw_texts = [f.read().decode("utf-8", errors="replace") for f in text_files]
    doc_labels = [f.name for f in text_files]
    stopwords = [
        line.strip() for line in
        stop_file.read().decode("utf-8", errors="replace").splitlines()
        if line.strip()
    ]

    st.session_state.raw_texts = raw_texts
    st.session_state.doc_labels = doc_labels
    st.session_state.stopwords = stopwords

    st.success(
        f"✓ Corpus loaded: **{len(raw_texts):,} documents**, "
        f"**{len(stopwords):,} stopwords**."
    )

    with st.expander("Preview of the first document (200 characters)"):
        st.text(raw_texts[0][:200] + "…")

    with st.spinner("Cleaning the corpus…"):
        corpus_wo, corpus_re = mta.preprocess_corpus(
            raw_texts, stopwords, min_word_length=min_word_length
        )
    st.session_state.corpus_wo = corpus_wo
    st.session_state.corpus_re = corpus_re

    st.info("Corpus cleaned. Continue with section 1.2 below.", icon="✅")
else:
    st.warning(
        "Please load both a corpus of texts AND a stopwords file "
        "to continue.",
        icon="📁",
    )

st.divider()


# =============================================================================
# 1.2 — BUILD MATRICES
# =============================================================================

st.subheader("1.2 — Build the term-document matrices")

if st.session_state.corpus_wo is None:
    st.info("This section unlocks once your corpus is loaded above.",
            icon="🔒")
else:
    st.markdown(
        "MTA builds two matrices: a TF-IDF one for NMF, a Count one for LDA. "
        "The parameters below filter words that are too rare or too frequent."
    )

    use_default = st.radio(
        "Use default values?",
        ["Yes (recommended to start)", "No, I want to tune"],
        horizontal=True,
    )

    if use_default.startswith("Yes"):
        min_df, max_df = 2, 0.95
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            min_df = st.slider(
                "min_df (minimum proportion of documents)",
                0.0, 0.5, 0.01, step=0.01,
                help="E.g. 0.01 = words appearing in at least 1% of documents.",
            )
        with col_b:
            max_df = st.slider(
                "max_df (maximum proportion of documents)",
                0.5, 1.0, 0.95, step=0.01,
                help="E.g. 0.92 = words appearing in more than 92% of "
                     "documents are ignored.",
            )

    if st.button("🔨 Build the matrices", type="primary"):
        with st.spinner("TF-IDF + Count vectorization…"):
            try:
                m = mta.build_matrices(
                    st.session_state.corpus_wo,
                    st.session_state.stopwords,
                    min_df=min_df, max_df=max_df,
                )
                st.session_state.matrices = m
                # Invalidate downstream results, since the matrices changed
                st.session_state.metrics = None
                st.session_state.nmf_results = None
                st.session_state.lda_results = None
                st.success(
                    f"✓ Matrices built. Vocabulary kept: "
                    f"**{len(m['tf_names']):,} words**."
                )
            except ValueError as e:
                st.error(
                    "Vectorization error: your corpus is likely too small, "
                    "or the min_df/max_df filters are too strict. "
                    f"Details: {e}"
                )

    if st.session_state.matrices is not None:
        with st.expander("TF-IDF matrix preview (5 docs × 10 words)"):
            st.dataframe(
                st.session_state.matrices["df_tfidf"].iloc[:5, :10]
            )

        st.success(
            "✓ Ready! Continue with **📊 Topic models** in the left sidebar.",
            icon="➡️",
        )
