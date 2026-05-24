#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
streamlit_app.py
================

MTA — entry point. Uses st.navigation (Streamlit 1.36+) so we can
control sidebar labels, icons and ordering explicitly. This avoids
two problems of the older `pages/` auto-detection:

1. Filename emojis getting mangled (encoded as 'ð¥' etc.) in the URL
   bar of some browsers. With st.navigation, filenames are plain ASCII
   and emojis are passed as a separate `icon=` argument.
2. The fallback "streamlit app" label in the menu, which we replace
   with "MTA Menu".
"""

import streamlit as st

st.set_page_config(
    page_title="MTA — Topic Modelling",
    page_icon="📚",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Define each page explicitly. The first one is the default landing page.
# ---------------------------------------------------------------------------

home_page = st.Page(
    "home.py",
    title="Home",
    icon="🏠",
    default=True,
)
load_page = st.Page(
    "pages/1_Load_corpus.py",
    title="Load corpus",
    icon="📥",
)
models_page = st.Page(
    "pages/2_Topic_models.py",
    title="Topic models",
    icon="📊",
)
weights_page = st.Page(
    "pages/3_Word_weights.py",
    title="Word weights",
    icon="🔍",
)
evolution_page = st.Page(
    "pages/4_Topic_evolution.py",
    title="Topic evolution",
    icon="📈",
)
context_page = st.Page(
    "pages/5_Semantic_context.py",
    title="Semantic context",
    icon="🧠",
)
groups_page = st.Page(
    "pages/6_Group_comparison.py",
    title="Group comparison",
    icon="⚖️",
)
network_page = st.Page(
    "pages/7_Network_views.py",
    title="Network views",
    icon="🕸",
)

# Build the menu. The `position="sidebar"` puts it on the left, as before.
# The dict keys ("MTA Menu") replace the auto-generated section label.
pg = st.navigation(
    {
        "MTA Menu": [home_page, load_page, models_page,
                     weights_page, evolution_page, context_page,
                     groups_page, network_page],
    },
    position="sidebar",
)

pg.run()
