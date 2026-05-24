#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mta_network.py
==============

Bipartite network visualizations for MTA topic models.

Three figure builders (all pure, no I/O — return matplotlib Figures):
  • plot_topic_document_network(doctopic, labels, ...)
        Topic ↔ Document bipartite network
  • plot_topic_word_network(topicwords, vocab, ...)
        Topic ↔ top-N words bipartite network
  • plot_combined_network(doctopic, topicwords, labels, vocab, ...)
        Topics, documents (circles) and words (squares) in one view

Design choices:
  • ForceAtlas2 layout (via fa2_modified) — same algorithm as Gephi's
    Force Atlas 2, producing organic node placement; falls back to
    NetworkX spring layout if fa2_modified is not installed.
  • Solarized palette (Ethan Schoonover) for warm, publication-grade
    colors that work well on white backgrounds.
  • Curved (arc) edges — easier to follow overlapping connections.
  • Two size-encoding modes:
      - emphasize_differences=False (default): sqrt-faithful scaling.
        Topic node sizes reflect the actual mass differences; on a
        balanced corpus, nodes look similar (the encoding is honest).
      - emphasize_differences=True: min-max stretch. The smallest topic
        is rendered near s_min and the largest near s_max, so even small
        but real differences become visible. Distorts proportions.

Used by the CLI (MTA_v2.py), the Streamlit app and the batch mode.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
import networkx as nx

try:
    from fa2_modified import ForceAtlas2
    _HAS_FA2 = True
except ImportError:  # pragma: no cover
    _HAS_FA2 = False


# =============================================================================
# COLOR HELPERS — Solarized palette
# =============================================================================

# Solarized accent colors (Ethan Schoonover, https://ethanschoonover.com/solarized/)
# Chosen ordering balances contrast between adjacent topics.
_SOLARIZED_ACCENTS = [
    "#268bd2",  # blue
    "#cb4b16",  # orange
    "#859900",  # green
    "#d33682",  # magenta
    "#b58900",  # yellow
    "#6c71c4",  # violet
    "#2aa198",  # cyan
    "#dc322f",  # red
]

# Solarized base tones (for backgrounds, text)
_SOL_BASE03 = "#002b36"
_SOL_BASE02 = "#073642"
_SOL_BASE01 = "#586e75"
_SOL_BASE00 = "#657b83"
_SOL_BASE0  = "#839496"
_SOL_BASE1  = "#93a1a1"
_SOL_BASE2  = "#eee8d5"
_SOL_BASE3  = "#fdf6e3"


def _topic_palette(n_topics: int):
    """
    Return a list of distinct Solarized colors for K topics.
    Cycles through 8 accent colors; for K > 8, adjusts saturation
    to differentiate further topics.
    """
    if n_topics <= len(_SOLARIZED_ACCENTS):
        return [mcolors.to_rgba(c) for c in _SOLARIZED_ACCENTS[:n_topics]]
    # For larger K, recycle the palette with HSV variations
    out = []
    for i in range(n_topics):
        base = mcolors.to_rgba(_SOLARIZED_ACCENTS[i % len(_SOLARIZED_ACCENTS)])
        cycle = i // len(_SOLARIZED_ACCENTS)
        if cycle > 0:
            # Shift hue slightly each cycle
            h, s, v = mcolors.rgb_to_hsv(base[:3])
            h = (h + 0.07 * cycle) % 1.0
            s = max(0.4, s - 0.1 * cycle)
            r, g, b = mcolors.hsv_to_rgb((h, s, v))
            base = (r, g, b, base[3])
        out.append(base)
    return out


def _desaturate(rgba, factor: float = 0.50, lighten: float = 0.30):
    """Soft, light variant of a topic color for satellite nodes."""
    r, g, b, a = rgba
    h, s, v = mcolors.rgb_to_hsv((r, g, b))
    s *= factor
    v = min(1.0, v + lighten * (1.0 - v))
    r2, g2, b2 = mcolors.hsv_to_rgb((h, s, v))
    return (r2, g2, b2, a)


# =============================================================================
# LABEL & LAYOUT HELPERS
# =============================================================================

def _truncate(label: str, max_len: int = 24) -> str:
    """Truncate long labels with an ellipsis."""
    if len(label) <= max_len:
        return label
    keep = max_len - 1
    left = (keep + 1) // 2
    right = keep // 2
    return label[:left] + "…" + label[-right:]


def _layout_force_atlas2(
    G: nx.Graph,
    topic_node_ids: Sequence[str],
    seed: int = 42,
    iterations: int = 800,
    topic_init_radius: float = 1.0,
) -> dict:
    """
    Apply ForceAtlas2 layout, with topics initialized on a wide circle
    so the algorithm starts from a sensible global structure (otherwise
    FA2 with a random start can leave the graph in an unbalanced shape).

    Parameters tuned for bipartite topic/document graphs of medium size
    (10-200 nodes). Larger graphs may benefit from `iterations=2000+`.

    Falls back to a spring layout if fa2_modified is not available.
    """
    n_topics = len(topic_node_ids)

    # Build initial positions: topics on a circle, others near their topic
    rng = np.random.RandomState(seed)
    init_pos = {}
    for k, t_id in enumerate(topic_node_ids):
        angle = 2 * np.pi * k / n_topics + np.pi / 2
        init_pos[t_id] = (
            float(topic_init_radius * np.cos(angle)),
            float(topic_init_radius * np.sin(angle)),
        )
    for node in G.nodes():
        if node in init_pos:
            continue
        # Place near weighted centroid of topic neighbors
        topic_neighbors = [n for n in G.neighbors(node) if n in init_pos]
        if topic_neighbors:
            weights = np.array([G[node][n].get("weight", 1.0)
                                for n in topic_neighbors])
            cx = float(sum(w * init_pos[n][0]
                           for n, w in zip(topic_neighbors, weights))
                       / weights.sum())
            cy = float(sum(w * init_pos[n][1]
                           for n, w in zip(topic_neighbors, weights))
                       / weights.sum())
            jx, jy = rng.uniform(-0.05, 0.05, size=2)
            init_pos[node] = (cx + jx, cy + jy)
        else:
            init_pos[node] = tuple(rng.uniform(-0.5, 0.5, size=2))

    if not _HAS_FA2:
        # Fallback to spring layout
        pos = nx.spring_layout(G, pos=init_pos,
                               fixed=list(topic_node_ids),
                               iterations=200, seed=seed,
                               weight="weight", k=0.4)
        return pos

    # Run ForceAtlas2
    # Note: strongGravityMode=True + higher gravity keeps disconnected
    # sub-graphs (one per topic when topics share no documents) within
    # a compact area rather than spreading to the canvas edges.
    forceatlas2 = ForceAtlas2(
        # Behavior
        outboundAttractionDistribution=True,   # Dissuade hubs
        linLogMode=False,
        adjustSizes=False,
        edgeWeightInfluence=1.0,
        # Performance
        jitterTolerance=1.0,
        barnesHutOptimize=True,
        barnesHutTheta=1.2,
        multiThreaded=False,
        # Tuning — gravity tuned for bipartite topic graphs
        scalingRatio=2.0,
        strongGravityMode=True,
        gravity=4.0,
        verbose=False,
    )
    pos = forceatlas2.forceatlas2_networkx_layout(
        G, pos=init_pos, iterations=iterations,
    )

    # Rescale to a canonical [-1, 1] box (so figsize controls actual size)
    xs = np.array([p[0] for p in pos.values()])
    ys = np.array([p[1] for p in pos.values()])
    x_range = max(xs.max() - xs.min(), 1e-9)
    y_range = max(ys.max() - ys.min(), 1e-9)
    scale = 2.0 / max(x_range, y_range)
    cx, cy = (xs.max() + xs.min()) / 2, (ys.max() + ys.min()) / 2
    pos = {n: ((p[0] - cx) * scale, (p[1] - cy) * scale)
           for n, p in pos.items()}
    return pos


def _draw_topic_labels(ax, pos, topic_node_ids, topic_names,
                       font_size: float = 9.5):
    """Draw topic labels on a translucent white box for readability."""
    for k, t_id in enumerate(topic_node_ids):
        x, y = pos[t_id]
        ax.text(x, y, topic_names[k],
                fontsize=font_size, fontweight="bold",
                color=_SOL_BASE03,
                ha="center", va="center",
                bbox=dict(facecolor="white", edgecolor="none",
                          alpha=0.88, boxstyle="round,pad=0.30"),
                zorder=10)


def _draw_curved_edges(ax, pos, edges_data, alpha=0.55, rad=0.15):
    """
    Draw bent (arc) edges between node pairs.

    edges_data : list of (u, v, weight, color, width)
    """
    for u, v, _w, color, width in edges_data:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        arrow = FancyArrowPatch(
            posA=(x0, y0), posB=(x1, y1),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-",           # non-directional (no head)
            color=color, alpha=alpha,
            lw=width, zorder=2,
            shrinkA=0, shrinkB=0,
        )
        ax.add_patch(arrow)


def _weight_to_width(weights: np.ndarray, w_min: float = 0.4,
                     w_max: float = 5.0) -> np.ndarray:
    """
    Map a 1D array of edge weights to line widths, using sqrt scaling so
    differences are visible without crushing the largest weights.
    """
    if len(weights) == 0 or weights.max() <= 0:
        return np.full_like(weights, w_min, dtype=float)
    norm = np.sqrt(weights / weights.max())
    return w_min + norm * (w_max - w_min)


def _topic_sizes(topic_mass: np.ndarray, s_min: float, s_max: float,
                 emphasize_differences: bool = False) -> np.ndarray:
    """
    Convert per-topic cumulated weight mass to node sizes.

    Two modes:
      • emphasize_differences=False (default): faithful sqrt scaling.
        Sizes span s_min..s_max proportionally to sqrt(mass / mass.max()).
        On a balanced corpus, all topic nodes will look similar (because
        their masses are similar — the encoding is honest).

      • emphasize_differences=True: min-max stretch. The smallest topic
        is rendered near s_min, the largest near s_max, regardless of
        the actual ratio. Useful to spot small but real differences.
        Note: this distorts proportions; a topic twice as massive as
        another will not look twice as big — it will look "as big as it
        gets" relative to its neighbours.
    """
    if topic_mass.max() <= 0:
        return np.full_like(topic_mass, (s_min + s_max) / 2, dtype=float)

    if emphasize_differences:
        # Min-max stretch on sqrt-transformed masses (so the spread is
        # not crushed by a single very-large topic, but the smallest
        # topic still gets near s_min and largest near s_max).
        sm = np.sqrt(topic_mass)
        if sm.max() - sm.min() < 1e-12:
            return np.full_like(topic_mass, (s_min + s_max) / 2, dtype=float)
        norm = (sm - sm.min()) / (sm.max() - sm.min())
        # Leave a small safety margin so the smallest doesn't disappear
        return s_min + 0.15 * (s_max - s_min) + norm * 0.85 * (s_max - s_min)

    # Faithful mode: sqrt of normalized mass
    norm = np.sqrt(topic_mass / topic_mass.max())
    return s_min + norm * (s_max - s_min)


def _add_weight_legend(ax, title_below=False):
    """
    Tiny inline legend explaining the visual encoding.
    """
    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=_SOL_BASE01,
               markeredgecolor="black", markersize=14,
               label="Topic-Knoten — Größe = Gewichtssumme"),
        Line2D([0], [0], color=_SOL_BASE01, lw=4.0,
               label="Kante — Stärke = Verbindungs­gewicht"),
    ]
    leg = ax.legend(
        handles=handles,
        loc="lower right",
        frameon=True, framealpha=0.92,
        fancybox=True, edgecolor=_SOL_BASE1,
        fontsize=7.5, labelcolor=_SOL_BASE02,
        handletextpad=0.6, borderpad=0.6,
    )
    leg.get_frame().set_facecolor(_SOL_BASE3)
    return leg


# =============================================================================
# 1. TOPIC ↔ DOCUMENT NETWORK
# =============================================================================

def plot_topic_document_network(
    doctopic: np.ndarray,
    labels: Sequence[str],
    topic_names: Optional[Sequence[str]] = None,
    min_weight_pct: float = 0.10,
    figsize: Tuple[float, float] = (10.0, 10.0),
    seed: int = 42,
    title: Optional[str] = None,
    max_doc_label_len: int = 24,
    show_weight_legend: bool = True,
    emphasize_differences: bool = False,
) -> Figure:
    """
    Render a bipartite topic↔document network as a matplotlib figure.

    Parameters
    ----------
    doctopic : ndarray (D, K)
        Document-topic weights.
    labels : sequence of str
        Document labels.
    topic_names : sequence of str, optional
        Topic display names (default: "Topic 1", "Topic 2", …).
    min_weight_pct : float in [0, 1]
        Keep edges where the doc-topic weight is ≥ this fraction of the
        document's max topic weight.
    figsize : tuple
        Figure size in inches. Default 10×10 (square, publication-ready).
    seed : int
        Layout reproducibility.
    title : str, optional
    max_doc_label_len : int
        Truncate document labels longer than this (with an ellipsis).
    show_weight_legend : bool
        Show the small inline weight legend.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_docs, n_topics = doctopic.shape
    if topic_names is None:
        topic_names = [f"Topic {i + 1}" for i in range(n_topics)]

    row_max = doctopic.max(axis=1, keepdims=True)
    row_max[row_max == 0] = 1.0
    rel = doctopic / row_max

    G = nx.Graph()
    palette = _topic_palette(n_topics)

    topic_weight_mass = np.zeros(n_topics)
    topic_node_ids = []
    for k in range(n_topics):
        t_id = f"T{k}"
        topic_node_ids.append(t_id)
        G.add_node(t_id, kind="topic")

    doc_dominant = np.argmax(doctopic, axis=1)
    doc_node_ids = []
    edges_data = []
    for d in range(n_docs):
        connected = [k for k in range(n_topics) if rel[d, k] >= min_weight_pct]
        if not connected:
            continue
        d_id = f"D{d}"
        doc_node_ids.append(d_id)
        G.add_node(d_id, kind="doc", label=labels[d], topic=doc_dominant[d])
        for k in connected:
            w = float(doctopic[d, k])
            G.add_edge(d_id, f"T{k}", weight=w)
            topic_weight_mass[k] += w

    if G.number_of_edges() == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5,
                "Kein Dokument erreicht die Mindestgewichtschwelle.\n"
                f"Aktueller Schwellenwert: {min_weight_pct:.0%}.\n"
                "Senken Sie 'min_weight_pct'.",
                ha="center", va="center", fontsize=11, color=_SOL_BASE01)
        ax.axis("off")
        return fig

    # Layout
    pos = _layout_force_atlas2(G, topic_node_ids, seed=seed)

    # Render
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Prepare edges with colors and widths
    edge_weights = np.array([G[u][v]["weight"]
                             for u, v in G.edges()])
    edge_widths = _weight_to_width(edge_weights, w_min=0.5, w_max=4.0)
    edge_drawables = []
    for (u, v), w, ew in zip(G.edges(), edge_weights, edge_widths):
        # Topic id is the one starting with T
        t_id = u if u.startswith("T") else v
        topic_idx = int(t_id[1:])
        edge_drawables.append((u, v, w, palette[topic_idx], ew))

    _draw_curved_edges(ax, pos, edge_drawables, alpha=0.55, rad=0.15)

    # Document nodes — slightly larger than v1, with stronger edge
    doc_colors = [_desaturate(palette[G.nodes[n]["topic"]])
                  for n in doc_node_ids]
    nx.draw_networkx_nodes(
        G, pos, nodelist=doc_node_ids,
        node_size=120, node_color=doc_colors,
        edgecolors=_SOL_BASE01, linewidths=0.7,
        ax=ax,
    )

    # Topic nodes — sized by attached weight mass
    topic_sizes = _topic_sizes(topic_weight_mass, s_min=600, s_max=3500,
                               emphasize_differences=emphasize_differences)
    topic_colors = [palette[k] for k in range(n_topics)]
    nx.draw_networkx_nodes(
        G, pos, nodelist=topic_node_ids,
        node_size=topic_sizes, node_color=topic_colors,
        edgecolors=_SOL_BASE03, linewidths=1.5,
        ax=ax,
    )

    # Document labels — offset slightly below node
    doc_labels = {n: _truncate(G.nodes[n]["label"], max_doc_label_len)
                  for n in doc_node_ids}
    label_pos = {n: (pos[n][0], pos[n][1] - 0.04) for n in doc_node_ids}
    nx.draw_networkx_labels(
        G, label_pos, labels=doc_labels,
        font_size=6.8, font_color=_SOL_BASE01, ax=ax,
    )

    # Topic labels
    _draw_topic_labels(ax, pos, topic_node_ids, topic_names, font_size=10.0)

    if title:
        ax.set_title(title, fontsize=12, pad=14, color=_SOL_BASE02)

    if show_weight_legend:
        _add_weight_legend(ax)

    # Padding so labels and legend stay inside
    ax.margins(0.10)

    fig.tight_layout()
    return fig


# =============================================================================
# 2. TOPIC ↔ TOP-N WORDS NETWORK
# =============================================================================

def plot_topic_word_network(
    topicwords: np.ndarray,
    vocab: Sequence[str],
    topic_names: Optional[Sequence[str]] = None,
    top_n: int = 50,
    figsize: Tuple[float, float] = (10.0, 10.0),
    seed: int = 42,
    title: Optional[str] = None,
    show_weight_legend: bool = True,
    emphasize_differences: bool = False,
) -> Figure:
    """
    Bipartite topic↔(top-N words) network.

    For each topic, the top_n words by weight are added; a word can be
    linked to several topics if it ranks high in more than one.
    """
    n_topics, vocab_size = topicwords.shape
    if topic_names is None:
        topic_names = [f"Topic {i + 1}" for i in range(n_topics)]
    top_n = min(top_n, vocab_size)

    palette = _topic_palette(n_topics)

    word_to_topics: dict = {}
    for k in range(n_topics):
        top_idx = np.argsort(topicwords[k])[::-1][:top_n]
        for j in top_idx:
            w = float(topicwords[k, j])
            if w <= 0:
                continue
            word_to_topics.setdefault(vocab[j], []).append((k, w))

    G = nx.Graph()
    topic_weight_mass = np.zeros(n_topics)
    topic_node_ids = []
    for k in range(n_topics):
        t_id = f"T{k}"
        topic_node_ids.append(t_id)
        G.add_node(t_id, kind="topic")

    word_node_ids = []
    for word, links in word_to_topics.items():
        dom = max(links, key=lambda x: x[1])[0]
        w_id = f"W::{word}"
        word_node_ids.append(w_id)
        G.add_node(w_id, kind="word", label=word, topic=dom)
        for k, w in links:
            G.add_edge(w_id, f"T{k}", weight=w)
            topic_weight_mass[k] += w

    if G.number_of_edges() == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Keine Wörter mit positiven Topic-Gewichten.",
                ha="center", va="center", fontsize=11, color=_SOL_BASE01)
        ax.axis("off")
        return fig

    pos = _layout_force_atlas2(G, topic_node_ids, seed=seed,
                                iterations=1000)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    edge_weights = np.array([G[u][v]["weight"] for u, v in G.edges()])
    edge_widths = _weight_to_width(edge_weights, w_min=0.4, w_max=3.5)
    edge_drawables = []
    for (u, v), w, ew in zip(G.edges(), edge_weights, edge_widths):
        t_id = u if u.startswith("T") else v
        topic_idx = int(t_id[1:])
        edge_drawables.append((u, v, w, palette[topic_idx], ew))

    _draw_curved_edges(ax, pos, edge_drawables, alpha=0.50, rad=0.15)

    # Word nodes — size also encodes max link strength (small but visible)
    word_max_w = {n: max(G[n][nb]["weight"] for nb in G.neighbors(n))
                  for n in word_node_ids}
    max_global = max(word_max_w.values()) if word_max_w else 1.0
    word_sizes = [60 + (word_max_w[n] / max_global) ** 0.5 * 90
                  for n in word_node_ids]
    word_colors = [_desaturate(palette[G.nodes[n]["topic"]])
                   for n in word_node_ids]
    nx.draw_networkx_nodes(
        G, pos, nodelist=word_node_ids,
        node_size=word_sizes, node_color=word_colors,
        edgecolors=_SOL_BASE01, linewidths=0.5,
        ax=ax,
    )

    # Topic nodes
    topic_sizes = _topic_sizes(topic_weight_mass, s_min=700, s_max=3500,
                               emphasize_differences=emphasize_differences)
    nx.draw_networkx_nodes(
        G, pos, nodelist=topic_node_ids,
        node_size=topic_sizes,
        node_color=[palette[k] for k in range(n_topics)],
        edgecolors=_SOL_BASE03, linewidths=1.5,
        ax=ax,
    )

    # Word labels — offset below
    word_labels = {n: G.nodes[n]["label"] for n in word_node_ids}
    label_pos = {n: (pos[n][0], pos[n][1] - 0.03) for n in word_node_ids}
    nx.draw_networkx_labels(
        G, label_pos, labels=word_labels,
        font_size=6.8, font_color=_SOL_BASE01, ax=ax,
    )

    _draw_topic_labels(ax, pos, topic_node_ids, topic_names, font_size=10.0)

    if title:
        ax.set_title(title, fontsize=12, pad=14, color=_SOL_BASE02)

    if show_weight_legend:
        _add_weight_legend(ax)

    ax.margins(0.10)
    fig.tight_layout()
    return fig


# =============================================================================
# 3. COMBINED NETWORK
# =============================================================================

def plot_combined_network(
    doctopic: np.ndarray,
    topicwords: np.ndarray,
    labels: Sequence[str],
    vocab: Sequence[str],
    topic_names: Optional[Sequence[str]] = None,
    top_n_words: int = 25,
    min_doc_weight_pct: float = 0.20,
    figsize: Tuple[float, float] = (12.0, 12.0),
    seed: int = 42,
    title: Optional[str] = None,
    max_doc_label_len: int = 22,
    show_weight_legend: bool = True,
    emphasize_differences: bool = False,
) -> Figure:
    """
    3-mode network: topics, documents (circles), top-N words (squares).
    """
    n_docs, n_topics = doctopic.shape
    if topic_names is None:
        topic_names = [f"Topic {i + 1}" for i in range(n_topics)]
    palette = _topic_palette(n_topics)

    G = nx.Graph()
    topic_weight_mass = np.zeros(n_topics)
    topic_node_ids = []
    for k in range(n_topics):
        t_id = f"T{k}"
        topic_node_ids.append(t_id)
        G.add_node(t_id, kind="topic")

    # Documents
    row_max = doctopic.max(axis=1, keepdims=True)
    row_max[row_max == 0] = 1.0
    rel = doctopic / row_max
    doc_dominant = np.argmax(doctopic, axis=1)
    doc_node_ids = []
    for d in range(n_docs):
        connected = [k for k in range(n_topics)
                     if rel[d, k] >= min_doc_weight_pct]
        if not connected:
            continue
        d_id = f"D{d}"
        doc_node_ids.append(d_id)
        G.add_node(d_id, kind="doc", label=labels[d],
                   topic=doc_dominant[d])
        for k in connected:
            w = float(doctopic[d, k])
            G.add_edge(d_id, f"T{k}", weight=w, kind="doc")
            topic_weight_mass[k] += w

    # Words
    top_n_words = min(top_n_words, topicwords.shape[1])
    word_to_topics: dict = {}
    for k in range(n_topics):
        top_idx = np.argsort(topicwords[k])[::-1][:top_n_words]
        for j in top_idx:
            w = float(topicwords[k, j])
            if w <= 0:
                continue
            word_to_topics.setdefault(vocab[j], []).append((k, w))

    word_node_ids = []
    for word, links in word_to_topics.items():
        dom = max(links, key=lambda x: x[1])[0]
        w_id = f"W::{word}"
        word_node_ids.append(w_id)
        G.add_node(w_id, kind="word", label=word, topic=dom)
        for k, w in links:
            G.add_edge(w_id, f"T{k}", weight=w, kind="word")
            topic_weight_mass[k] += w * 0.5

    if G.number_of_edges() == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Graph leer.", ha="center", va="center",
                fontsize=11, color=_SOL_BASE01)
        ax.axis("off")
        return fig

    pos = _layout_force_atlas2(G, topic_node_ids, seed=seed,
                                iterations=1200)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Split edges by kind for differential styling
    doc_edges = []
    word_edges = []
    for u, v in G.edges():
        w = G[u][v]["weight"]
        kind = G[u][v].get("kind")
        t_id = u if u.startswith("T") else v
        topic_idx = int(t_id[1:])
        item = (u, v, w, palette[topic_idx])
        if kind == "doc":
            doc_edges.append(item)
        else:
            word_edges.append(item)

    if word_edges:
        weights = np.array([w for _, _, w, _ in word_edges])
        widths = _weight_to_width(weights, w_min=0.3, w_max=2.8)
        drawables = [(u, v, w, c, ew)
                     for (u, v, w, c), ew in zip(word_edges, widths)]
        _draw_curved_edges(ax, pos, drawables, alpha=0.40, rad=0.18)

    if doc_edges:
        weights = np.array([w for _, _, w, _ in doc_edges])
        widths = _weight_to_width(weights, w_min=0.5, w_max=4.0)
        drawables = [(u, v, w, c, ew)
                     for (u, v, w, c), ew in zip(doc_edges, widths)]
        _draw_curved_edges(ax, pos, drawables, alpha=0.65, rad=0.12)

    # Document nodes (circles)
    doc_colors = [_desaturate(palette[G.nodes[n]["topic"]])
                  for n in doc_node_ids]
    nx.draw_networkx_nodes(
        G, pos, nodelist=doc_node_ids,
        node_size=140, node_color=doc_colors,
        edgecolors=_SOL_BASE01, linewidths=0.7,
        node_shape="o", ax=ax,
    )
    # Word nodes (squares)
    word_colors = [_desaturate(palette[G.nodes[n]["topic"]])
                   for n in word_node_ids]
    nx.draw_networkx_nodes(
        G, pos, nodelist=word_node_ids,
        node_size=75, node_color=word_colors,
        edgecolors=_SOL_BASE01, linewidths=0.5,
        node_shape="s", ax=ax,
    )

    # Topic nodes
    topic_sizes = _topic_sizes(topic_weight_mass, s_min=800, s_max=3800,
                               emphasize_differences=emphasize_differences)
    nx.draw_networkx_nodes(
        G, pos, nodelist=topic_node_ids,
        node_size=topic_sizes,
        node_color=[palette[k] for k in range(n_topics)],
        edgecolors=_SOL_BASE03, linewidths=1.6,
        ax=ax,
    )

    # Labels
    doc_labels = {n: _truncate(G.nodes[n]["label"], max_doc_label_len)
                  for n in doc_node_ids}
    nx.draw_networkx_labels(
        G, {n: (pos[n][0], pos[n][1] - 0.045) for n in doc_node_ids},
        labels=doc_labels, font_size=6.8, font_color=_SOL_BASE01, ax=ax,
    )
    for n in word_node_ids:
        x, y = pos[n]
        ax.text(x, y - 0.030, G.nodes[n]["label"],
                fontsize=6.3, color=_SOL_BASE00,
                ha="center", va="center", style="italic")

    _draw_topic_labels(ax, pos, topic_node_ids, topic_names, font_size=10.5)

    # Three-symbol legend (kind of nodes) + weight legend
    kind_legend = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=_SOL_BASE01, markeredgecolor=_SOL_BASE03,
               markersize=12, label="Topic"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=_SOL_BASE1, markeredgecolor=_SOL_BASE01,
               markersize=8, label="Dokument"),
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor=_SOL_BASE1, markeredgecolor=_SOL_BASE01,
               markersize=7, label="Wort"),
    ]
    leg1 = ax.legend(handles=kind_legend, loc="lower left",
                     frameon=True, framealpha=0.92, fancybox=True,
                     edgecolor=_SOL_BASE1, fontsize=8.5,
                     labelcolor=_SOL_BASE02)
    leg1.get_frame().set_facecolor(_SOL_BASE3)
    ax.add_artist(leg1)

    if show_weight_legend:
        _add_weight_legend(ax)

    if title:
        ax.set_title(title, fontsize=12, pad=14, color=_SOL_BASE02)

    ax.margins(0.10)
    fig.tight_layout()
    return fig
