#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mta_core.py
===========

Core logic of MTA (Multi-Text Analyser), extracted from MTA.py and
reorganized as pure functions, with no input(), print() or savefig().

- All inputs are passed as arguments.
- All outputs are returned (DataFrames, matplotlib figures, lists).
- No global variables, no disk side effects.

This module is meant to be called either from MTA.py (CLI, unchanged)
or from streamlit_app.py (web UI).

Author: refactoring for teaching purposes, based on MTA.py
by C. Papilloud (https://github.com/cp1972/mta-app).
"""

import re
import warnings
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn import decomposition
from sklearn.decomposition import LatentDirichletAllocation

from scipy.cluster.hierarchy import cophenet, linkage
from scipy.spatial.distance import pdist

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
warnings.filterwarnings("ignore", category=FutureWarning, module='sklearn')
np.seterr(divide='ignore', invalid='ignore')


# =============================================================================
# 0. CHART HELPERS — used by every chart function below AND by the CLI
#    (MTA_v3.py) AND by the Streamlit PNG export, so that long filenames
#    don't break the layout. Defined first because plot_topic_distribution
#    references DEFAULT_LABEL_MAXLEN as a default argument value.
# =============================================================================

DEFAULT_LABEL_MAXLEN = 30


def truncate_label(label: str, max_len: int = DEFAULT_LABEL_MAXLEN) -> str:
    """
    Shorten a long label by replacing the middle with '…', keeping the
    beginning and the end visible. Useful for filenames that share a
    common prefix (date) AND a common suffix (.txt extension) — both
    are preserved.

    >>> truncate_label('2020-03-15_die_zeit_lockdown_beginnt_in_sachsen.txt')
    '2020-03-15_die_zei…_in_sachsen.txt'

    Short labels are returned unchanged.
    """
    if len(label) <= max_len:
        return label
    keep = max_len - 1  # 1 char for the ellipsis
    left_len = (keep + 1) // 2
    right_len = keep // 2
    return label[:left_len] + "…" + label[-right_len:]


def truncate_labels(labels, max_len: int = DEFAULT_LABEL_MAXLEN):
    """Vectorised version of truncate_label for a list/Index."""
    return [truncate_label(str(l), max_len) for l in labels]


def auto_figsize(n_documents: int, base_width: float = 8.0,
                 base_height: float = 5.0,
                 width_per_doc: float = 0.20,
                 max_width: float = 24.0,
                 max_height: float = 7.0) -> tuple:
    """
    Compute a matplotlib figsize that scales with the number of documents.
    Width grows linearly with the number of documents, capped at max_width.
    """
    width = min(max_width, max(base_width, n_documents * width_per_doc))
    height = base_height if n_documents <= 60 else min(max_height, base_height * 1.2)
    return (width, height)


def wrap_ylabel(text: str, max_chars_per_line: int = 22) -> str:
    """
    Insert a newline into a long Y-axis label so matplotlib doesn't have
    to compress the figure horizontally. Tries to break at a space close
    to the middle.
    """
    if len(text) <= max_chars_per_line:
        return text
    mid = len(text) // 2
    best = -1
    for i, ch in enumerate(text):
        if ch == " " and abs(i - mid) < abs(best - mid):
            best = i
    if best < 0:
        return text
    return text[:best] + "\n" + text[best + 1:]


# =============================================================================
# 1. PREPROCESSING
# =============================================================================

def _remove_stopwords(corpus: List[str], stops: List[str]) -> List[str]:
    """Remove stopwords word by word, case-insensitive."""
    out = []
    for sentence in corpus:
        tmp = [w for w in sentence.lower().split() if w.strip('"') not in stops]
        out.append(' '.join(tmp))
    return out


def _remove_digits(lst: List[str]) -> List[str]:
    return [re.sub(r'[0-9]', '', i) for i in lst]


def _remove_urls(lst: List[str]) -> List[str]:
    pattern = (r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)'''
               r'''(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+'''
               r'''(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|'''
               r'''[^\s`!()\[\]{};:'".,<>?«»""''])'''
               r''')''')
    return [re.sub(pattern, '', i) for i in lst]


def _remove_dots(lst: List[str]) -> List[str]:
    return [re.sub(r'\.\.+', ' ', i) for i in lst]


def _remove_extra_spaces(lst: List[str]) -> List[str]:
    return [re.sub(r' +', ' ', i) for i in lst]


def preprocess_corpus(
    raw_texts: List[str],
    stopwords: List[str],
    min_word_length: int = 3,
) -> Tuple[List[str], List[str]]:
    """
    Apply the MTA cleaning pipeline to a list of raw texts.

    Parameters
    ----------
    raw_texts : list[str]
        Content of the corpus files (one element = one document).
    stopwords : list[str]
        List of stopwords to remove.
    min_word_length : int
        Minimum length of a word to keep (2 to 9 in the original MTA).

    Returns
    -------
    corpus_wo : list[str]
        Cleaned corpus, ready for vectorization (TF-IDF, CountVectorizer).
    corpus_re : list[str]
        Minimally cleaned corpus, keeping punctuation. Useful for
        extracting representative sentences per topic.
    """
    # Progressive cleaning as in the original MTA.py
    c = _remove_stopwords(raw_texts, stopwords)
    c = _remove_digits(c)
    c = _remove_urls(c)
    c = _remove_dots(c)

    # Second pass for stopwords + residual URLs
    c = [
        ' '.join(
            w for w in sent.split()
            if not (w.startswith('www') or w.startswith('http'))
            and w not in stopwords
        )
        for sent in c
    ]
    c = _remove_extra_spaces(c)

    # Remove punctuation then filter by word length
    c_no_punct = [re.sub(r'\W+', ' ', i) for i in c]
    corpus_wo = [
        " ".join(w for w in sent.split(" ") if len(w) >= int(min_word_length))
        for sent in c_no_punct
    ]

    # Corpus preserving punctuation, for sentence extraction
    corpus_re = [re.sub(r"[^a-zA-Z'.,;:!?-]+", ' ', i) for i in raw_texts]

    return corpus_wo, corpus_re


# =============================================================================
# 2. TF-IDF AND COUNT MATRICES
# =============================================================================

def build_matrices(
    corpus_wo: List[str],
    stopwords: List[str],
    min_df: float = 2,
    max_df: float = 0.95,
) -> Dict:
    """
    Build the TF-IDF matrix (for NMF) and the Count matrix (for LDA).

    The min_df and max_df parameters follow the sklearn convention:
    - integers >= 1: absolute number of documents
    - floats 0..1: proportion of documents

    Returns
    -------
    dict with the following keys:
        tf_matrix     : sparse TF-IDF matrix for NMF
        tf_names      : feature names (words)
        dense_a       : dense TF-IDF matrix (np.array) for KMeans
        lda_matrix    : sparse Count matrix for LDA
        lda_names     : feature names (words) for LDA
        df_tfidf      : pandas TF-IDF DataFrame (documents x words), for inspection
    """
    tfidf_vec = TfidfVectorizer(
        min_df=min_df, max_df=max_df,
        encoding='utf-8', analyzer='word',
        ngram_range=(1, 1), stop_words=stopwords,
    )
    lda_vec = CountVectorizer(
        min_df=min_df, max_df=max_df,
        encoding='utf-8', analyzer='word',
        ngram_range=(1, 1), stop_words=stopwords,
    )

    tf_matrix = tfidf_vec.fit_transform(corpus_wo)
    tf_names = tfidf_vec.get_feature_names_out()
    dense_a = np.asarray(tf_matrix.todense())

    lda_matrix = lda_vec.fit_transform(corpus_wo)
    lda_names = lda_vec.get_feature_names_out()

    df_tfidf = pd.DataFrame(dense_a, columns=tf_names)

    return {
        "tf_matrix": tf_matrix,
        "tf_names": tf_names,
        "dense_a": dense_a,
        "lda_matrix": lda_matrix,
        "lda_names": lda_names,
        "df_tfidf": df_tfidf,
    }


# =============================================================================
# 3. CROSS-VALIDATION METRICS
# =============================================================================

def _turning_points(lst):
    """Indices of turning points (local minima or maxima)."""
    out = []
    for i in range(1, len(lst) - 1):
        if (lst[i - 1] > lst[i] < lst[i + 1]) or (lst[i - 1] < lst[i] > lst[i + 1]):
            out.append(i)
    return out


def compute_topic_metrics(
    tf_matrix,
    lda_matrix,
    dense_a: np.ndarray,
    max_topics: int = 10,
    progress_callback=None,
) -> Dict:
    """
    Compute cross-validation metrics to suggest an optimal number of
    topics: Elbow, Silhouette, Calinski-Harabasz, Davies-Bouldin
    (KMeans++), and Cophenet (NMF and LDA).

    Parameters
    ----------
    progress_callback : callable, optional
        Function called with (i, total, label) to report progress.
        Allows Streamlit to display st.progress().

    Returns
    -------
    dict with all the curves and the suggested turning points.
    """
    num_c = max_topics + 1
    ks = list(range(2, num_c))
    X_scaled = dense_a.T

    km_elbow, km_silhouette, km_calinski, km_bouldin = {}, {}, {}, {}
    coph_corr_nmf, coph_corr_lda = [], []

    # KMeans++ on the 4 metrics
    for idx, i in enumerate(ks):
        km = KMeans(n_clusters=i, random_state=0, init="k-means++", n_init=10).fit(X_scaled)
        km_elbow[i] = km.inertia_
        km_silhouette[i] = silhouette_score(X_scaled, km.fit_predict(X_scaled))
        km_calinski[i] = calinski_harabasz_score(X_scaled, km.labels_)
        km_bouldin[i] = davies_bouldin_score(X_scaled, km.labels_)
        if progress_callback:
            progress_callback(idx + 1, len(ks) * 3, "KMeans++ metrics")

    # NMF Cophenet
    for idx, i in enumerate(ks):
        nmf = decomposition.NMF(n_components=i, random_state=1, init='nndsvd', max_iter=400)
        doctopic = nmf.fit_transform(tf_matrix)
        link = linkage(doctopic, 'ward')
        coph_corr_nmf.append(cophenet(link, pdist(doctopic))[0])
        if progress_callback:
            progress_callback(len(ks) + idx + 1, len(ks) * 3, "NMF Cophenet")

    # LDA Cophenet
    for idx, i in enumerate(ks):
        lda = LatentDirichletAllocation(
            n_components=i, evaluate_every=-1,
            learning_method='online', n_jobs=-1,
            learning_offset=50., random_state=100, batch_size=128,
        )
        doctopic_lda = lda.fit_transform(lda_matrix)
        link = linkage(doctopic_lda, 'ward')
        coph_corr_lda.append(cophenet(link, pdist(doctopic_lda))[0])
        if progress_callback:
            progress_callback(2 * len(ks) + idx + 1, len(ks) * 3, "LDA Cophenet")

    # Turning points, translated into topic counts (+2 because ks starts at 2)
    suggestions = {
        "Elbow":             [i + 2 for i in _turning_points(list(km_elbow.values()))],
        "Silhouette":        [i + 2 for i in _turning_points(list(km_silhouette.values()))],
        "Calinski-Harabasz": [i + 2 for i in _turning_points(list(km_calinski.values()))],
        "Davies-Bouldin":    [i + 2 for i in _turning_points(list(km_bouldin.values()))],
        "Cophenet NMF":      [i + 2 for i in _turning_points(coph_corr_nmf)],
        "Cophenet LDA":      [i + 2 for i in _turning_points(coph_corr_lda)],
    }

    return {
        "ks": ks,
        "elbow": km_elbow,
        "silhouette": km_silhouette,
        "calinski": km_calinski,
        "bouldin": km_bouldin,
        "cophenet_nmf": dict(zip(ks, coph_corr_nmf)),
        "cophenet_lda": dict(zip(ks, coph_corr_lda)),
        "suggestions": suggestions,
    }


def plot_metrics(metrics: Dict) -> plt.Figure:
    """Create the 2x3 summary figure of metrics."""
    fig, axs = plt.subplots(2, 3, sharex=True, figsize=(11, 6))

    def _scat(ax, d, title, color):
        ax.scatter(list(d.keys()), list(d.values()), s=18, edgecolor=color, alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel('Number of topics')
        ax.set_ylabel('Score')

    _scat(axs[0, 0], metrics["elbow"],        "Elbow",             '#b58900')
    _scat(axs[0, 1], metrics["silhouette"],   "Silhouette",        '#cb4b16')
    _scat(axs[0, 2], metrics["cophenet_nmf"], "NMF Cophenet",      '#268bd2')
    _scat(axs[1, 0], metrics["calinski"],     "Calinski-Harabasz", '#dc322f')
    _scat(axs[1, 1], metrics["bouldin"],      "Davies-Bouldin",    '#d33682')
    _scat(axs[1, 2], metrics["cophenet_lda"], "LDA Cophenet",      '#2aa198')
    fig.tight_layout()
    return fig


# =============================================================================
# 4. NMF AND LDA MODELS
# =============================================================================

def run_nmf(tf_matrix, n_topics: int) -> Dict:
    """Train an NMF model and return its main components."""
    nmf = decomposition.NMF(n_components=n_topics, random_state=1, init='nndsvd', max_iter=400)
    doctopic = nmf.fit_transform(tf_matrix)
    topicwords = nmf.components_
    link = linkage(doctopic, 'ward')
    coph = cophenet(link, pdist(doctopic))[0]
    return {
        "model": nmf,
        "doctopic": doctopic,
        "topicwords": topicwords,
        "linkage": link,
        "cophenet": coph,
    }


def run_lda(lda_matrix, n_topics: int) -> Dict:
    """Train an LDA model and return its main components."""
    lda = LatentDirichletAllocation(
        n_components=n_topics, evaluate_every=-1,
        learning_method='online', n_jobs=-1,
        learning_offset=50., random_state=100, batch_size=128,
    )
    doctopic = lda.fit_transform(lda_matrix)
    topicwords = lda.components_
    link = linkage(doctopic, 'ward')
    coph = cophenet(link, pdist(doctopic))[0]
    return {
        "model": lda,
        "doctopic": doctopic,
        "topicwords": topicwords,
        "linkage": link,
        "cophenet": coph,
    }


# =============================================================================
# 5. EXPLOITING TOPICS
# =============================================================================

def top_words_per_topic(
    topicwords: np.ndarray,
    feature_names: np.ndarray,
    n_words: int = 20,
) -> pd.DataFrame:
    """
    Return a DataFrame (n_words rows x n_topics columns) with the most
    representative words of each topic.
    """
    rows = []
    for topic in topicwords:
        idx_sorted = np.argsort(topic)[::-1][:n_words]
        rows.append([feature_names[i] for i in idx_sorted])
    df = pd.DataFrame(rows).T  # transpose to have words as rows
    df.columns = [f"Topic_{i}" for i in range(df.shape[1])]
    return df


def topic_distribution_per_doc(
    doctopic: np.ndarray,
    doc_labels: List[str],
) -> pd.DataFrame:
    """
    Normalize the topic weights within each document (sum = 1) and
    return a labelled DataFrame (documents x topics).
    """
    norm = doctopic / np.sum(doctopic, axis=1, keepdims=True)
    df = pd.DataFrame(
        norm,
        index=doc_labels,
        columns=[f"Topic_{i}" for i in range(norm.shape[1])],
    )
    return df


def dominant_topic_per_doc(distribution: pd.DataFrame) -> pd.DataFrame:
    """For each document, return the dominant topic and its weight."""
    dom = distribution.idxmax(axis=1)
    weight = distribution.max(axis=1)
    return pd.DataFrame({
        "Document": distribution.index,
        "Dominant topic": dom.values,
        "Weight": weight.values,
    })


# =============================================================================
# 5b. WORD-LEVEL WEIGHTS (used by Step 5 "Word weight analysis")
# =============================================================================

def words_weight_per_topic(
    topicwords: np.ndarray,
    feature_names: np.ndarray,
    words: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    For each requested word, return its weight in every topic.

    The lookup is performed on the FULL topic-word matrix (model.components_),
    not just on the top-N most representative words — so the user sees the
    word's contribution to every topic, even very small ones.

    Search is case-insensitive (consistent with the lower-cased preprocessing
    pipeline used in MTA).

    Parameters
    ----------
    topicwords : np.ndarray
        Shape (n_topics, n_features). model.components_ for NMF or LDA.
    feature_names : np.ndarray
        Vocabulary, length n_features.
    words : list[str]
        Words requested by the user (any case).

    Returns
    -------
    df : pd.DataFrame
        Index = words found (lower-cased), columns = Topic_0, Topic_1, ...,
        values = weights in topicwords. Empty DataFrame if no word is found.
    not_found : list[str]
        Words that are not in the vocabulary.
    """
    # Normalize for case-insensitive lookup
    vocab = {w: i for i, w in enumerate(feature_names)}
    rows = {}
    not_found = []
    for w in words:
        w_norm = w.strip().lower()
        if not w_norm:
            continue
        if w_norm in vocab:
            idx = vocab[w_norm]
            rows[w_norm] = topicwords[:, idx]
        else:
            not_found.append(w_norm)

    if not rows:
        return pd.DataFrame(), not_found

    df = pd.DataFrame(
        rows,
        index=[f"Topic_{i}" for i in range(topicwords.shape[0])],
    ).T  # words on rows, topics on columns
    df.index.name = "Word"
    return df, not_found


def words_weight_per_document(
    matrix,
    feature_names: np.ndarray,
    doc_labels: List[str],
    words: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    For each requested word, return its weight in every document.

    Uses the full term-document matrix (TF-IDF for NMF context, or Count
    for LDA context — caller chooses which one to pass). Search is
    case-insensitive.

    Parameters
    ----------
    matrix : sparse matrix
        Shape (n_documents, n_features). tf_matrix or lda_matrix from
        build_matrices().
    feature_names : np.ndarray
        Vocabulary, length n_features.
    doc_labels : list[str]
        Document filenames, length n_documents.
    words : list[str]
        Words requested by the user.

    Returns
    -------
    df : pd.DataFrame
        Index = document labels, columns = words found, values = weights.
        Empty DataFrame if no word is found.
    not_found : list[str]
        Words not in the vocabulary.
    """
    vocab = {w: i for i, w in enumerate(feature_names)}
    cols = {}
    not_found = []
    for w in words:
        w_norm = w.strip().lower()
        if not w_norm:
            continue
        if w_norm in vocab:
            idx = vocab[w_norm]
            # Slice the column and densify just that one column (memory-safe)
            cols[w_norm] = np.asarray(matrix[:, idx].todense()).ravel()
        else:
            not_found.append(w_norm)

    if not cols:
        return pd.DataFrame(), not_found

    df = pd.DataFrame(cols, index=doc_labels)
    df.index.name = "Document"
    return df, not_found


def best_sentences_per_topic(
    topic_words_df: pd.DataFrame,
    corpus_re: List[str],
    min_matches: int = 3,
    top_words_to_match: int = 50,
) -> pd.DataFrame:
    """
    For each topic, retrieve sentences from the corpus containing at
    least `min_matches` of the `top_words_to_match` topic keywords.

    Return a DataFrame sorted by topic then by descending frequency.
    """
    # Reconstruct the list-of-lists structure used by the original MTA
    n_topics = topic_words_df.shape[1]
    topic_words = [topic_words_df.iloc[:, t].tolist()[:top_words_to_match]
                   for t in range(n_topics)]

    sentences = [s for doc in corpus_re for s in re.split(r'[?.!…]', doc) if s.strip()]

    matches = []
    for t in range(n_topics):
        kw_set = set(topic_words[t])
        for s in sentences:
            words_in_s = set(s.lower().split())
            n_match = len(kw_set & words_in_s)
            if n_match >= min_matches:
                matches.append((t, n_match, s.strip().replace('\n', ' ')))

    df = pd.DataFrame(matches, columns=['Topic', 'Frequency', 'Sentence'])
    df = df.sort_values(by=['Topic', 'Frequency'], ascending=[True, False])
    return df.reset_index(drop=True)


# =============================================================================
# 6. PLOTS
# =============================================================================

# Labels available in 3 languages, used by every chart label across the app.
# Keys are stable identifiers used in the UI code; values are the translations.
# This dictionary is the single source of truth for chart-text translation:
# both the Streamlit pages and the CLI MTA.py can read from it.
_LABELS = {
    "en": {
        # generic axes
        "documents":         "Documents",
        "documents_sorted":  "Documents (sorted alphabetically)",
        "topics":            "Topics",
        "topic":             "Topic",
        "weight":            "Weight",
        "weight_of_topics":  "Weight of topics",
        "weight_rm":         "Rolling-mean weight of topics",
        "number_of_topics":  "Number of topics",
        "score":             "Score",
        "year":              "Year",
        "yearly_weight":     "Mean weight of topic (per year)",
        "word":              "Word",
        "words":             "Words",
        "similar_word":      "Similar word",
        "similarity":        "Cosine similarity",
        "rank":              "Rank",
        "query_word":        "Query word",
        "frequency":         "Frequency",
        "sentence":          "Sentence",
        "dominant_topic":    "Dominant topic",
        # composite titles
        "distr":             "Distribution of topics across texts (%)",
        "other_topics":      "Other topics (sum of {n})",
        "nmf_topics":        "NMF topics",
        "lda_topics":        "LDA topics",
        "jaccard":           "Jaccard",
    },
    "fr": {
        "documents":         "Documents",
        "documents_sorted":  "Documents (triés alphabétiquement)",
        "topics":            "Topics",
        "topic":             "Topic",
        "weight":            "Poids",
        "weight_of_topics":  "Poids des topics",
        "weight_rm":         "Poids moyen mobile des topics",
        "number_of_topics":  "Nombre de topics",
        "score":             "Score",
        "year":              "Année",
        "yearly_weight":     "Poids moyen du topic (par année)",
        "word":              "Mot",
        "words":             "Mots",
        "similar_word":      "Mot similaire",
        "similarity":        "Similarité cosinus",
        "rank":              "Rang",
        "query_word":        "Mot recherché",
        "frequency":         "Fréquence",
        "sentence":          "Phrase",
        "dominant_topic":    "Topic dominant",
        "distr":             "Distribution des topics dans les textes (%)",
        "other_topics":      "Autres topics (somme de {n})",
        "nmf_topics":        "Topics NMF",
        "lda_topics":        "Topics LDA",
        "jaccard":           "Jaccard",
    },
    "de": {
        "documents":         "Texte",
        "documents_sorted":  "Texte (alphabetisch sortiert)",
        "topics":            "Topics",
        "topic":             "Topic",
        "weight":            "Gewicht",
        "weight_of_topics":  "Gewicht der Topics",
        "weight_rm":         "Rollender Mittelwert der Topic-Gewichte",
        "number_of_topics":  "Anzahl Topics",
        "score":             "Wert",
        "year":              "Jahr",
        "yearly_weight":     "Mittleres Gewicht des Topics (pro Jahr)",
        "word":              "Wort",
        "words":             "Wörter",
        "similar_word":      "Ähnliches Wort",
        "similarity":        "Kosinus-Ähnlichkeit",
        "rank":              "Rang",
        "query_word":        "Suchwort",
        "frequency":         "Häufigkeit",
        "sentence":          "Satz",
        "dominant_topic":    "Dominantes Topic",
        "distr":             "Verteilung der Topics in den Texten (%)",
        "other_topics":      "Andere Topics (Summe von {n})",
        "nmf_topics":        "NMF-Topics",
        "lda_topics":        "LDA-Topics",
        "jaccard":           "Jaccard",
    },
}


def get_labels(language: str = "en") -> Dict[str, str]:
    """
    Return the label dictionary for the requested language.
    Falls back to English if `language` is not recognized.
    """
    return _LABELS.get(language, _LABELS["en"])


def plot_topic_distribution(
    distribution: pd.DataFrame,
    language: str = "en",
    label_maxlen: int = DEFAULT_LABEL_MAXLEN,
) -> plt.Figure:
    """Stacked bar chart of topic distribution per document.

    Long document filenames are truncated in the X axis (via
    truncate_label) so the figure stays readable. The original
    filenames remain in the underlying DataFrame and in any
    accompanying CSV/JSON export — only the chart labels are shortened.
    """
    lbl = _LABELS.get(language, _LABELS["en"])
    n = len(distribution)
    fig, ax = plt.subplots(figsize=auto_figsize(n))

    # Truncate document labels for display only — keep underlying data intact
    display_df = distribution.copy()
    display_df.index = truncate_labels(display_df.index, label_maxlen)

    display_df.plot(kind='bar', stacked=True, colormap='Paired', ax=ax, width=0.85)
    ax.set_xlabel(lbl["documents"])
    ax.set_ylabel(wrap_ylabel(lbl["distr"]))
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), prop={'size': 9})
    plt.xticks(rotation=90, fontsize=7 if n > 40 else 8)

    # subplots_adjust gives us full control over margins
    fig.subplots_adjust(left=0.10, right=0.85, top=0.95, bottom=0.30)
    return fig


def plot_topic_comparison(
    topic_words_nmf: pd.DataFrame,
    topic_words_lda: pd.DataFrame,
    top_n: int = 20,
) -> Optional[plt.Figure]:
    """
    Jaccard similarity heatmap between NMF and LDA topics, based on
    their `top_n` most important words.
    Only works if NMF and LDA have the same number of topics.
    """
    n_nmf = topic_words_nmf.shape[1]
    n_lda = topic_words_lda.shape[1]
    if n_nmf != n_lda:
        return None

    sim = np.zeros((n_nmf, n_lda))
    for i in range(n_nmf):
        wi = set(topic_words_nmf.iloc[:top_n, i])
        for j in range(n_lda):
            wj = set(topic_words_lda.iloc[:top_n, j])
            sim[i, j] = len(wi & wj) / float(top_n)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(sim, annot=True, fmt='.2f', cmap="PiYG",
                vmin=0, vmax=1, ax=ax)
    ax.set_xlabel("LDA topics")
    ax.set_ylabel("NMF topics")
    fig.tight_layout()
    return fig


# =============================================================================
# 7. TOPIC EVOLUTION THROUGH TEXTS  (used by page 4)
# =============================================================================

def rolling_mean_distribution(
    doctopic: np.ndarray,
    doc_labels: List[str],
    window: int = 2,
) -> pd.DataFrame:
    """
    Sort documents by filename (alphabetical = chronological if filenames
    start with a date), then compute a rolling mean of the topic
    distribution over that ordering. Mirrors the menu-2 behaviour of
    the original MTA.py.

    Parameters
    ----------
    doctopic : np.ndarray
        Shape (n_documents, n_topics). model.fit_transform(X) for NMF/LDA.
    doc_labels : list[str]
        Document filenames. Used both as the sort key and as the result index.
    window : int
        Rolling-mean window size in number of documents (>= 1).

    Returns
    -------
    pd.DataFrame
        Index = sorted document labels. Columns = 'RM_Topic_0', 'RM_Topic_1', ...
        Each cell = rolling mean of the corresponding topic weight,
        with min_periods=1 (so the first rows are not NaN, the window
        just grows from 1 up to `window`).
    """
    n_topics = doctopic.shape[1]
    df = pd.DataFrame(
        doctopic,
        index=doc_labels,
        columns=[f"Topic_{i}" for i in range(n_topics)],
    )
    df = df.sort_index(ascending=True)
    rm = df.rolling(window=max(1, int(window)), min_periods=1, center=False).mean()
    rm.columns = [f"RM_{c}" for c in rm.columns]
    rm.index.name = "Document"
    return rm


def yearly_topic_evolution(
    rolling_mean_df: pd.DataFrame,
    year_extractor=None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Group a rolling-mean DataFrame by year (extracted from each row index)
    and return the per-year mean of each topic. Mirrors the YYYY grouping
    of the original MTA.py menu 2.

    Parameters
    ----------
    rolling_mean_df : pd.DataFrame
        Output of rolling_mean_distribution: index = filenames,
        columns = 'RM_Topic_*'.
    year_extractor : callable, optional
        Function (filename -> str) returning the year string for a
        filename. If None, defaults to the first 4 characters of the
        filename — which works for 'YYYY-...' filenames as in the
        original MTA.py.

    Returns
    -------
    grouped : pd.DataFrame
        Index = year strings (sorted), columns = same as input.
        One row per year, value = mean of the rolling means within that year.
    bad_labels : list[str]
        Filenames whose first 4 characters do NOT look like a year
        (not 4 consecutive digits). The caller can warn the user about
        these; they are dropped from the grouping.
    """
    import re
    if year_extractor is None:
        year_extractor = lambda s: str(s)[:4]

    years = []
    bad_labels = []
    for label in rolling_mean_df.index:
        y = year_extractor(label)
        if re.fullmatch(r"\d{4}", str(y)):
            years.append(str(y))
        else:
            years.append(None)
            bad_labels.append(label)

    df = rolling_mean_df.copy()
    df["__year__"] = years
    df = df.dropna(subset=["__year__"])
    if df.empty:
        return pd.DataFrame(columns=rolling_mean_df.columns), bad_labels

    grouped = df.groupby("__year__").mean(numeric_only=True)
    grouped.index.name = "Year"
    grouped = grouped.sort_index()
    return grouped, bad_labels


# =============================================================================
# 8. SEMANTIC CONTEXT (page 5 — word neighbourhoods, 2D clouds, sub-corpus)
# =============================================================================

def tokenize_for_cooccurrence(
    corpus_wo: List[str],
) -> List[List[str]]:
    """
    Turn the cleaned corpus into a list of token lists. Reuses the
    splitting that build_matrices() does implicitly (whitespace).
    """
    return [doc.split() for doc in corpus_wo if doc.strip()]


def build_cooccurrence_embeddings(
    corpus_wo: List[str],
    window: int = 5,
    min_count: int = 2,
    n_dims: int = 100,
    random_state: int = 0,
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Compute word embeddings via co-occurrence + Truncated SVD.

    Step 1: count how often each word co-occurs with each other word
            within a window of `window` words on each side.
    Step 2: take log(1 + counts) to dampen frequent words.
    Step 3: Truncated SVD to reduce to `n_dims` dimensions.

    Parameters
    ----------
    corpus_wo : list[str]
        Cleaned corpus (output of preprocess_corpus).
    window : int
        Half-window size (so total span is 2*window + 1).
    min_count : int
        Drop words that appear in fewer documents than this.
    n_dims : int
        Number of latent dimensions.

    Returns
    -------
    embeddings : dict[str, np.ndarray]
        word → vector of shape (n_dims,)
    vocab : list[str]
        Vocabulary in deterministic order.
    """
    from collections import Counter, defaultdict
    from scipy.sparse import lil_matrix, csr_matrix
    from sklearn.decomposition import TruncatedSVD

    tokenized = tokenize_for_cooccurrence(corpus_wo)

    # 1. Build vocabulary with min_count filter (count over documents,
    # consistent with sklearn's min_df convention used elsewhere).
    doc_freq = Counter()
    for doc in tokenized:
        for w in set(doc):
            doc_freq[w] += 1
    vocab = sorted([w for w, c in doc_freq.items() if c >= min_count])
    if len(vocab) < 2:
        return {}, vocab
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)

    # 2. Build co-occurrence matrix. Use a dict-of-Counter for memory
    # efficiency on sparse data, then convert to CSR.
    cooc = defaultdict(Counter)
    for doc in tokenized:
        for i, w in enumerate(doc):
            if w not in word_to_idx:
                continue
            wi = word_to_idx[w]
            j_start = max(0, i - window)
            j_end = min(len(doc), i + window + 1)
            for j in range(j_start, j_end):
                if j == i:
                    continue
                w2 = doc[j]
                if w2 not in word_to_idx:
                    continue
                cooc[wi][word_to_idx[w2]] += 1

    # Convert to sparse matrix
    rows, cols, vals = [], [], []
    for wi, neighs in cooc.items():
        for wj, c in neighs.items():
            rows.append(wi)
            cols.append(wj)
            vals.append(np.log1p(c))  # smooth out frequency effects
    if not vals:
        return {}, vocab
    M = csr_matrix((vals, (rows, cols)), shape=(V, V), dtype=np.float32)

    # 3. SVD reduction. n_dims must be < V.
    actual_dims = min(n_dims, V - 1)
    svd = TruncatedSVD(n_components=actual_dims, random_state=random_state)
    reduced = svd.fit_transform(M)

    embeddings = {vocab[i]: reduced[i] for i in range(V)}
    return embeddings, vocab


def most_similar_words(
    embeddings: Dict[str, np.ndarray],
    word: str,
    topn: int = 10,
) -> List[Tuple[str, float]]:
    """
    Return the topn most similar words to `word` using cosine similarity
    over an embeddings dict.

    Returns
    -------
    list of (word, similarity) tuples, sorted by descending similarity.
    Empty list if the word is not in the embeddings.
    """
    if word not in embeddings:
        return []
    v = embeddings[word]
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-12:
        return []
    v_unit = v / v_norm

    sims = []
    for w, vw in embeddings.items():
        if w == word:
            continue
        nw = np.linalg.norm(vw)
        if nw < 1e-12:
            continue
        sim = float(np.dot(v_unit, vw) / nw)
        sims.append((w, sim))
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:topn]


def pca_project_word_clusters(
    embeddings: Dict[str, np.ndarray],
    seed_words: List[str],
    neighbours_per_seed: int = 50,
    random_state: int = 0,
) -> pd.DataFrame:
    """
    For each seed word, get its `neighbours_per_seed` nearest neighbours,
    then project all of them (seeds + neighbours) jointly into 2D via PCA.

    A word can appear in multiple clusters: if "capital" is a neighbour
    of both "banks" AND "markets", it will appear twice — once per
    cluster — with the SAME (x, y) coordinates (since the vector is the
    same). Visually, the user sees that "capital" belongs to both
    semantic worlds.

    Seeds themselves are NEVER counted as neighbours of other seeds:
    a seed always appears once, marked IsSeed=True, in its own cluster.

    Returns
    -------
    pd.DataFrame with columns: Word, Cluster, x, y, IsSeed (bool).
    The 'Cluster' column = the seed word the row belongs to.
    """
    from sklearn.decomposition import PCA

    # Filter seeds that actually exist in the embeddings, preserving order
    valid_seeds = [s for s in seed_words if s in embeddings]
    seed_set = set(valid_seeds)
    if not valid_seeds:
        return pd.DataFrame(columns=["Word", "Cluster", "x", "y", "IsSeed"])

    # Build rows: for each seed, add the seed itself + its neighbours
    # (excluding any other seed, so seeds keep their unique identity).
    rows = []
    for seed in valid_seeds:
        # The seed itself
        rows.append({"Word": seed, "Cluster": seed, "IsSeed": True})
        # Its neighbours, minus any word that is itself a seed
        neighbours = most_similar_words(embeddings, seed,
                                        topn=neighbours_per_seed
                                        + len(valid_seeds))
        added = 0
        for w, _ in neighbours:
            if w in seed_set:
                continue  # never add a seed as a neighbour
            rows.append({"Word": w, "Cluster": seed, "IsSeed": False})
            added += 1
            if added >= neighbours_per_seed:
                break

    # Compute PCA on UNIQUE words (so PCA isn't biased by duplicates),
    # then look up each row's coordinates from this PCA.
    unique_words = sorted({r["Word"] for r in rows})
    if len(unique_words) < 2:
        return pd.DataFrame(columns=["Word", "Cluster", "x", "y", "IsSeed"])
    X = np.stack([embeddings[w] for w in unique_words])
    pca = PCA(n_components=2, random_state=random_state)
    Y = pca.fit_transform(X)
    coords = {w: (Y[i, 0], Y[i, 1]) for i, w in enumerate(unique_words)}

    out = pd.DataFrame(rows)
    out["x"] = out["Word"].map(lambda w: coords[w][0])
    out["y"] = out["Word"].map(lambda w: coords[w][1])
    return out


def plot_semantic_cloud(
    df_cloud: pd.DataFrame,
    annotate_neighbours: bool = True,
    max_annotations_per_cluster: int = 15,
):
    """
    Render a 2D scatter plot of the semantic cloud (output of
    pca_project_word_clusters). Seeds are shown as large black diamonds
    with bold labels; neighbours are coloured dots, optionally annotated
    with small text labels.

    Parameters
    ----------
    df_cloud : DataFrame
        Output of `pca_project_word_clusters`, with columns
        Word / Cluster / IsSeed / x / y.
    annotate_neighbours : bool
        If True, draw small text labels next to the closest neighbours
        of each seed. If False, only seed labels are drawn.
    max_annotations_per_cluster : int
        Maximum number of neighbour labels per cluster. Labels go to
        the closest neighbours to the seed in PCA space; other points
        are still drawn but unlabeled. The full word list is always
        in df_cloud and can be exported via CSV/JSON.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    if df_cloud.empty:
        return None

    fig, ax = plt.subplots(figsize=(9, 7))

    # Categorical palette
    from matplotlib import colormaps
    palette = colormaps["tab10"]
    clusters = list(df_cloud["Cluster"].unique())
    colour_map = {c: palette(i % 10) for i, c in enumerate(clusters)}

    for cluster in clusters:
        sub = df_cloud[df_cloud["Cluster"] == cluster]
        colour = colour_map[cluster]
        neighbours = sub[~sub["IsSeed"]]
        ax.scatter(neighbours["x"], neighbours["y"], label=cluster,
                   color=colour, alpha=0.55, s=40)
        # Annotate the N closest neighbours (Euclidean in PCA plane =
        # visual proximity), provided the user wants labels at all.
        if annotate_neighbours and len(neighbours) > 0:
            seed_row = sub[sub["IsSeed"]]
            if not seed_row.empty:
                sx, sy = seed_row.iloc[0]["x"], seed_row.iloc[0]["y"]
                dists = ((neighbours["x"] - sx) ** 2
                         + (neighbours["y"] - sy) ** 2)
                closest = neighbours.iloc[
                    dists.argsort()[:max_annotations_per_cluster]
                ]
                for _, row in closest.iterrows():
                    ax.annotate(row["Word"], (row["x"], row["y"]),
                                fontsize=7, alpha=0.75, color=colour,
                                xytext=(4, 2), textcoords="offset points")
        # Seed: big black diamond with bold label
        for _, row in sub[sub["IsSeed"]].iterrows():
            ax.scatter(row["x"], row["y"], marker="D", s=160,
                       color="black", zorder=5,
                       edgecolor="white", linewidths=1.5)
            ax.annotate(row["Word"], (row["x"], row["y"]),
                        fontsize=13, fontweight="bold",
                        xytext=(10, 2), textcoords="offset points",
                        zorder=6)

    ax.legend(loc="best", fontsize=9, title="Seed word")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("PCA axis 1", fontsize=9, alpha=0.6)
    ax.set_ylabel("PCA axis 2", fontsize=9, alpha=0.6)
    # Subtitle explaining the label-cap so readers understand why some
    # points are unlabeled.
    if annotate_neighbours:
        fig.text(
            0.5, 0.01,
            f"Labels shown for the top-{max_annotations_per_cluster} "
            f"closest words to each seed (other points stay unlabeled "
            f"to keep the plot readable). "
            f"Full word list in the accompanying CSV/JSON.",
            ha="center", va="bottom",
            fontsize=8, color="#555", style="italic",
        )
        fig.tight_layout(rect=(0, 0.04, 1, 1))
    else:
        fig.tight_layout()
    return fig


def best_documents_for_words(
    tfidf_matrix,
    feature_names: np.ndarray,
    doc_labels: List[str],
    seed_words: List[str],
    similar_words_map: Dict[str, List[str]],
    min_mean_weight: float = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    For a set of seed words and their similar words, compute, for each
    document, the mean TF-IDF weight across that union of words. Return
    documents sorted by that mean, optionally filtered to those above a
    threshold.

    This reproduces the menu-4 logic of original MTA.py: the "best files
    for chosen words" feature, used to build sub-corpora.

    Parameters
    ----------
    tfidf_matrix : sparse matrix
        Shape (n_documents, n_features). The TF-IDF matrix.
    feature_names : np.ndarray
        Vocabulary, length n_features.
    doc_labels : list[str]
        Document filenames.
    seed_words : list[str]
        Words entered by the user.
    similar_words_map : dict[str, list[str]]
        For each seed word, its list of similar words.
    min_mean_weight : float, optional
        If given, keep only documents whose mean is strictly greater.

    Returns
    -------
    df : pd.DataFrame
        Index = doc_labels (filtered, sorted by mean desc).
        Columns = the union of seed words and their neighbours that
        actually appear in the vocabulary, plus 'mean'.
    full_means : pd.Series
        Mean over ALL documents (unfiltered), so the UI can show
        descriptive statistics.
    """
    # Build the union of words to look up
    all_words = list(seed_words)
    for v in similar_words_map.values():
        all_words.extend(v)
    # Keep only words present in the vocabulary
    vocab = {w: i for i, w in enumerate(feature_names)}
    cols_in_vocab = [w for w in dict.fromkeys(all_words) if w in vocab]

    if not cols_in_vocab:
        return pd.DataFrame(), pd.Series(dtype=float)

    # Slice the relevant columns of the TF-IDF matrix
    idxs = [vocab[w] for w in cols_in_vocab]
    sub = np.asarray(tfidf_matrix[:, idxs].todense())
    df = pd.DataFrame(sub, index=doc_labels, columns=cols_in_vocab)
    df["mean"] = df.mean(axis=1)

    full_means = df["mean"].copy()

    if min_mean_weight is not None:
        df = df.loc[df["mean"] > float(min_mean_weight)]

    df = df.sort_values("mean", ascending=False)
    df.index.name = "Document"
    return df, full_means


# =============================================================================
# 9. GROUP COMPARISON  (used by page 6 "Group comparison")
# =============================================================================

def extract_groups_from_filenames(
    filenames: List[str],
    position: int,
    separator: str = "_",
) -> Tuple[Dict[str, str], List[str]]:
    """
    Extract a group code from each filename at a given 1-indexed
    position within the underscore-separated parts.

    Example
    -------
    >>> extract_groups_from_filenames(
    ...     ["interview_F_25-34_001.txt", "interview_M_45-59_002.txt"],
    ...     position=2,
    ... )
    ({"interview_F_25-34_001.txt": "F",
      "interview_M_45-59_002.txt": "M"}, [])

    Parameters
    ----------
    filenames : list[str]
        Document filenames (possibly with .txt extension).
    position : int
        1-indexed position of the group code in the split filename.
    separator : str
        Separator character (default "_").

    Returns
    -------
    groups : dict[str, str]
        Mapping filename → group code, only for files where the position
        existed.
    skipped : list[str]
        Filenames that did NOT have a part at the given position; the
        caller should report these to the user.
    """
    groups = {}
    skipped = []
    for fname in filenames:
        # Strip extension before splitting (so .txt doesn't shift positions)
        base = fname.rsplit(".", 1)[0]
        parts = base.split(separator)
        if 1 <= position <= len(parts):
            groups[fname] = parts[position - 1]
        else:
            skipped.append(fname)
    return groups, skipped


def extract_groups_from_csv(
    csv_path: str,
    doc_labels: List[str],
    filename_column: str = "filename",
) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    """
    Load groupings from a CSV file. The CSV must have a 'filename' column
    (or the column name passed via `filename_column`); every other
    column is treated as a separate grouping (e.g. 'gender', 'age_band',
    'income_band').

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    doc_labels : list[str]
        The doc_labels currently loaded, used to filter the CSV to only
        the files present in the corpus.
    filename_column : str
        Name of the filename column. Default 'filename'.

    Returns
    -------
    groupings : dict[str, dict[str, str]]
        For each grouping column, a {filename: group_code} mapping.
        E.g. {"gender": {"file1.txt": "F", ...},
              "age_band": {"file1.txt": "25-34", ...}}
    skipped : list[str]
        Filenames present in doc_labels but NOT in the CSV.
    """
    df = pd.read_csv(csv_path)
    if filename_column not in df.columns:
        raise ValueError(
            f"CSV must have a column named {filename_column!r}; "
            f"found columns: {list(df.columns)}"
        )
    grouping_cols = [c for c in df.columns if c != filename_column]
    if not grouping_cols:
        raise ValueError(
            f"CSV must have at least one column other than {filename_column!r} "
            f"to define groups."
        )

    # Index by filename for fast lookup
    df_indexed = df.set_index(filename_column)
    groupings = {col: {} for col in grouping_cols}
    skipped = []
    for label in doc_labels:
        if label in df_indexed.index:
            for col in grouping_cols:
                val = df_indexed.loc[label, col]
                # Skip NaN values: that file is excluded from that grouping
                if pd.notna(val):
                    groupings[col][label] = str(val)
        else:
            skipped.append(label)
    return groupings, skipped


def compute_group_statistics(
    distribution: pd.DataFrame,
    groups: Dict[str, str],
) -> pd.DataFrame:
    """
    Compute mean and standard deviation per group, per topic.

    Parameters
    ----------
    distribution : pd.DataFrame
        Topic distribution per document (rows = documents, cols = topics).
    groups : dict[str, str]
        Mapping filename → group code. Files not in this dict are ignored.

    Returns
    -------
    pd.DataFrame
        Multi-column DataFrame: rows = topics, columns = (group, statistic).
        Statistics: 'n', 'mean', 'std'.
    """
    df = distribution.copy()
    df["__group__"] = df.index.map(groups)
    df = df.dropna(subset=["__group__"])
    if df.empty:
        return pd.DataFrame()

    topic_cols = [c for c in df.columns if c != "__group__"]
    grouped = df.groupby("__group__")[topic_cols].agg(["count", "mean", "std"])
    # Rearrange so rows are topics: transpose and reorder
    grouped = grouped.swaplevel(axis=1).sort_index(axis=1)
    return grouped.T


def benjamini_hochberg_correction(pvalues: List[float]) -> List[float]:
    """
    Apply Benjamini-Hochberg FDR correction to a list of p-values.
    Returns the corrected (adjusted) p-values in the same order as input.

    The correction is monotonic: adjusted p-values are non-decreasing
    when sorted by raw p-value, and capped at 1.0.
    """
    pvals = np.asarray(pvalues, dtype=float)
    n = len(pvals)
    if n == 0:
        return []
    # Sort, apply correction, unsort
    order = np.argsort(pvals)
    ranked = np.empty(n)
    ranked[order] = np.arange(1, n + 1)
    adjusted = pvals * n / ranked
    # Enforce monotonicity: walk from largest to smallest
    sorted_adj = adjusted[order]
    for i in range(n - 2, -1, -1):
        sorted_adj[i] = min(sorted_adj[i], sorted_adj[i + 1])
    final = np.empty(n)
    final[order] = sorted_adj
    return np.minimum(final, 1.0).tolist()


def compare_groups_pairwise(
    distribution: pd.DataFrame,
    groups: Dict[str, str],
    min_n_warning: int = 30,
) -> pd.DataFrame:
    """
    For each topic and each pair of groups, compute Welch's t-test and
    Mann-Whitney U, then apply Benjamini-Hochberg correction on the
    full set of p-values.

    Parameters
    ----------
    distribution : pd.DataFrame
        Topic distribution per document (rows = documents, cols = topics).
    groups : dict[str, str]
        Mapping filename → group code.
    min_n_warning : int
        If either group has fewer documents than this, flag the test
        as "small sample". Default 30.

    Returns
    -------
    pd.DataFrame
        Long-format table with one row per (topic, group_A, group_B) pair
        and columns: topic, group_A, group_B, n_A, n_B, mean_A, mean_B,
        p_welch, p_mannwhitney, p_welch_BH, p_mannwhitney_BH,
        small_sample (bool).
    """
    from scipy import stats

    df = distribution.copy()
    df["__group__"] = df.index.map(groups)
    df = df.dropna(subset=["__group__"])
    if df.empty:
        return pd.DataFrame()

    topic_cols = [c for c in df.columns if c != "__group__"]
    group_codes = sorted(df["__group__"].unique())
    if len(group_codes) < 2:
        return pd.DataFrame()  # No pairs to test

    rows = []
    for topic in topic_cols:
        for i, ga in enumerate(group_codes):
            for gb in group_codes[i + 1:]:
                values_a = df[df["__group__"] == ga][topic].values
                values_b = df[df["__group__"] == gb][topic].values
                n_a, n_b = len(values_a), len(values_b)
                if n_a < 2 or n_b < 2:
                    # Can't run tests with n<2 in either group
                    p_welch = np.nan
                    p_mwu = np.nan
                else:
                    # Welch's t-test (does NOT assume equal variances)
                    try:
                        _, p_welch = stats.ttest_ind(
                            values_a, values_b, equal_var=False
                        )
                    except Exception:
                        p_welch = np.nan
                    # Mann-Whitney U (non-parametric)
                    try:
                        _, p_mwu = stats.mannwhitneyu(
                            values_a, values_b, alternative="two-sided"
                        )
                    except Exception:
                        p_mwu = np.nan
                rows.append({
                    "topic":         topic,
                    "group_A":       ga,
                    "group_B":       gb,
                    "n_A":           n_a,
                    "n_B":           n_b,
                    "mean_A":        float(values_a.mean()) if n_a else np.nan,
                    "mean_B":        float(values_b.mean()) if n_b else np.nan,
                    "p_welch":       float(p_welch) if not np.isnan(p_welch) else np.nan,
                    "p_mannwhitney": float(p_mwu) if not np.isnan(p_mwu) else np.nan,
                    "small_sample":  (n_a < min_n_warning) or (n_b < min_n_warning),
                })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Apply Benjamini-Hochberg correction on each metric (skipping NaNs)
    for col in ["p_welch", "p_mannwhitney"]:
        valid = out[col].notna()
        if valid.any():
            corrected_vals = benjamini_hochberg_correction(
                out.loc[valid, col].tolist()
            )
            out[f"{col}_BH"] = np.nan
            out.loc[valid, f"{col}_BH"] = corrected_vals
        else:
            out[f"{col}_BH"] = np.nan

    return out
