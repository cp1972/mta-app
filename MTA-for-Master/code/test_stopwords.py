"""Test for the stopword handling (MTA 3.6).

Run from this folder:  python3 test_stopwords.py   (or: pytest test_stopwords.py)

The same 13 German stopwords are written to a file in seven ways that users actually produce. With every
variant, no stopword may remain, neither in the cleaned corpus (used by semantic context and networks)
nor in the TF-IDF matrix (used by the topic models). Up to 3.5 only the first variant worked, and even
there stopwords followed by punctuation ("alles.", "nicht,") stayed in the cleaned corpus.
"""
import unicodedata
import warnings

import mta_core as mta

warnings.filterwarnings("ignore")

DOCS = [
    "Die Arbeiter, die in der Fabrik arbeiten, und die Händler. Aber für die Arbeiter ist das nicht über alles.",
    "Und die Fabrik wächst, aber die Arbeiter bleiben. Für die Händler ist es über die Zeit schwierig, und sie bleiben.",
    "Die Händler und die Arbeiter: über die Fabrik und für die Zeit. Aber sie wachsen nicht.",
] * 3
STOP = ["die", "der", "und", "in", "aber", "für", "über", "ist", "das", "nicht", "es", "sie", "alles"]

VARIANTS = {
    "UTF-8, one word per line": "\n".join(STOP).encode("utf-8"),
    "UTF-8 with byte-order mark": ("﻿" + "\n".join(STOP)).encode("utf-8"),
    "Windows-1252, CRLF": "\r\n".join(STOP).encode("cp1252"),
    "macOS, decomposed umlauts (NFD)": unicodedata.normalize("NFD", "\n".join(STOP)).encode("utf-8"),
    "capitalised entries": "\n".join(w.capitalize() for w in STOP).encode("utf-8"),
    "one line, commas": ", ".join(STOP).encode("utf-8"),
    "one line, spaces": " ".join(STOP).encode("utf-8"),
}


def remaining(raw: bytes):
    sw = mta.read_stopwords(raw)
    wo, _ = mta.preprocess_corpus(DOCS, sw, min_word_length=3)
    m = mta.build_matrices(wo, sw, min_df=1, max_df=1.0)
    in_corpus = sorted({w for d in wo for w in d.split() if w in STOP})
    in_matrix = sorted(set(m["tf_names"]) & set(STOP))
    return len(sw), in_corpus, in_matrix


def test_all_variants():
    for name, raw in VARIANTS.items():
        n, in_corpus, in_matrix = remaining(raw)
        assert n == len(STOP), (name, n)
        assert not in_corpus, (name, in_corpus)
        assert not in_matrix, (name, in_matrix)


def test_text_in_nfd_and_capitals():
    """The text, too, may come from macOS (NFD) and use capitals: 'Über' at the start of a sentence."""
    docs = [unicodedata.normalize("NFD", "Über die Fabrik. Für die Arbeiter wächst die Fabrik.")] * 3
    sw = mta.read_stopwords("über\nfür\ndie".encode("utf-8"))
    wo, _ = mta.preprocess_corpus(docs, sw, min_word_length=3)
    assert not {w for d in wo for w in d.split()} & {"über", "für", "die"}, wo


if __name__ == "__main__":
    for name, raw in VARIANTS.items():
        n, c, m = remaining(raw)
        print(f"{name:34} {n:2} stopwords · left in corpus: {c or '–'} · left in matrix: {m or '–'}")
    test_all_variants()
    test_text_in_nfd_and_capitals()
    print("✓ all tests passed")
