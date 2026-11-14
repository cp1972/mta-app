"""Test for large corpora and empty documents (MTA 3.8).

Run: python test_large.py
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.sparse import random as sparse_random
from scipy.spatial.distance import pdist

import mta_core as mta

ok = True


def check(cond, msg):
    global ok
    print(("✓ " if cond else "✗ ") + msg)
    ok = ok and cond


# 1. Empty document: row of zeros, dominant topic "none (empty document)"
doctopic = np.array([[0.2, 0.8], [0.0, 0.0], [0.5, 0.1]])
dist = mta.topic_distribution_per_doc(doctopic, ["a.txt", "b.txt", "c.txt"])
check(not dist.isna().any().any(), "no NaN in the topic distribution")
check((dist.loc["b.txt"] == 0).all(), "empty document keeps a row of zeros")
dom = mta.dominant_topic_per_doc(dist)
check(dom["Dominant topic"].tolist() == ["Topic_1", mta.EMPTY_TOPIC_LABEL, "Topic_0"],
      "dominant topics: Topic_1, none (empty document), Topic_0")
check(float(dom["Weight"].iloc[1]) == 0.0, "weight of the empty document is 0")

# 2. Cophenetic correlation: unchanged below the limit, on a sample above it
rng = np.random.default_rng(3)
small = rng.random((200, 5))
_, c_small, n_small = mta.cophenet_corr(small)
c_ref = cophenet(linkage(small, "ward"), pdist(small))[0]
check(n_small == 200 and abs(c_small - c_ref) < 1e-12,
      "small matrix: all documents, same coefficient as before")
big = rng.random((mta.COPHENET_MAX_DOCS + 1500, 5))
_, c1, n1 = mta.cophenet_corr(big)
_, c2, _ = mta.cophenet_corr(big)
check(n1 == mta.COPHENET_MAX_DOCS, f"large matrix: sample of {mta.COPHENET_MAX_DOCS} documents")
check(c1 == c2, "sample is reproducible (fixed seed)")

# 3. Dense sample within the limit
m = sparse_random(3000, 400, density=0.01, format="csr", random_state=1)
d = mta.dense_sample(m, max_cells=10**9)
check(d.shape == (3000, 400) and np.allclose(d, m.toarray()), "below the limit: full dense matrix")
d = mta.dense_sample(m, max_cells=400 * 500)
check(d.shape == (500, 400), "above the limit: sample of 500 documents")

print("✓ all tests passed" if ok else "✗ some tests failed")
raise SystemExit(0 if ok else 1)
