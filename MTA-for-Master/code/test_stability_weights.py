"""Test of the topic weights averaged over the runs of the stability check (MTA 3.9).

Run: python test_stability_weights.py
"""
import numpy as np
from scipy.sparse import random as sparse_random, vstack, csr_matrix

import mta_core as mta

ok = True


def check(cond, msg):
    global ok
    print(("✓ " if cond else "✗ ") + msg)
    ok = ok and cond


# a small corpus with three clear topics and one empty document
rng = np.random.default_rng(7)
blocks = []
for t in range(3):
    m = np.zeros((40, 60))
    m[:, t * 20:(t + 1) * 20] = rng.poisson(2.0, (40, 20))
    m += rng.poisson(0.1, (40, 60))
    blocks.append(m)
X = csr_matrix(np.vstack(blocks + [np.zeros((1, 60))]))
labels = [f"d{i}" for i in range(X.shape[0])]

res = mta.topic_stability(X, 3, n_runs=8, doc_labels=labels)
W = res["weights"]
mean = W[[f"Topic_{k}" for k in range(3)]].to_numpy()
sd = W[[f"Topic_{k} sd" for k in range(3)]].to_numpy()

check(W.shape == (121, 7) and list(W["Document"]) == labels, "one row per document: mean and sd for every topic")
check(np.allclose(mean[:120].sum(1), 1.0), "the mean weights of a document sum to 1")
check(np.all(mean[120] == 0) and np.all(sd[120] == 0), "the empty document keeps zeros")
check(np.all(sd >= 0) and np.all(mean >= 0), "weights and standard deviations are not negative")
cons = res["documents"]["Consensus topic"].str.replace("Topic_", "").replace("", "-1").astype(int).to_numpy()
check((mean[:120].argmax(1) == cons[:120]).mean() >= 0.95, "the strongest mean weight is the consensus topic (≥ 95 % of documents)")
check(float(sd[:120].mean()) < 0.1, "clear topics: small spread over the runs")

# the other tables are unchanged by 3.9: the same runs give the same consensus and agreement
res2 = mta.topic_stability(X, 3, n_runs=8, doc_labels=labels)
check(res2["documents"].equals(res["documents"]) and res2["weights"].equals(W), "reproducible (fixed seeds)")

# LDA: the same table
resl = mta.topic_stability(X, 3, method="lda", n_runs=3, doc_labels=labels)
ml = resl["weights"][[f"Topic_{k}" for k in range(3)]].to_numpy()
check(np.allclose(ml[:120].sum(1), 1.0), "LDA: mean weights sum to 1")

print("✓ all tests passed" if ok else "✗ some tests failed")
raise SystemExit(0 if ok else 1)
