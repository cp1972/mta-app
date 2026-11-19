"""Test of the residual topic and the two-stage refinement (MTA 3.10).

Run: python test_refine.py
"""
import numpy as np

import mta_core as mta

ok = True


def check(cond, msg):
    global ok
    print(("✓ " if cond else "✗ ") + msg)
    ok = ok and cond


# Two clear themes of 60 documents each, and 120 documents that share the common
# words of the corpus ("make", "please", ...) and differ only by three smaller
# themes hidden under them: a residual topic in stage 1, three topics in stage 2.
rng = np.random.default_rng(3)
sfx = [a + b for a in "abcdef" for b in "xyz"]      # letters only: MTA removes digits
vocab = {name: [name + x for x in sfx[:15]]
         for name in ("garden", "music", "cook", "travel", "sport")}
common = ["common" + x for x in sfx]


def doc(words, n, extra=()):
    w = list(rng.choice(words, n)) + list(extra)
    rng.shuffle(w)
    return " ".join(w)


texts, truth = [], []
for name in ("garden", "music"):
    for _ in range(60):
        texts.append(doc(vocab[name], 25)); truth.append(name)
for name in ("cook", "travel", "sport"):
    for _ in range(40):
        texts.append(doc(common, 30, rng.choice(vocab[name], 4))); truth.append(name)
texts.append("")                                   # an empty document
truth.append("")
labels = [f"d{i:03d}" for i in range(len(texts))]
corpus_wo, _ = mta.preprocess_corpus(texts, [], min_word_length=3)
m = mta.build_matrices(corpus_wo, [], min_df=2, max_df=0.95)

s1 = mta.topic_stability(m["tf_matrix"], 3, n_runs=6, doc_labels=labels)
rt = mta.residual_topics(s1)
check(list(rt.columns) == ["Topic", "Documents (consensus)", "Share", "Residual"]
      and abs(rt["Share"].sum() - 1) < 1e-9, "residual table: share of documents per topic")
rec = mta.refinement_recommendation(s1)
check(rec is not None and len(rec["topics"]) == 1 and rec["share"] >= 0.40,
      "one residual topic (≥ 40 % of the documents) is recommended for refinement")
r = rec["topics"][0]
check(mta.refinement_recommendation(s1, share=0.95) is None, "threshold is configurable: none above 95 %")
cons = s1["documents"]["Consensus topic"].tolist()
check(sum(c == f"Topic_{r}" for c, t in zip(cons, truth) if t in ("cook", "travel", "sport")) >= 110,
      "the residual topic holds the documents of the hidden themes")

ref = mta.refine_residual(corpus_wo, [], s1, rec["topics"], ks=[2, 3, 4], n_runs=6,
                          doc_labels=labels)
check(list(ref["table"]["K"]) == [2, 3, 4] and len(ref["index"]) == rec["n_documents"],
      "stage 2: stability over K on the residual documents only")
check(not any(n.startswith("garden") or n.startswith("music") for n in ref["names"]),
      "stage 2 has the vocabulary of the residual documents")

mg = mta.merge_stages(s1, ref, 3, names1=m["tf_names"])
keep = [k for k in range(3) if k != r]
expect = [f"Topic_{k}" for k in keep] + [f"Topic_{r}.{j}" for j in range(3)]
check(mg["names"] == expect and list(mg["topics"]["Topic"]) == expect,
      f"merged topics: {', '.join(expect)}")
D = mg["documents"]
check(len(D) == len(texts) and set(D.loc[D["Stage"] == 2, "Stage 1 topic"]) == {f"Topic_{r}"},
      "merged documents: stage 2 only for documents of the residual topic")
part = {}
for t, top in zip(truth, D["Topic"]):
    if t in ("cook", "travel", "sport"):
        part.setdefault(t, []).append(top)
major = {t: max(set(v), key=v.count) for t, v in part.items()}
purity = np.mean([v.count(major[t]) / len(v) for t, v in part.items()])
check(len(set(major.values())) == 3 and purity >= 0.9,
      f"the hidden themes become three topics of stage 2 (purity {purity:.2f})")
W = mg["weights"]
mean = W[expect].to_numpy()
rest = f"Topic_{r} (not refined)"
cols = expect + [rest]
allw = W[cols].to_numpy()
check(mg["not_refined"] == rest and np.allclose(allw[:-1].sum(1), 1.0) and np.all(allw[-1] == 0),
      "weights (with the unrefined residual) sum to 1 per document; the empty document keeps zeros")
check(np.all(W[[f"{c} sd" for c in cols]].to_numpy() >= 0), "standard deviations are not negative")
own = (W["Stage 2 weights"] == "own").to_numpy()
w1r = s1["weights"][f"Topic_{r}"].to_numpy()
check(set(W["Stage 2 weights"]) == {"own", "not refined"} and own.sum() == len(ref["index"])
      and np.all(W.loc[own, rest] == 0)
      and np.allclose(W.loc[~own, rest], w1r[~own])
      and np.all(W.loc[~own, [f"Topic_{r}.{j}" for j in range(3)]].to_numpy() == 0),
      "documents outside the residual topic: residual weight kept unrefined, not given to any stage-2 topic")
check(np.allclose(W.loc[own, [f"Topic_{r}.{j}" for j in range(3)]].sum(axis=1), w1r[own]),
      "documents of the residual topic: residual weight divided among the stage-2 topics")
check(np.allclose(W[f"Topic_{keep[0]}"], s1["weights"][f"Topic_{keep[0]}"]),
      "the topics of stage 1 keep their weights")
check("Top words" in mg["topics"] and all(mg["topics"]["Top words"].str.len() > 0),
      "merged topics with top words of their own stage")

# reproducible
ref2 = mta.refine_residual(corpus_wo, [], s1, rec["topics"], ks=[3], n_runs=6, doc_labels=labels)
mg2 = mta.merge_stages(s1, ref2, 3)
check(mg2["documents"].equals(mg["documents"]) and np.allclose(mg2["weights"][cols], W[cols]),
      "reproducible (fixed seeds)")

# a corpus without residual topic: no recommendation
s_bal = mta.topic_stability(m["tf_matrix"][:120], 2, n_runs=4)
check(mta.refinement_recommendation(s_bal, share=0.6) is None, "balanced topics: no recommendation")

# LDA
s1l = mta.topic_stability(m["lda_matrix"], 3, method="lda", n_runs=3, doc_labels=labels)
recl = mta.refinement_recommendation(s1l, share=0.3)
if recl:
    refl = mta.refine_residual(corpus_wo, [], s1l, recl["topics"], ks=[3], method="lda",
                               n_runs=3, doc_labels=labels)
    mgl = mta.merge_stages(s1l, refl, 3)
    check(np.allclose(mgl["weights"][mgl["names"] + [mgl["not_refined"]]].to_numpy()[:-1].sum(1), 1.0),
          "LDA: merged weights sum to 1")
else:
    check(False, "LDA: a residual topic at 30 %")

print("✓ all tests passed" if ok else "✗ some tests failed")
raise SystemExit(0 if ok else 1)
