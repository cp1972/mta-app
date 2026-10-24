"""Test for the commands of the app (MTA 3.7).

Run from this folder:  python3 test_commands.py   (or: pytest test_commands.py)

1. The commands are read back by the batch mode exactly as written: every
   setting arrives with its value, on Mac/Linux and on Windows quoting.
2. The pages of the app are driven with Streamlit's test harness on a small
   corpus (12 documents in three groups, built from the English demo corpus).
   Every analysis writes its command to the session log; each logged command
   is run in batch mode and its tables are compared with the results the app
   computed. The script for Mac/Linux must also pass `bash -n`.

Part 2 needs Streamlit and takes a minute or two.
"""
import glob
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd

import mta_commands as mc
import mta_core as mta
from MTA_v3 import build_parser

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.join(HERE, "..", "examples")


# =============================================================================
# 1. Round trip of the arguments
# =============================================================================

def _windows_argv(line: str) -> list:
    """Split a command line as the C runtime of python.exe does
    (CommandLineToArgvW rules)."""
    args, cur, i, inq, have = [], [], 0, False, False
    while i < len(line):
        c = line[i]
        if c == "\\":
            j = i
            while j < len(line) and line[j] == "\\":
                j += 1
            n = j - i
            if j < len(line) and line[j] == '"':
                cur.append("\\" * (n // 2))
                if n % 2:
                    cur.append('"')
                    i = j + 1
                else:
                    i = j
                have = True
                continue
            cur.append("\\" * n)
            i = j
            have = True
            continue
        if c == '"':
            inq = not inq
            have = True
        elif c in " \t" and not inq:
            if have:
                args.append("".join(cur))
            cur, have = [], False
        else:
            cur.append(c)
            have = True
        i += 1
    if have:
        args.append("".join(cur))
    return args


TRICKY = ["Meine Texte", "C:\\Daten\\Ordner mit Leerzeichen\\", 'a"b', "Über,Ärger",
          "50% & mehr", "x/ y", "", "a\\\\b", "(Klammer)", "Critique ↔ Innovation"]


def test_round_trip():
    ctx = {"min_word_length": 4, "min_df": 0.01, "max_df": 0.9, "language": "de"}
    args = mc.command("x", ctx, ["--action", "axis-analysis", "--axis-x", "0,1 / 2",
                                 "--axis-x-label", "Critique ↔ Innovation",
                                 "--words", "Über,Ärger", "--stability-threshold",
                                 repr(2 / 3)] + mc.model_args("nmf", {"n_topics": 7,
                                 "init": "random", "seed": 42}))
    values = {"CORPUS": "Meine Texte/", "STOPWORDS": "stop words.txt"}
    for platform in ("posix", "windows"):
        line = mc.render(args, platform, values=values)
        argv = (shlex.split(line) if platform == "posix" else _windows_argv(line))[2:]
        a = build_parser().parse_args(argv)
        # (on Windows without the trailing separator, which would escape the quote)
        assert a.corpus == ("Meine Texte/" if platform == "posix" else "Meine Texte"), platform
        assert a.stopwords == "stop words.txt", platform
        assert (a.min_word_length, a.min_df, a.max_df, a.language) == (4, 0.01, 0.9, "de")
        assert (a.n_topics, a.init, a.seed) == (7, "random", 42)
        assert a.axis_x == "0,1 / 2" and a.axis_x_label == "Critique ↔ Innovation"
        assert a.words == "Über,Ärger" and a.stability_threshold == 2 / 3
        assert a.output.replace("\\", "/") == "MTA-Results/x"
    # integer min_df stays a number of documents
    a = build_parser().parse_args(shlex.split(mc.render(mc.corpus_args(
        {"min_word_length": 3, "min_df": 2, "max_df": 0.95}), values=values))[2:])
    assert a.min_df == 2 and isinstance(a.min_df, int)
    for s in TRICKY:
        want = s.replace('"', "'")        # a double quote is written as ' on Windows
        assert _windows_argv(mc._quote_win(s)) == [want], s
        assert shlex.split(shlex.quote(s)) == [s], s
    # values beginning with '-' reach argparse as values
    args = mc.command("x", ctx, ["--action", "compare-groups", "--group-separator", "-_",
                                 "--axis-x-label", "-links", "--words", "a,b"])
    for platform in ("posix", "windows"):
        line = mc.render(args, platform, values=values)
        argv = (shlex.split(line) if platform == "posix" else _windows_argv(line))[2:]
        a = build_parser().parse_args(argv)
        assert (a.group_separator, a.axis_x_label, a.words) == ("-_", "-links", "a,b")


# =============================================================================
# 2. The app against the batch mode
# =============================================================================

def _make_corpus(folder):
    """12 documents: the six English demo texts cut in halves, named
    <year>_<group>_<nn>.txt (groups soc, eco, comp; years 2019–2021)."""
    os.makedirs(folder)
    texts = [open(p, encoding="utf-8").read()
             for p in sorted(glob.glob(os.path.join(EXAMPLES, "demo_corpus_en", "*.txt")))]
    groups = ["soc", "soc", "eco", "eco", "comp", "comp"]
    n = 0
    for i, t in enumerate(texts):
        words = t.split()
        for h, part in enumerate((words[:len(words) // 2], words[len(words) // 2:])):
            n += 1
            year = 2019 + (n % 3)
            with open(os.path.join(folder, f"{year}_{groups[i]}_{n:02d}.txt"), "w",
                      encoding="utf-8") as f:
                f.write(" ".join(part))


def _state(corpus, stop_path):
    paths = sorted(glob.glob(os.path.join(corpus, "*.txt")))
    raw = [mta.decode_text_file(open(p, "rb").read()) for p in paths]
    labels = [os.path.basename(p) for p in paths]
    stop = mta.read_stopwords(stop_path)
    wo, re_ = mta.preprocess_corpus(raw, stop, min_word_length=3)
    m = mta.build_matrices(wo, stop, min_df=2, max_df=0.95)
    return dict(raw_texts=raw, doc_labels=labels, stopwords=stop, corpus_wo=wo,
                corpus_re=re_, matrices=m,
                corpus_settings={"min_word_length": 3, "file_names": labels,
                                 "stopwords_name": os.path.basename(stop_path),
                                 "n_stopwords": len(stop)},
                matrix_settings={"min_word_length": 3, "min_df": 2, "max_df": 0.95},
                command_log=[], cmd_paths={"CORPUS": corpus, "STOPWORDS": stop_path,
                                           "GROUPS": ""})


def _by_label(widgets, label):
    return next(w for w in widgets if w.label == label)


def _run(args, values, workdir):
    argv = []
    for a in args:
        if a in (mc.CORPUS, mc.STOPWORDS, mc.GROUPS):
            argv.append(values[mc._VARS[a]])
        elif a.startswith(mc.OUTPUT):
            argv.append(os.path.join(workdir, "out", a[len(mc.OUTPUT):]))
        else:
            argv.append(a)
    r = subprocess.run([sys.executable, os.path.join(HERE, "MTA_v3.py"), "--format", "csv"]
                       + argv, cwd=workdir, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    return argv[argv.index("--output") + 1]


def _csv(folder, name):
    return pd.read_csv(os.path.join(folder, name + ".csv"), index_col=0)


def _same(df_batch, df_app, what):
    a = df_batch.to_numpy(dtype=float)
    b = np.asarray(df_app, dtype=float)
    assert a.shape == b.shape, (what, a.shape, b.shape)
    assert np.allclose(a, b, atol=1e-9, equal_nan=True), (what, np.nanmax(np.abs(a - b)))


def test_app_against_batch():
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        print("  (Streamlit not installed: part 2 skipped)")
        return
    tmp = tempfile.mkdtemp(prefix="mta_cmd_")
    try:
        corpus = os.path.join(tmp, "texts")
        _make_corpus(corpus)
        stop_path = os.path.join(EXAMPLES, "stopwords_en.txt")
        state = _state(corpus, stop_path)
        values = {"CORPUS": corpus, "STOPWORDS": stop_path, "GROUPS": ""}

        def page(name):
            at = AppTest.from_file(os.path.join(HERE, "pages", name), default_timeout=300)
            for k, v in state.items():
                at.session_state[k] = v
            at.run()
            assert not at.exception, [e.value for e in at.exception]
            return at

        # --- page 2: models with non-default settings, CV, stability ----------
        at = page("2_Topic_models.py")
        _by_label(at.number_input, "Number of topics — NMF").set_value(4)
        _by_label(at.number_input, "Number of topics — LDA").set_value(3)
        _by_label(at.selectbox, "NMF start").set_value("random")
        _by_label(at.number_input, "NMF seed").set_value(7)
        _by_label(at.number_input, "LDA seed").set_value(5)
        _by_label(at.slider, "Maximum number of topics to test").set_value(4)
        _by_label(at.slider, "Numbers of topics (K)").set_value((3, 4))
        _by_label(at.number_input, "Runs per K").set_value(5)
        at.run()
        for b in ("🟢 Run NMF", "🔵 Run LDA", "📊 Compute cross-validation metrics",
                  "🧪 Check stability"):
            _by_label(at.button, b).click().run()
            assert not at.exception, (b, [e.value for e in at.exception])
        for k in ("nmf_results", "lda_results", "nmf_words", "lda_words", "metrics",
                  "stability", "nmf_settings", "lda_settings", "cv_settings"):
            state[k] = at.session_state[k]
        app = {"nmf": state["nmf_results"], "lda": state["lda_results"]}
        labels = state["doc_labels"]
        m = state["matrices"]

        # --- pages 3–8 --------------------------------------------------------
        at = page("3_Word_weights.py")
        _by_label(at.text_input, "Words to analyze (comma- or space-separated)") \
            .input("capital, Banks markets").run()
        at = page("4_Topic_evolution.py")
        at.slider(key="window_LDA").set_value(3).run()
        at = page("5_Semantic_context.py")
        _by_label(at.slider, "Context window size").set_value(4)
        _by_label(at.slider, "Minimum word frequency").set_value(1)
        _by_label(at.slider, "Embedding dimensions").set_value(40)
        at.run()
        _by_label(at.text_input, "Words to analyze (comma- or space-separated)") \
            .input("capital algorithms").run()
        _by_label(at.slider, "Number of similar words to show per query word").set_value(8).run()
        at = page("6_Group_comparison.py")
        _by_label(at.radio, "Which model do you want to compare on?").set_value("LDA").run()
        at = page("7_Network_views.py")
        at.slider(key="doc_min_edge").set_value(20).run()
        at.button(key="gen_doc").click().run()
        at = page("8_Axis_analysis.py")
        at.multiselect(key="axis_X_left").set_value([0]).run()
        at.multiselect(key="axis_X_right").set_value([1, 2]).run()
        at.multiselect(key="axis_Y_right").set_value([3]).run()
        at.text_input(key="axis_X_label").input("Critique ↔ Innovation").run()
        at.number_input(key="stats_group_position").set_value(2).run()
        assert not at.exception, [e.value for e in at.exception]

        log = state["command_log"]
        ids = [e["id"] for e in log]
        for need in ("nmf", "lda", "cv", "stability", "word_weights_NMF", "word_weights_LDA",
                     "evolution_NMF", "evolution_LDA", "semantic", "groups_lda",
                     "network_doc_NMF", "axis_NMF"):
            assert need in ids, (need, ids)
        last = {e["id"]: e for e in log}          # the final settings of each analysis

        # --- the script must be valid bash ------------------------------------
        sh = os.path.join(tmp, "session.sh")
        open(sh, "w", encoding="utf-8").write(mc.script(log, "posix", values))
        assert subprocess.run(["bash", "-n", sh]).returncode == 0
        bat = mc.script(log, "windows", values)
        assert bat.count(r".venv\Scripts\python.exe") == len(log)
        assert "\n" not in bat.replace("\r\n", ""), "the .bat needs CRLF line ends"

        # --- run the commands, compare with the app ---------------------------
        def dist(model):
            return mta.topic_distribution_per_doc(app[model]["doctopic"], labels)

        o = _run(last["nmf"]["args"], values, tmp)
        _same(_csv(o, "nmf_distribution"), dist("nmf"), "NMF distribution")
        o = _run(last["lda"]["args"], values, tmp)
        _same(_csv(o, "lda_distribution"), dist("lda"), "LDA distribution")

        o = _run(last["cv"]["args"], values, tmp)
        mt = state["metrics"]
        _same(_csv(o, "cv_metrics")[["Elbow", "Silhouette", "Cophenet NMF", "Cophenet LDA"]],
              np.column_stack([list(mt[k].values()) for k in
                               ("elbow", "silhouette", "cophenet_nmf", "cophenet_lda")]),
              "cross-validation")

        o = _run(last["stability"]["args"], values, tmp)
        tab = state["stability"]["table"].set_index("K")
        _same(_csv(o, "nmf_stability_over_k"), tab, "stability")

        for model, vocab, term_doc in (("NMF", m["tf_names"], m["tf_matrix"]),
                                       ("LDA", m["lda_names"], m["lda_matrix"])):
            words = ["capital", "Banks", "markets"]
            o = _run(last[f"word_weights_{model}"]["args"], values, tmp)
            t, _ = mta.words_weight_per_topic(app[model.lower()]["topicwords"], vocab, words)
            d, _ = mta.words_weight_per_document(term_doc, vocab, labels, words)
            _same(_csv(o, "word_weights_topics"), t, f"word weights in topics {model}")
            _same(_csv(o, "word_weights_documents"), d, f"word weights in documents {model}")

        for model, window in (("NMF", 2), ("LDA", 3)):
            o = _run(last[f"evolution_{model}"]["args"], values, tmp)
            rm = mta.rolling_mean_distribution(app[model.lower()]["doctopic"], labels,
                                               window=window)
            _same(_csv(o, f"{model.lower()}_rolling_mean"), rm, f"evolution {model}")
            y, _ = mta.yearly_topic_evolution(rm)
            _same(_csv(o, f"{model.lower()}_yearly_evolution"), y, f"yearly {model}")

        o = _run(last["semantic"]["args"], values, tmp)
        emb, _ = mta.build_cooccurrence_embeddings(state["corpus_wo"], window=4,
                                                   min_count=1, n_dims=40)
        seeds = ["capital", "algorithms"]
        sims = {w: mta.most_similar_words(emb, w, topn=8) for w in seeds}
        got = pd.read_csv(os.path.join(o, "semantic_similar_words.csv"))
        assert list(got["Similar word"]) == [s for w in seeds for s, _ in sims[w]]
        _same(got[["Similarity"]], [[v] for w in seeds for _, v in sims[w]], "similar words")
        a = last["semantic"]["args"]
        thr = float(a[a.index("--best-docs-threshold") + 1])
        best, _ = mta.best_documents_for_words(m["tf_matrix"], m["tf_names"], labels, seeds,
                                               {w: [s for s, _ in sims[w]] for w in seeds},
                                               min_mean_weight=thr)
        _same(_csv(o, "semantic_best_documents")[best.columns], best, "best documents")

        o = _run(last["groups_lda"]["args"], values, tmp)
        groups, _ = mta.extract_groups_from_filenames(labels, position=2, separator="_")
        st_ = mta.compute_group_statistics(dist("lda"), groups)
        got = pd.read_csv(os.path.join(o, "group_stats_position_2.csv"), index_col=[0, 1])
        _same(got, st_, "group statistics")
        tests = mta.compare_groups_pairwise(dist("lda"), groups)
        got = pd.read_csv(os.path.join(o, "group_tests_position_2.csv"), index_col=0)
        _same(got.select_dtypes("number"), tests.select_dtypes("number"), "group tests")

        a = last["network_doc_NMF"]["args"]
        assert a[a.index("--network-min-edge") + 1] == "0.2"
        _run(a, values, tmp)       # figures only: it must run

        a = last["axis_NMF"]["args"]
        o = _run(a, values, tmp)
        coords = mta.project_documents_on_axes(app["nmf"]["doctopic"], [([0], [1, 2]), ([], [3])])
        _same(_csv(o, "axis_projection_nmf_coords")[["axis_x", "axis_y"]], coords, "axis")
        # the labels name the export columns: custom on X, the app's automatic one on Y
        ylab = a[a.index("--axis-y-label") + 1]
        assert a[a.index("--axis-x-label") + 1] == "Critique ↔ Innovation"
        ew = mta.axis_endpoint_words(app["nmf"]["topicwords"], list(m["tf_names"]), [], [3],
                                     top_n=30)
        assert ylab == "→ " + "/".join(w for w, _ in ew["right"][:3]), ylab
        exp = mta.build_axis_export_dataframe(
            labels=labels, coords=coords, axis_titles=["Critique ↔ Innovation", ylab],
            doctopic=app["nmf"]["doctopic"], topicwords=app["nmf"]["topicwords"],
            vocab=list(m["tf_names"]))
        got = pd.read_csv(os.path.join(o, "axis_export_nmf.csv"), index_col=0)
        assert ([c for c in got.columns if c.startswith("axis_")]
                == [c for c in exp.columns if c.startswith("axis_")]), "axis columns"
        print(f"  {len(last)} analyses of the app reproduced by their commands")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_round_trip()
    print("✓ round trip of the arguments")
    test_app_against_batch()
    print("✓ all tests passed")
