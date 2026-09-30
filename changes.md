# Major changes in versions of MTA

## MTA version 3.4 -- May 2026 -- Minor release

User-facing simplification of the axis analysis feature. The 3.2 release introduced "Axis projection" (visual scatter), the 3.3 release added "Axis statistics" (enriched export + ANOVA) as a separate page/action/menu. After user feedback that the strong overlap between the two — same model choice, same multi-select widgets for the topic poles, same custom-label fields — was confusing students who had to redefine their axes twice, the two are now unified everywhere.

### What's new

  - **Streamlit: one page with tabs.** The former pages 8 ("Axis projection") and 9 ("Axis statistics") are merged into a single page **"Axis analysis"**. The axes are defined once in a shared header section, and the user then chooses between two tabs: **📍 Projection (visual)** — interactive Altair scatter, endpoint annotations, publication-ready matplotlib export — and **📊 Statistics (ANOVA + enriched export)** — group factor selector, summary table with both classical F + Tukey HSD and Welch F + BH-corrected t-tests side by side, convergence indicator per axis, per-axis pairwise comparisons, boxplots. The page file is renamed `8_Axis_analysis.py`; the previous `9_Axis_statistics.py` is removed.

  - **CLI: one action `axis-analysis`** that does everything in a single pass (the topic model is fit once, the projection figure and the enriched export + ANOVA tables + boxplots are all produced together). The former actions `axis-projection` (3.2) and `axis-stats` (3.3) remain accepted as **deprecated aliases**: scripts that use them keep working, but emit a deprecation notice on stderr. They run the unified action, which is a superset of what they used to do (so no script breaks; some get more output than before).

  - **Interactive menu: one entry 7 "Axis analysis"** instead of separate 7 and 8. After collecting the axes, the menu asks: "Projection only / Statistics only / Both (recommended)" so users still have fine-grained control without seeing the same configuration screens twice.

### Why this matters pedagogically

  - The fusion reflects the **conceptual unity** of the operation: once you have defined what your axes mean, looking at them visually and testing them statistically are two ways of asking the same question. Splitting them into two pages suggested they were two separate analyses requiring two separate workflows.

  - The fusion also reflects what the **third lecture session** argues (*Erwartung 1* nuanced): topic-modelling outputs can serve as the basis for both qualitative interpretation (the visual axis) and quantitative inference (the ANOVA on the same axis). Having both behind a single page makes the bridge concrete rather than splitting it into "the qualitative side" and "the quantitative side" of the same tool.

### Backward compatibility

  - All scripts using `--action axis-projection` or `--action axis-stats` keep working. They produce a superset of their previous output (projection now also produces the enriched export and ANOVA; stats now also produces the projection figure), since fitting the topic model is the costly step and producing both downstream outputs is essentially free.

  - The internal Python functions `_action_axis_projection` and `_action_axis_stats` are replaced by a single `_action_axis_analysis`. If you have your own scripts calling `mta_core.build_axis_export_dataframe`, `axis_anova_one_way`, `plot_axis_anova_boxplots`, `plot_axis_projection` or `project_documents_on_axes` directly — all of these are unchanged.

## MTA version 3.3 -- May 2026 -- Minor release

Axis statistics — the natural follow-up of the Axis projection feature (3.2). Once documents have been projected onto user-defined semantic axes, this release lets the researcher run a one-way ANOVA on each axis to test whether documents from different groups (derived from filenames) have significantly different positions. It also enriches the CSV export so that the projection results can flow directly into Stata or R for further analysis.

### What's new

  - **Three new functions in `mta_core.py`**: `build_axis_export_dataframe()` builds a DataFrame with document, metadata, axis coordinates, dominant topic and all K topic weights — ready to load into Stata/R. `axis_anova_one_way()` runs BOTH classical F-test with Tukey HSD post-hoc AND Welch's robust F-test with Benjamini-Hochberg-corrected pairwise t-tests, plus the effect size η² (proportion of axis variance explained by the group factor). `plot_axis_anova_boxplots()` draws boxplots of axis coordinates per group, one subplot per axis, with Tab10 colors per group.

  - **Why two tests side by side.** The classical F-test (with Tukey post-hoc) is the textbook epistemological reference, but assumes equal variances across groups. Welch's F-test (with Welch t-tests + BH for pairs) drops that assumption. Showing both lets the user see *convergence* (both significant or both not — robust conclusion) and *divergence* (variance heterogeneity should be discussed; the Welch result is more trustworthy in this case). This is the same statistical pattern as in the "Group comparison" page (Welch + BH on raw topic weights), extended here to the user-defined axes.

  - **Streamlit page 9 — "Axis statistics"**: reuses the same axis-definition UI as page 8 (multi-selects for poles, custom labels, 1–3 axes), then adds a group factor selector (filename position + separator + minimum group size). Displays the enriched export preview with a download button, an ANOVA summary table (F and p for both tests, plus η² for each axis), per-axis convergence indicator (green for "both significant", blue for "both not", yellow for divergence), pairwise tables for Tukey and Welch+BH side by side, and the boxplots with matplotlib export.

  - **Batch action `axis-stats`** for `MTA_v3.py`. Same axis-definition options as `axis-projection` (reuses `--axis-x/y/z` and `--axis-{x,y,z}-label`), plus two new options: `--axis-stats-group-position` (default: same as `--group-position`, the factor used elsewhere in MTA; can be overridden to test against a different filename position) and `--axis-stats-min-group-size` (default: 3; drop groups smaller than this).

  - **Interactive CLI menu entry 8** — same logic from the terminal: prompts for axes, group position, separator, minimum group size, then prints the ANOVA summary to the console and saves all four tables + boxplots PDF/PNG.

  - **Pedagogical paragraph** in `MTA-for-Master/README.md` showing how to use the enriched export in R (`lm(axis_x ~ group_pos2, data=df)`) and in Stata (`regress axis_x i.group_pos2`). The point: once the axis coordinates are exported, they become ordinary continuous variables that can serve as either dependent or independent variables in any statistical model — the methodological bridge between qualitative and quantitative work that the third lecture session argues for.

### Internal changes

  - `axis_anova_one_way()` implements Welch's F manually since scipy doesn't ship it as a single call (only as a step within `f_oneway`). The implementation follows Welch (1951): group weights $w_i = n_i / s_i^2$, weighted grand mean, and the corrected denominator with the $(k^2-1)/3$ term.
  - All four CSV outputs (`axis_export`, `axis_anova_summary`, `axis_anova_welch_pairwise`, `axis_anova_tukey_pairwise`, plus `axis_anova_group_summary`) use sanitized column names (alphanumeric + underscore, capped at 30 chars) so that they load cleanly in Stata.

## MTA version 3.2 -- May 2026 -- Minor release

Axis projection — user-defined semantic axes on the doctopic matrix — is added to all three interfaces (Streamlit, interactive CLI menu, batch mode). This is a new analytical tool, not just a visualization: it lets the researcher project documents onto 1, 2 or 3 axes that they define themselves as oppositions between pools of topics.

### What's new

  - **Three new functions in `mta_core.py`**: `axis_direction_vector(n_topics, left_pole, right_pole)` builds a normalized contrast vector ($+1/|R|$ on the right pole, $-1/|L|$ on the left pole, 0 elsewhere); `project_documents_on_axes(doctopic, axes)` projects documents onto 1, 2 or 3 user-defined axes; `axis_endpoint_words(topicwords, vocab, left_pole, right_pole, top_n)` aggregates topic-word weights to identify the most characteristic words at each axis extremity. Plus `plot_axis_projection(...)` for matplotlib-based 1D, 2D and 3D rendering.

  - **Why this matters methodologically.** Automatic PCA on the doctopic matrix picks the directions of maximum variance — often hard to interpret. Axis projection inverts the logic: the researcher defines each axis as an opposition between two pools of topics, and the documents are projected onto these *interpretable-by-design* axes. This brings into MTA the spirit of Bourdieu's correspondence analysis (axes as oppositions, not as variance directions) and of Slapin & Proksch's text scaling (Wordfish/Wordscores). The same operation provides a methodological bridge between qualitative and quantitative work, in line with the following argument: the resulting coordinates can be used as continuous variables in regression, clustering, or other quantitative analyses.

  - **Streamlit page 8 — "Axis projection"** with: a topic browser to help the user pick informed poles, multi-select widgets for the left and right poles of axes X, Y, Z (1 to 3 axes), per-axis custom-label fields, a display-options panel (top words / topic names / both at extremities, dot coloring by dominant topic / group from filename / none), an Altair interactive view for 1D and 2D (zoom, pan, hover), a downloadable coordinates table (CSV), and a "Publication-ready export" expander that renders the same projection with matplotlib for PDF/PNG download.

  - **Interactive CLI menu entry 7** — same projection from the terminal: shows the topic summaries, asks the user for axes 1–3 (one at a time, format `"LEFT / RIGHT"`), optional custom labels, color choice (dominant topic / group / none), then generates the matplotlib figure plus CSV/JSON.

  - **Batch action `axis-projection`** for `MTA_v3.py`. New options: `--axis-method {nmf,lda}`, `--axis-x "0,1 / 2,3"`, `--axis-y "..."`, `--axis-z "..."`, `--axis-x-label`, `--axis-y-label`, `--axis-z-label`, `--axis-color-by {dominant-topic,group,none}`, `--axis-endpoint-words N`. The action is *not* included in `--action all` because it requires user-defined poles — there is no sensible default.

### Internal changes

  - `Sequence` added to typing imports in `mta_core.py` to support the new function signatures.
  - All new code is fully self-contained: no new external dependencies, no changes to existing functionality.

### Post-test refinements

  - **Altair interactive view now shows endpoint annotations.** The first release of axis projection only annotated the matplotlib export with the endpoint words/topics; the Altair interactive chart was bare. The chart now displays top words (or topic names, or both, per the user's choice) at all axis extremities: vertically stacked at the left and right edges for the X axis, horizontally tiled (5 per row, wrapping into bands near the top and bottom edges) for the Y axis. The horizontal tiling on the Y axis prevents the long vertical stacks from intruding into the scatter cloud at the center of the plot.
  - **Default number of endpoint items raised** from 5 to 15, with the slider range extended from 5–30 (was 3–15). On real corpora, 5 words at each extremity is rarely enough to characterise a pole — 15 is a better default, and 30 is the practical upper bound before the figure becomes unreadable.
  - **Mode "Both" no longer loses the topic names.** A bug in the matplotlib helper `_box(words, n_top)` was re-slicing the list to `n_top` even when the caller had already prepared a mixed `top_words + separator + topic_names` list — which silently truncated the topic names. The slicing is now done entirely by the caller, so "Both" mode shows both groups as intended.
  - **Group coloring now works in axis projection.** The Streamlit page, the batch CLI action and the interactive menu all called `extract_groups_from_filenames(...)` and treated its return value as a DataFrame with a `.columns` attribute. The function actually returns a `(dict, list)` tuple — `(filename → group_code, skipped_filenames)`. Three call sites were rewritten to unpack the tuple correctly, report skipped files to the user, and build the color list in the order of `labels` so that the scatter coloring aligns with each document. The user-visible error `'tuple' object has no attribute 'columns'` is gone.

## MTA version 3.1 -- May 2026 -- Minor release

Network views — bipartite graph visualizations of the topic model — are added to all three interfaces (Streamlit, interactive CLI menu, batch mode).

### What's new

  - **Three publication-ready bipartite network graphs.** After running NMF or LDA, MTA can now render the topic model as: (a) `Topic ↔ Document` — each document connected to the topic(s) it weighs strongly on; (b) `Topic ↔ Top-N words` — each topic connected to its most representative words; (c) `Combined` — topics, documents (as circles) and words (as squares) on the same canvas. Topic node sizes encode the cumulated weight attached to each topic; edge thicknesses encode the strength of each link.
  - **ForceAtlas2 layout.** Same algorithm as Gephi, via `fa2-modified`. Produces organic node placements where strongly connected nodes cluster together. Falls back to NetworkX spring layout if the package is unavailable.
  - **Solarized color palette** (Ethan Schoonover) for warm, publication-grade colors that work on white backgrounds. Curved edges (arc3, rad=0.15) replace straight lines, making overlapping edges easier to follow.
  - **`emphasize_differences` toggle.** Two size-encoding modes for topic nodes: faithful (sqrt scaling — on a balanced corpus, nodes look similar; that's the data) or stretched (min-max — the smallest and largest topics are pulled apart so even modest differences become visible). The toggle is exposed in the Streamlit page, the interactive menu (question on prompt) and the CLI flag `--emphasize-differences`.
  - **Streamlit page 7 — "Network views"** with three tabs (one per graph kind), interactive sliders for the minimum edge weight and top-N parameters, and per-graph PNG + PDF download buttons.
  - **Interactive CLI menu entry 6** — same three graphs from the terminal, asking the user for the method (NMF/LDA), kind of graph, top-N, threshold and emphasis flag.
  - **Batch action `network`** for `MTA_v3.py`. New options: `--network-method {nmf,lda}`, `--network-kind {doc,word,combined,all}`, `--network-top-n N`, `--network-min-edge X`, `--emphasize-differences`. The action is also included when `--action all` is used.
  - **Semantic cloud — configurable label density.** The 2D semantic cloud previously labelled the 15 closest words to each seed (hard-coded), which left some users wondering why most points stayed unlabeled on dense clouds. The cap is now exposed via a new CLI flag `--max-labels-per-cluster N` (default 15), a prompt in the interactive menu, and a slider on the Streamlit page (Section 2 → "Publication-ready export"). A small subtitle is also printed on the figure ("Labels shown for the top-N closest words to each seed…") so the choice is documented in the output itself.

### Internal changes

  - `plot_semantic_cloud` was moved from `MTA_v3.py` to `mta_core.py` so that both the CLI and the Streamlit page can call it for publication-ready static exports. A thin wrapper under the old private name `_plot_semantic_cloud` is kept in `MTA_v3.py` so existing call sites keep working.

### Bug fixes

  - **Windows launcher (`start_MTA.bat`).** The Windows menu launcher had Unix-style line endings (LF only) instead of Windows-style (CRLF). `cmd.exe` parses multi-line `.bat` files that contain labels (`:menu`, `:streamlit`, …) and `goto` instructions correctly only when lines end with CRLF; without it, the file was parsed as one long line and produced cryptic errors such as `'ho' n'est pas reconnu en tant que commande` (truncations of `echo`) or `'ot' n'est pas reconnu` (truncations of `not`). Fixed by reconverting all `.bat` files to CRLF and adding a `.gitattributes` file at the repository root that pins line endings per file extension: `*.bat` → CRLF, `*.sh` / `*.command` → LF, `*.py` → LF, binary types (PDF, PNG, ZIP, …) are marked as binary so Git never touches them. This guarantees correct line endings on every fresh clone, regardless of OS or `core.autocrlf` setting.

  - **macOS launchers (`*.command`, `*.sh`).** The macOS double-clickable launchers and the Unix shell scripts lacked the Unix executable bit (mode `0644` instead of `0755`). Users would see misleading error messages mentioning insufficient privileges, with nothing visibly wrong in the Finder's permissions panel (which hides the Unix execute bit by default). Symptom: a `.command` file refuses to run even after right-clicking → Open → Confirm to bypass Gatekeeper. Fixed by setting `chmod +x` on all `.command` and `.sh` files and committing the executable bit so that future `git clone` operations preserve it. The ZIP distribution also carries the bit, so users extracting the ZIP on macOS or Linux get working launchers out of the box. *Users who already cloned the repository before this fix* can simply run, in a Terminal inside the `MTA-for-Master/` folder: `chmod +x *.command *.sh`.

The visualization engine lives in a new module `mta_network.py`. New dependencies: `networkx>=3.0`, `fa2-modified>=0.4`. Both are added to `requirements.txt`. No existing functionality has been changed.

## MTA version 3.0 -- May 2026 -- Major release

Version 3.0 is a full reorganization of MTA. The single `MTA.py` terminal script has been split into a reusable engine and two front-ends. Older `MTA.py` versions (≤ 2.0) remain available in the `archive/` directory of this repository for users who still rely on them.

### What's new

  - **Three ways to run MTA, one engine.** All analyses now go through `mta_core.py`, a pure-Python library with no I/O side effects. On top of it sit (a) a Streamlit web app for click-and-drag use, (b) a modernized CLI (`MTA_v3.py`) with both an interactive menu and a non-interactive batch mode for Stata/R/shell pipelines, and (c) double-clickable launchers for Windows, macOS and Linux. The original `MTA.py` is no longer the entry point.
  - **No more Anaconda dependency.** The new installers download an isolated Python 3.12 via `uv` directly into the MTA folder. Nothing is touched outside that folder; uninstalling means dragging the folder to the trash. This replaces the Anaconda-based installation path that was needed on Windows and macOS.
  - **Streamlit web app.** A six-page workflow runs locally in the browser: Load corpus → Topic models → Word weights → Topic evolution → Semantic context → Group comparison. Data never leaves the user's machine. Pages are locked until their prerequisites are met to guide first-time users.
  - **Group comparison (new analysis).** Compares topic-weight distributions across groups (e.g. F vs M, age bands, sources) using Welch's t-test with Benjamini-Hochberg correction. Box-plots are generated automatically for topics that survive correction. Groups can be defined either from filename patterns or from a separate CSV.
  - **Topic evolution as a first-class step.** What used to be a side feature ("year stamp at the beginning of files") is now a dedicated page/action with rolling means and optional yearly aggregation.
  - **Co-occurrence embeddings as default.** The semantic-context analysis now defaults to a co-occurrence + PCA method that needs no external dependency. Word2Vec via gensim remains available as an option but `gensim` itself is now optional, which lightens installation significantly.
  - **Multilingual chart output.** A single language toggle (English/French/German) propagates to every axis label, legend and title across both interfaces.
  - **Exports.** Every table is exported as both CSV and JSON (split orientation, easy to read from Stata/R); every figure as both PDF and PNG.
  - **Batch / scripting mode rebuilt around arguments.** `MTA_v3.py --corpus … --stopwords … --action {nmf,lda,evolution,word-weights,semantic,compare-groups,network,all}` replaces the input-piping technique documented in the old `automate.md`. Existing Stata workflows that piped a text file into `MTA.py` should be ported; the old method still works against the archived `MTA.py`.

In short, version 3.0 keeps the analytical core of MTA 2.0 (NMF, LDA, cross-validation with Cophenet, Word2Vec-style similarities) and rebuilds everything around it: installation, user interface, scripting interface, and outputs.

## MTA version 2.0 -- January 2025 -- Major release

  - Rework of the crossvalidation tests -- the results of the tests are displayed directly to the user with the best number(s) of topics including Elbow, Silouhette, Calinski Harabasz and Davis Bouldin scores.
  - We have added the Cophenet correlation coefficients for NMF and LDA to the crossvalidation scores, enabling the user to see converging results through all the tests and the Cophenet values.
  - Ergonomic improvement: in comparison to older versions of MTA, the user can continue the analysis without looking at the 'Cluster_Metrics' plot and without the need to re-run MTA to apply its optimal number of topics.
  - Accordingly, we have updated the 'Cluster_Metrics' plot, adding two plots translating the results of the Cophenet values for NMF and LDA.
  - We have written two functions to catch the turning points in each crossvalidation tests and in the Cophenet values for NMF and LDA to output the best number(s) of topics to the user.
  - We have removed BERT Topic-Modelling method, which has not been useful for users in our tests; therefore, we avoid the difficulties related to the installation of hdbscan, as well as the computing costs of BERT models, which are high on low-end hardware.

In sum, this version of MTA comes with better crossvalidation tests, complete linkage to Cophenet values to advise users immediately about the best number(s) of topics they can use to compute an optimal topic model with NMF and/or LDA, without leaving the interface of MTA. It is also lighter and easier to install than the previous 1.8 and 1.9 versions of MTA, particularly on Windows systems.

## MTA version 1.9 -- September 2024 -- Minor release

  - Lazy loading of modules for better RAM management and better code execution.
  - Cosmetic changes for heatmaps comparing NMF and LDA topics -- explicit labels on x and y axes, smaller fonts for labels.
  - Better regexp for parsing corpus.
  - Conditional statement for BERT -- MTA checks automatically if you seem to have enough texts to perform a BERT evaluation of best number of topics.
  - Rewrite the function to crossvalidate the optimal number(s) of topics to speed up crossvalidation; take two new tests (naive Elbow and Calinski Harabasz) and reject two old ones (BIC and Gauss).
  - Implement progress bar in crossvalidation operations.
  - Suppression of the stdin output of crossvalidation since it slows the process and does not add anything really useful to the interpretation of optimal number(s) of topics.
  - Word2Vec takes more similar words to your input words in menu entry 4.
  - New facility to save list of files corresponding to cluster of words provided with menu entry 4; with this list, you can retain only the files corresponding to the cluster of similar words attached to word(s) given at the MTA prompt in this menu. You can then perform a topic analysis on these selected documents. For sh/bash/zsh/fish user, you could do the following to copy the needed files mentioned in the list to a new directory:

    - make a new directory: mkdir mynewdir
    - use the following onliner: for file in `cat BestFiles_ChoosenWords_dateofthefile.csv`; do cp "$file" /path/to/mynewdir ; done

This will copy the files listed in the document BestFiles_ChoosenWords_dateofthefile.csv to your mynewdir. Replace "BestFiles_ChoosenWords_dateofthefile" with the name of the appropriate file in your MTA-Results directory (you might have several of them).

## MTA verion 1.8 -- September 2023 -- Minor release

  - New function to better catch errors due to the use of BERTopic models in case of too few remaining vocabulary: MTA does not crash anymore if you have to few words to build topic models; this function enables to get rid of the minimal amount of texts that we use to apply/not apply BERTopic; keep in mind that we are not using BERTopic as a model, rather as an estimation method providing the maximal amount of topics in your dataset;
  - Improve the documentation: it mainly regards a workaround to install BERTopic on Windows -- with anaconda, you have to install hdbscan from conda forge _before_ installing bertopic with pip (pip install bertopic).
  - Now MTA version is called 'MTA.py' only.

## MTA3-1.7 -- March 2023 -- Major release

MTA3-1.7 is a major release with significant improvements/changes compared to other MTA versions and further refinement of RAM management for the analysis of big corpora.

Overview of major changes:

  - **new Bertopic models** (neural language models or so-called Large Language models from Google) to estimate the minimal/maximal number of topics in your corpus if you have a significant corpus (big vocabulary) -- **Please refer to the install document to know how to deal with the installation of bertopic and hdbscan**;
  - put the word2vec model at the specific menu 4 to generate it on demand if you want to use the utilities in this menu; with this, we improve our RAM management drastically! As an example, analyzing 15.000 newspapers articles as with MTA3-1.6.py (see below) requires now less than 3 GB RAM, and MTA scales significantly better when menu 4 is used
  - simplification of the outputs, i.e. wordclouds have been removed; there are several reasons for this, because the generation of wordclouds is computer intensive for no noticeable analytical gain, and because you can generate your clouds outside of MTA quickly in our days, using the weight of words generated with MTA btw;
  - rewriting the function for the kmeans++ estimates, enabling the interpretation of the best number of topics; this function now performs the given tests, skipping the gaussian mixture if you have more than 2000 documents (because it is too slow on regular desk computers).

## MTA3-1.6 -- November 2022 -- Minor release

MTA3-1.6 is a minor release with slight improvements in the tuning of the algorithms and the gain in RAM for analysing big corpora.

Overview of significant changes:

  - simplification of the tuning parameters; it is now relatively obvious to perform fine-grained analysis of your corpus with the knowledge that you have about your data;
  - removing word2vec as a complementary cross-validation method because other cross-validation methods perform better at the level of guessing of adequate topic number;
  - gain in performance regarding the storing of data in RAM: analyzing 15.000 newspapers articles now requires less than 7 GB RAM, which is quite a performance compared to MTA-1.5
  - further correction of layout and text in MTA.

# Deprecated -- History

## MTA3-1.5 -- March 2022 -- Major release

MTA3-1.5 is a major release with several improvements in tuning the algorithms, the plots and the code itself, consuming fewer resources.

Overview of major changes:

  - the code has been improved to support more texts while remaining kind with RAM by systematically deleting objects not used anymore in the analysis, f.ex. analyzing 15.000 newspapers articles now requires less than 12 GB RAM, a significant improvement over previous versions of MTA;
  - the code has been improved with more parts of the code being packaged into functions -- better readability and better management of the code;
  - the scikit-learn and word2vec algorithms have been more fine-tuned to better adapt to the number of texts under investigation, giving better results both for topic analysis as for words and documents embeddings;
  - the user gets some advice to fine-tune the scikit-learn algorithm based on simple rules;
  - the training of the word2vec algorithms has been improved from 5 (default) to 10 iterations (epochs), which gives slightly better clustering results;
  - the plot displays better, with better handling of x-axis depending on the quantity of data under investigation while giving more choice for the user (kind of font and fontsize);
  - above 2000 documents, BIC and Log-Likelihood have been dropped out of the metrics to prevent slowing down MTA too much;
  - some verbosity has been removed to ease the readability of the results during the analysis (as f.ex. the output of files taken in the analysis is reduced to the three first files in your directory, etc.).

## MTA3-1.4 -- December 2021

MTA3-1.4 is a minor release where the most important changes regard better plotting utilities and better handling of word2vec to avoid overfitting and underfitting models.

Overview of major changes:

  - w2vec models have been calibrated to better fit the properties of the vocabulary in your data;
  - plots' new utilities enable you to print most of the plot in MTA (excluded: wordcloud plots and similarity plots) with one of the most used fonts in scientific publications (provided the fonts have been installed on your computer) and with a better rendering of fonts size in the plots
  - similarity heatmaps get other colours
  - under the hood: large parts of the code have been rewritten into functions to keep the script light and swift; comments in the script have been further corrected

## MTA3-1.3 -- September 2021

MTA3-1.3 is a minor release where the most critical change regards the adaptation to the updated gensim package (4.+).

Overview of significant changes:

  - w2vec models are now compatible with the new syntax introduced with gensim 4.+
  - cleaning the code where it should have been cleaned for a long time :)

## MTA3-1.2 -- February 2021

MTA3-1.2 is a minor release where the most critical change regards the better handling of stopwords.

Overview of major changes:

  - better handling of stopwords and stoplist: it affects mainly Windows and MacOS users working with languages other than English whose stopwords/stoplists were not read with the correct UTF-8 encoding;
  - better handling of value errors: these errors show up when you are working with few texts

## MTA3-1.1 -- June 2020

MTA3-1.1.py is the MTA version for python 3.x. In this release, we have introduced new functionalities to automate the execution of MTA and to run it from a do-file within the Stata16 application. Please see [[automate|the corresponding introduction on how to automate your use of MTA]].

Overview of major changes:

  - Better handling of NMF and LDA analysis throughout MTA -- now MTA ask you if you want to perform a LDA analysis; if not, it won't ask again in other menu entries.
  - Remove the multiplots -- multiplots for NMF or LDA analysis with more than 100 documents have been removed since the plot quality is not good and because they slow down the analysis.
  - Remove the correlation plot for more than 50 documents -- the correlation plot has been removed if you perform an analysis with more than 50 documents since it slows down the program too much and it produces a plot of bad quality (almost unreadable).
  - Introduce new Weight of Topics barplot for documents with year stamp -- this plot replaces the multiplots for NMF and LDA analysis when you have more than 100 documents and a year stamp at the beginning of your files (i.e. YYYY-File.txt). Such a plot lets you see the evolution of the weight of your topics through the years based on the mean of those weights per year.
  - Better rendering of best words per topic, best sentences per topic, word2vec associated words to given terms using pandas dataframes.
  - Drop a regexp added to TfidfVectorizer of scikit-learn which was too aggressive in rejecting words from the vocabulary.
  - Added an inverse normalization after the .fit_ method of the nmf algorithm to get better results when using network csv files with third applications (f.ex. gephi). Applied this inverse normalization to the wordclouds plots.

## MTA-1.0 -- December 2019 -- Major release

MTA-1.0 is a significant release. The software has been completely restructured, resulting in less code and more functionalities. Also, with this release, we are eventually jumping into python 3.x exclusively -- we provide a version of MTA-1.0 for python 2.7, but it is the last one.

Overview of major changes:

  - The menu has been simplified to four entries (instead of eight until now). This results in putting more software facilities into the topic modelling (first menu entry) to have more cross-validation regarding the results of NMF and LDA topic models.
  - Matrices: we now have better handling of corpora (tokenization and vectorization), resulting in using the same corpora during the topic analysis (algorithms agnostic corpora). This is an important step forward because it enables us to better compare the /different/ behaviour of the algorithms on the /same/ corpus of texts instead of comparing how the algorithms handle differently the same corpus
  - LDA algorithm: we dropped gensim for scikit-learn LDA algorithm (also based on Gibbs sampler) -- this gives us the possibility to unit the methods in MTA, and to provide each analysis for both NMF and LDA algorithms; we were not very happy with the implementation of LDA in Python via gensim, and still, we are not very happy with this implementation via scikit-learn, which seems to work well for short texts, but not so well for large corpora.
  - Estimation of the best number of topics: we have now five metrics (before MTA-1.0, only two) to help in the decision regarding the best number of topics, including silhouette (for Kmeans++ and Word2Vec), [Davis Bouldin](https://scikit-learn.org/stable/modules/clustering.html#davies-bouldin-index), [BIC and Log-Likelihood](https://scikit-learn.org/stable/modules/mixture.html)one plot to summarize all results easing the comparison)
  - Every plot and csv output related to both algorithms are saved in your MTA folder as usual
  - Wordclouds for all NMF and LDA topics and per NMF and LDA topic saved as pdf files in your MTA folder; this functionality has been added since MTA version 0.9, and it has been correspondingly extended
  - TSNE for word embeddings: we now use Word2Vec with TSNE and the [propagation algorithm](https://scikit-learn.org/stable/modules/clustering.html#affinity-propagation|affinity) to plot semantic similarities for chosen words of the corpus, and to inspect semantic clusters based on these chosen words
  - PyLDAVIS: it has been dropped because of the [bug](https://github.com/cpsievert/LDAvis/pull/41) related to the _.prepare method of pyLDAvis in scikit-learn regarding the red bars -- they do not give the estimated frequencies of words qua topic (The implementation is there in MTA for both NMF and LDA, and as soon as this bug has been corrected, we will reintroduce pyLDAvis as another way to look at the results of the topic model analysis)
  - NLTK has been dropped in favour of its own tokenizer functions and filtering of stopwords
  - MTA-1.0 Python 3.x only: Reintroducing a modified version of best sentences per topic for NMF and LDA algorithms
  - MTA-1.0 Python 3.x only: Comparison of first 50 topic words for NMF and LDA algorithms -- enables to see the percentage of common words and to establish a correspondence between the topics of both algorithms
  - MTA-1.0 Python 3.x only: Reformatting of some .csv outputs for better readability and direct import in Stata and Gephi
  - Corrections of orthographic mistakes of utf-8 in some of the matplotlib results -- note: it remains problematic in python 2.7x to keep a good utf-8 encoding/decoding of a corpus with special characters; this is one of the reasons why we are moving to python 3.x as unique development language from now on. The other one is that python 2.7x will not be further updated
  - Corrections of users and deprecation warnings (Python 3.x) -- avoid junk lines in terminal output

## MTA-0.9 -- August 2019

  - Wordclouds for all NMF topics and per NMF topic saved as pdf files in your MTA folder; this functionality has been added for both the general analysis (menu 1) and the specific one (menu 2)
  - new csv file with weights of the 20 best words describing each NMF topic, enabling the use of the data with other applications (f.ex. to generate network graphs)

## MTA-0.8 -- January 2019

  - better handling of MTA-Results folder -- it is now saved with a timestamp enabling multiple runs of MTA without renaming the MTA-Results folder before each run. This way, the user keeps a kind of 'history' of his analyses
  - plots are saved in MTA-Results folder as PDF-files and are no more displayed interactively. This enables to avoid the issues with threading related to Tkinter completely, and hence possible crashes of MTA. These PDF-files have the following structure: NameOftheFile_NumberOfTopicsGivenByUser_Date_and_Time.pdf -- this enables to run MTA several times without overwriting existing results files
  - csv files saved in MTA-Results folder for further use with Excel or similar application. These csv files have the same structure as the PDF-files for plots
  - extending LDA comparison -- NMF and LDA were compared only for results regarding the analysis of a whole corpus. Now the comparison has been extended to analysing a selected corpus (selection-based words or concepts given by the user interactively). Keep in mind that this comparison applies to key features of topic models only
  - the output of BIC values and coherence score to estimate the best number of topics for NMF and LDA algorithms has been reformatted. Now you get these values directly in the Terminal and not as a plot from the optimal to the less optimal number of topics
  - menu entries 3 and 4 have been reworked as features inside menu entries 1 and 2. This enables direct access to the similarities between texts and between topics while running MTA for full or for selected corpus. The menu has been correspondingly updated

## MTA-0.7 -- November 2018

  - drop support for networkx plots which does not add much more information to topic models and does not provide a method to extract labels from Pandas dataframes anymore -- add instead a summary table showing the distribution of topics in texts and dominant topic in texts
  - drop support for nltk facilities (words associations, similarities, best con-texts for given words) -- replace instead by word2vec similar facilities with the added benefit of words embeddings (at the concept and document levels)
  - better handling of errors to guide the user through problem-solving tasks
  - MTA now makes a directory "MTA-Results" in the user's home directory to save some files generated during the analysis -- **this does not apply to plots**, which the user saves interactively during the analysis depending on his/her needs (feature)
  - correcting an annoying error with tkinter and threads in the first menu entry of MTA -- put plots back into the main thread to prevent crashes and corresponding refactoring of code parts in the first menu
  - corresponding changes reflected in MTA menu
  - translation of MTA for Python 2.7 to MTA for Python 3.x
  - improving the documentation on this wiki

## MTA-0.6 -- June 2018

  - add networkx support to plot texts against topics with weighted measures and corresponding weighted nodes and edges
  - add LDA-Topic modelling techniques from gensim package -- enables comparisons with NMF-Topic models for analysis on the entire corpus
  - add coherence measure for LDA models to predict the best number of topics for a given corpus
  - add support for LDA visualisation -- provide a Html file which can be opened in a browser to visualize results (from pyLDAvis)

## MTA-0.5 -- January 2018

  - add a module to see the evolution of topics through texts
  - add a module to retrieve the best texts given a topic
  - add a module to see the importance of a single word in all topics
  - add nltk facilities to retrieve best context expressions for two words
  - add nltk find facilities to retrieve concordances
  - add word2vec model to look for similarities between words in corpus

## MTA-0.4 -- August 2017

  - expand topic analysis with NFM algorithm to:
    - words association
    - texts classification for a given topic
    - texts classification for a given topic based on words association
  - drop more straightforward classification of documents based on similarities
  - introduce cross-validation of topic modelling with kmeans++ to predict the best number of topics for a given corpus based on BIC value

## MTA-0.3 -- May 2017

  - topic analysis with NFM algorithm for all texts
  - simple word associations and collocations
  - drop most typical words of a corpus to concentrate on typical words per topic

## MTA-0.2 -- March 2017

  - simple word associations and collocations
  - most typical words of a corpus

## MTA-0.1 -- January 2017

  - simple word associations and collocations
