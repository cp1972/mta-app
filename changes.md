# Major changes in versions of MTA

## MTA version 1.9 -- September 2024 -- Minor release

  - Cosmetic change for chart comparing NMF and LDA topics -- add explicit labels on x and y axes.
  - Conditional statement for BERT -- MTA checks automatically if you seem to have enough texts to perform a BERT evaluation of best number of topics.
  - Rewrite the function to crossvalidate the optimal number(s) of topics to speed up crossvalidation; take two new tests (naive Elbow and Calinski Harabasz) and reject two old ones (BIC and Gauss).
  - Implement progress bar in crossvalidation operations.
  - Suppression of the stdin output of crossvalidation since it slows the process and does not add anything really useful to the interpretation of optimal number(s) of topics.
  - Word2Vec takes more similar words to your input words in menu entry 4.
  - New facility to save list of files corresponding to cluster of words provided with menu entry 4; with this list, you can the corpus to retain only the files corresponding to the cluster of similar words attached to word(s) given at the MTA prompt in this menu. You can then perform a topic analysis on these selected documents. For sh/bash user, you could do the following to copy the needed files mentioned in the list to a new directory:

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

  - **new Berttopic models** (neural language models or so-called AI models) to estimate the minimal/maximal number of topics in your corpus if you have a significant corpus (more than 80 documents) -- **Please install bertopic library like this: pip3 install bertopic**;
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
