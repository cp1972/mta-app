# MTA -- Multi-Text Analyser

## Documentation

[Installation](install.md)

[Getting started](doc.md)

[Automation or batch usage and use with Stata](automate.md)

[Changelog](changes.md)

MTA is a Python script for topic-modelling analysis following the KISS principle, running interactively in your terminal and providing a handful of exploratory methods for topic model analysis. MTA can handle almost every kind of text sources from short words' lists to complete works or big collection of files (so called big data). It works in Windows and iOS via Anaconda, and as is in Linux and BSD operating systems.

## MTA does

 - Topic model analysis with NMF and LDA algorithms; word and document embeddings with the Word2Vec algorithms
 - Cross validation metrics for topic model analysis based on clustering algorithms aiming at guessing the best number of topics for a particular corpus
 - Similarity analysis of documents, words, semantic clusters and given selected words by the user based on words embeddings
 - Visualisation of results with high quality plots saved in a MTA-folder during the analysis, which you can easily redesign to your liking;
 - Results saved as csv files to be plotted or analyzed with third-party applications
 - Runs from within Stata with version 1.1 and above, and can be automatized with a text file or a do-file in Stata.

MTA analyses corpus in most common languages and it can be calibrated to support more languages -- for example, MTA has been adapted and used by PhD. scholars for analyzing files in Albanian, Japanese, Polish and Farsi languages.

MTA outputs plots in English, German or French as illustrations for scientific reports as well as csv files to use as tables in scientific reports, or to rework with other applications.

MTA is a research project. It has given life to other tools used in order to extract most of the important informations of text collections from simple sets of bash scripts to elaborated multi-platform R and python scripts. Interested in such tools or our research trend? Let's get in touch, drop us a mail.

## If you want to quote this software in your publication

Papilloud, C., 2017-2023, MTA: Multi-Text Analyser, http://soziologie.uni-halle.de/professuren/theorie/

BIBTEX bibliographical recording:

@misc{Papilloud1721,
author = {Papilloud, C.},
title = {MTA: Multi-Text Analyser},
howpublished = {\url{http://soziologie.uni-halle.de/professuren/theorie/}},
year = {2017–2023}
}

BIBTEX bibliographical recording for the related book to this software and the method used for topic model analysis:

@book{PapilloudHinneburg2018,
address = {Wiesbaden},
author = {Papilloud, C. and Hinneburg, A.},
publisher = {Springer},
title = {Einführung in die qualitative Analyse von Texten mit Topic-Modellen},
year = {2018},
}
