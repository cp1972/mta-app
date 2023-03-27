# MTA getting started

In order to use MTA, you have to convert your files into plain text (.txt) files. This is a **very important** step, as MTA does not provide a converter out of the box which is a feature -- the user has to tailor his/her files personally in order to get results corresponding to his/her needs.

## UTF-8 recommendation

MTA takes your files and your stopwords list in UTF-8 encoding. This means that you have to provide files -- including stopword lists -- with such encoding. If not, MTA will guess the encoding of your files and remove characters that can not be translated into UTF-8 format. This will harm your analysis, and can make MTA crashing.

# Running MTA -- Most common way

**Use MTA as is**: download MTA, open a terminal of your choice where you have saved MTA, and type:

```
python MTA.py
```

If you are a Windows' user and if you have installed the Anaconda software, run MTA in an Anaconda Terminal from the Anaconda Navigator.

Again, Windows' users have to make sure that they can read and write to the folder in which they grab their text files. In case of doubt, copy your file directly in a folder under C (as f.ex.: C:\mytextfiles\).

# Specific usage for Unixes/BSD

**Use MTA by mouse-click on the MTA script**: download MTA, open a terminal of your choice where you have saved MTA, and type as administrator (root):

```
chmod a +x MTA.py
```

Then you can use MTA by just clicking with your mouse on the script and choose the option 'Open in a terminal'

**Use MTA system-wide**: download MTA, open a terminal of your choice where you have saved MTA, and type as administrator (root):

```
chmod a +x MTA.py
```

Copy MTA to /usr/local/bin and rename it like this:

```
cp MTA.py /usr/local/bin
cd /usr/local/bin
mv MTA.py MTA
```

Then open a Terminal and type as normal user at the prompt:

```
$ MTA
```

# MTA-Results folder

When you first run MTA, it informs you that it creates a directory "MTA-Results" in your user's directory in order to store files generated during the analysis (PDF-files for plots, csv files for data, and the index of your corpus in a shard format).

This folder gets a timestamp so that you can run MTA several times, and each run will save its results in the appropriate "MTA-Results" folder.

# First steps once in MTA -- Path to your files

While running MTA, you firstly have to provide the path to your data, as f.ex.:

```
/john/textfiles/* <-- for Mac OS, Unixes/BSD
```

```
C:\mytextfiles\* <-- for Windows
```

**Look at the star at the end of the path** -- it is mandatory. MTA will grab all files in your directory in order to analyse them. You can have few files or a lot of files in this directory. The important thing to remember is that all the files you want to analyse have to be in one directory **without subdirectories**.

The case of too few files: if you have, say, only one or two files in your directory, MTA won't be able to output significant results because it won't be enough for MTA to do a topic model analysis. If you want to do a topic model analysis on one or two files, the solution is to split this or these files in some more files in order for MTA to proceed with the analysis. For example: you want to do a topic model analysis on an article or on a book, then split the article according to its paragraphs, or for the book, split it into chapters.

The case of different languages: if you have files in different languages, don't put them in the same directory. Make as much directories as needed, e.g. one directory for your French files, another one for your German files, another one for your English files etc. You will have to perform an analysis for each of your directory in order to make significant comparisons between your different files. The same advice could apply for other possible variables, like for example files from different authors, files regarding different topics etc. This way, you will greatly improve the accuracy of your analysis.

## Stopwords

It is mandatory to use a stopwords file adapted to the language of your files. MTA asks you for the location of this file on your computer. Your provide the full path to your stopwords file like this:

```
/john/stopwords/englishstopwords.txt <-- Mac OS, Unixes/BSD
```

```
C:\mystopwords\englishstopwords.txt <-- Windows
```

This feature enables you to provide your own stopwords file -- you can find such stopwords files online, or you can make a stopwords file yourself: just put the words you don't want to take into account in the analysis one by line into a .txt file, and you are good to go.

## Language for plots

MTA asks for the output language you want regarding the labels of the plot generated during the analysis. You can choose between German, French and English. Other languages could be added on demand -- please keep me informed of your needs, and I will add them to MTA.

MTA can output pdf files for plots with specific fonts -- be sure that you have these fonts installed on your computer before using them.

At this point, you are good to go with your analysis.
