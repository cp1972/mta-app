# Automate the use of MTA and use it with Stata

You can automate the use of MTA easily in order to optimize your workflow. To do this, you have two ways.

The first one is to do an analysis with MTA which will be saved in the MTA log file. You can take this log files and copy from it your inputs into a text file.

The second way is to parse the "print" inputs from the MTA script (f.ex. with grep: grep "print" MTA.py), and to edit the corresponding relevant value in a separate text file. You can edit your text file to have a finale input file like the following one:

```
/home/cpsozhome/Koenigsteinerschl/Textanalyse4/ZEIT/*
y
/home/cpsozhome/Stopwords/de.txt
5
n
de
a
1
n
4
n
2
3
y
4
erklärung deutschland corona
0
```

When your input text file is ready, you pass it to MTA with a redirection pipe ("|") like this

```
cat myinputs.txt | python3.8 MTA.py <-- Linux and MacOS
type myinputs.txt | python3.8 MTA.py <-- Windows
```

The main advantage to use an input text file is that you can easily keep track of your multiple analysis, you can comment them in the text file, and you can distribute your text file to collaborators who will be able to reproduce your analysis.

If you want to comment your myinputs.txt file to inform colleagues of what you have done, you could do:

```
# Here is the way I would like to write a comment to my input file in Linux or MacOS
#
/home/cpsozhome/Koenigsteinerschl/Textanalyse4/ZEIT/*
y
/home/cpsozhome/Stopwords/de.txt
5
n
de
a
1
n
4
n
2
3
y
4
erklärung deutschland corona
0
REM and here is the typical comment on windows
REM with a second line
```

The problem with this is that you have to remove these comments when you pass the input file to MTA because otherwise, MTA will take the first line, which is a comment, and take it as if it where the first line it needs to process -- in this case the path to your files. MTA will stop, because it wont find any files at '# Here is the way...' which is not a path to your files.

The solution to keep comments on your input file and pass it to MTA without the comments is the following one:

```
grep -v '#' myinputs.txt | python3.8 MTA.py <-- Linux and MacOS
fstring -v 'REM' myinputs.txt | type myinputs.txt | python3.8 MTA.py <-- Windows
```

# Use MTA from inside Stata

If you want to couple MTA to Stata (16 and above), it is easily done thanks to the new functionality of Stata which support python. You can call MTA from a Stata Command window like this:

```
python script /path/to/your/MTA.py <-- Linux and MacOS

python script C:\path\to\you\MTA.py <-- Windows
```

Stata provide the commands python search an python query to find your python installation and set the path to the python executable. Please, look at the Stata documentation or [this blog post](https://fintechprofessor.com/2019/06/30/quick-setup-of-python-with-stata-16/) in order to set your path the right way.

MTA will work as usual, outputing its files in the MTA directory it has created on your hard drive.

Using a Stata do-file, you can also automate the processing like this. Somewhere in your do-file, write f.ex. the following code block with path to MTA, to your corpus and your input variables corresponding to the input statements in MTA:

```
python script /path/to/your/MTA.py
/home/cpsozhome/Koenigsteinerschl/Textanalyse4/ZEIT/* <-- path to your corpus
y
/home/cpsozhome/Stopwords/de.txt
n
12
n
de
1
n
4
n
2
3
y
4
erklärung deutschland corona
0
```

Add you comments to the do-file either before or after this code block. Then you can execute the code block in Stata which will run MTA in batch mode.
