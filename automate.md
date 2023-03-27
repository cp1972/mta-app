# Automate the use of MTA and use it with Stata

You can automate the use of MTA easily in order to optimize your workflow. To do this, you have two ways.

The first one is to do an analysis with MTA which will be saved in the MTA log file. You can take this log files and copy from it you input into a text file.

The second way is to parse the inputs from the MTA script (f.ex. with grep), and to edit the corresponding relevant value in a separate text file.

When you have your text file ready, you pass it to MTA like this

```
cat myinputs.txt | python3.8 MTA.py
```

The main advantage to use an input text file is that you can easily keep track of your multiple analysis, you can comment them in the text file, and you can distribute your text file to collaborators who will be able to reproduce your analysis.

# Use MTA from inside Stata

If you want to couple MTA to Stata (16 and above), it is easily done thanks to the new functionality of Stata which support python. You can call MTA from a Stata Command window like this:

```
python script /path/to/your/MTA3-1.1.py <-- for Linux and MacOS

python script C:\path\to\you\MTA3-1.1.py <-- for Windows users
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
