# MTA -- Installation

MTA does not require installation. You can clone/download the script and run it in a terminal/command window. MTA presupposes that you have Python 3.x installed, as well as other required Python packages. In this regard, MTA rests on the shoulders of other giant applications that you have to install with a specific Python installer like pip or provided by your operating system.

From 1.1.2020, MTA have been developed for Python 3.x only. MTA for Python 2.x have been dropped.

In the following, we cover different scenarios on how to install an environment which lets you run MTA depending on the system you are using, as well as the needed python packages i.e. libraries you have to install to run MTA.

### Windows and Mac OS users -- Install Anaconda and the needed libraries for MTA

For Windows and Mac OS users not comfortable with the command line, I recommend using [Anaconda for windows](https://www.anaconda.com/download/#windows) [or for Mac](https://www.anaconda.com/download/#download) which installs for 32 or 64 bits architectures. You can then use the conda installer to install the requested packages for MTA -- note that most of the essential packages which MTA requires are already provided by Anaconda, so you only have to follow the steps below to install the missing packages:

  * open Anaconda Navigator; your should see 'Environments', and 'base(root)' on the top right;
  * click on base(root) and on 'open Terminal' in the expanded menu -- choose the Anaconda Terminal (called 'Terminal'), not the Python Terminal;
  * in the Terminal, install the following packages with conda like this:

```
conda install gensim python-louvain
```

In the Terminal, install the 'community' package with pip like this:

```
pip install community
```

It might be that windows users -- depending on their version of windows -- have to install qt5; from the Anaconda Navigator using the Terminal, type:

```
pip install python-qt5
```

With the new bertopic feature, the installation might throw errors with hdbscan. One solution is to install python-dev-tools with pip and hdbscan with conda, like this:

```
pip install python-dev-tools --user --upgrade

conda install -c conda-forge hdbscan
```

Then, install bertopic with pip:

```
pip install bertopic
```

After that, download MTA and open it from the Anaconda Navigator by using the Terminal -- there, too, make sure that MTA is in the folder where the Terminal opens (f.ex. in 'C:\Users\Downloads' if the Terminal shows a prompt like '<base>C:\Users\Downloads'); run MTA by typing in the Terminal (here, if you are using Python 3.8):

```
python3.8 MTA.py.
```

MTA will crash if your path is not writable for the user -- make sure you have sufficient permission on your path before processing. In doubt, ask your admin.

**For old version of MTA before 1.7 only**:

  * you miss two packages that you have to download [at Christoph Gohlke web page](https://www.lfd.uci.edu/~gohlke/pythonlibs/): these are 'pycairo' and 'wordcloud' -- choose the packages at the top of each list and download the suitable package for your system (either the 32-bit or the 64-bit package). For instance, look at the name of the file; f.ex. pycairo‑1.18.0‑cp27‑cp27m‑win_amd64.whl means a package pycairo, version 1.18.0 for python 2.7 (cp27) and for the Windows 64 bits operating system; if you have Anaconda installed with python3.4 f.ex., then you must use a package with cp34 in the filename;
  * once downloaded, copy/paste these packages there where your Terminal has been opened -- if your Terminal shows a prompt with: '<base>C:\Users\Downloads', then make sure that your two packages are under 'C:\Users\Downloads'
  * Finally, install these packages with pip in the Terminal, for example, here for a 64-bit windows OS:

```
pip install pycairo‑1.18.0‑cp27‑cp27m‑win_amd64.whl wrapt‑1.11.2‑cp27‑cp27m‑win_amd64.whl
```

## Unixes and BSD -- The command line way

Python has been installed in these operating systems natively, so you first have to install pip -- the Python package manager -- with your OS package manager, along with two dependencies: setuptools and wheel, for example, in Linux for Python 3.x:

```
sudo apt-get install python3-pip && pip3 install wheel setuptools
```

If you don't have pip, setuptools and wheel, you can install them via the get-pip.py, which can be grabbed from [the following page](https://pip.pypa.io/en/stable/installing/). Download this script, open a terminal or command line tool, and type:

```
python get-pip.py
```

Then, you also have to install the python-tk package -- given your OS, you'll find this package in your app-manager (sudo apt-get install python3-tk for Python 3.x; Mac OS users use port). After that, you need to install the required python libraries for MTA:

**With versions 1.7 or above**: Open a terminal or a command line tool of your choice, and copy the following line for Python 3.x users:

```
pip3 install numpy pandas matplotlib \
numexpr scikit-learn scipy gensim seaborn \
community python-louvain bertopic
```


**With versions 1.6 of MTA below (deprecated)**: Open a terminal or a command line tool of your choice, and copy the following line for Python 3.x users:

```
pip3 install numpy pandas matplotlib \
numexpr scikit-learn scipy gensim seaborn \
community python-louvain wordcloud
```
