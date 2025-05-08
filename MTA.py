#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import gc
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
warnings.filterwarnings("ignore", category=FutureWarning, module='sklearn')
warnings.filterwarnings("ignore", category=DeprecationWarning,module='gensim.models')
import io
import codecs
import errno
import glob
import os
from os.path import expanduser
import re, string
import locale
import math
import pandas
import numpy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib import rc, rcParams
matplotlib.use('Agg')
matplotlib.rc('figure', max_open_warning = 0)
import scipy.spatial as sp, scipy.cluster.hierarchy as hc
from scipy.sparse import csr_matrix
import logging
import heapq
import datetime
import seaborn as sns
import community
import importlib
from itertools import cycle
logging.getLogger('matplotlib.font_manager').disabled = True
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
pandas.options.mode.chained_assignment = None  # default='warn'
importlib.reload(sys) # pour garder le code en utf-8
#
# Save all outputs in same dir
#
home = expanduser("~")
save_home = r'MTA-Results_' + datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
os.makedirs(save_home)
#
#
# =======================
#     PRINT SUMMARY
# =======================
#
# Print a summary of screen output and input of user
#
summary = os.path.join(save_home, 'Summary_')

te = open(summary + datetime.datetime.now().strftime("_%d_%m_%Y_%H_%M_%S") + '.log',"w") # File where you need to keep the logs

class Unbuffered:

   def __init__(self, stream):

       self.stream = stream

   def write(self, data):

       self.stream.write(data)
       self.stream.flush()
       te.write(data)

   def flush(self):
       pass

sys.stdout=Unbuffered(sys.stdout)

# =======================
#     INTRODUCTION
# =======================
#
# Print title and infos
#
print("""

        \t\tMTA -- Multi-Text Analyser

Version: 2.0

Author: Christian Papilloud

Email: christian.papilloud at soziologie.uni-halle.de

Documentation: https://github.com/cp1972/mta-app

""")
#
#
#
print("""
        PLEASE READ THE FOLLOWING MESSAGE

        You have to provide text (*.txt) files and a stopwords file. If there
        is something you are missing right now, abort MTA (ctrl+C) and re-run
        it wit these needed basic elements.

        Your results will be saved in a MTA-Results directory in your home or
        user account or the directory where you run MTA from.
        """)
#
## Save the Shard Corpus Index into MTA folder
#
output_prefix = os.path.join(save_home, 'Index-Corpus')
w2vec = os.path.join(save_home, 'w2vec.model')
#
## Save plots and files into MTA folder
#
part_topics_docs_nmf = os.path.join(save_home, 'Part_Topics_Docs_NMF_')
mult_part_topics_docs_nmf = os.path.join(save_home, 'Weights_Topics_Docs_NMF_')
part_topics_docs_lda = os.path.join(save_home, 'Part_Topics_Docs_LDA_')
mult_part_topics_docs_lda = os.path.join(save_home, 'Weights_Topics_Docs_LDA_')
dom_topics_nmf = os.path.join(save_home, 'Dominant_Topics_NMF_')
dom_topics_lda = os.path.join(save_home, 'Dominant_Topics_LDA_')
corr_texts_nmf = os.path.join(save_home, 'Relations_Between_Texts_NMF_')
corr_texts_lda = os.path.join(save_home, 'Relations_Between_Texts_LDA_')
similar_texts_nmf = os.path.join(save_home, 'Similar_Texts_NMF_')
similar_texts_lda = os.path.join(save_home, 'Similar_Texts_LDA_')
similar_topics_nmflda = os.path.join(save_home, 'Similar_Topics_NMF-LDA_')
#
#
nmf_df = os.path.join(save_home, 'Weights_Words_NMF_Topics')
#mat_nmf_df = os.path.join(save_home, 'Weights_Words_NMF_Topics_Network')
lda_df = os.path.join(save_home, 'Weights_Words_LDA_Topics')
#mat_lda_df = os.path.join(save_home, 'Weights_Words_LDA_Topics_Network')
nmf_topic_words = os.path.join(save_home, 'Top_Words_NMF_Topics')
lda_topic_words = os.path.join(save_home, 'Top_Words_LDA_Topics')
#
rm_texts = os.path.join(save_home, 'Rolling_Mean_Texts_NMF_')
rm_time = os.path.join(save_home, 'Rolling_Mean_Time_NMF_')
rm_texts_lda = os.path.join(save_home, 'Rolling_Mean_Texts_LDA_')
rm_time_lda = os.path.join(save_home, 'Rolling_Mean_Time_LDA_')
#
sim_words = os.path.join(save_home, 'Similar_Words_')
sim_clust_words = os.path.join(save_home, 'Similar_Clusters_Words_')
clusters_w2vec = os.path.join(save_home, 'W2vec_Clusters_')
clusters_words = os.path.join(save_home, 'Words_Assoc_Clusters_')
clusters_metrics = os.path.join(save_home, 'Clusters_Metrics_')
cluster_words = os.path.join(save_home, 'Cluster_Similar_Words_')
#
sentnmf = os.path.join(save_home, 'Best_Sentences_NMF_')
sentlda = os.path.join(save_home, 'Best_Sentences_LDA_')
#
bestfileswords = os.path.join(save_home, 'BestFiles_ChoosenWords_')
#
#
# Load corpus
#
while True:
    in_files = input("""
Give the absolute path to a directory with your files: \n
        """)
    print("You entered: " + in_files)
    print("\nShowing three first files of all your files: \n")
    some_files = glob.glob(in_files)
    print(some_files[0:3])
    del some_files
    gc.collect()
    correct = input("\nDoes it look correct? (yes/no): ")
    print("You entered: " + correct)
    if correct == "no" or correct == "n":
        print("\nNot your files? Try again")
        continue
    elif correct == "yes" or correct == "y":
        print("\nTaking those files")
    break
#
# Keep only filenames for labeling matrices -- generator to iterate over files
# and make list of stopwords
#
k = (os.path.basename(x) for x in sorted(glob.glob(in_files)))
corp_labels = list(k)
#
# Load stopwords
#
while True:
    p = input("\nGive absolute path to a file with your stopwords: \n")
    print("You entered: " + p)
    if not os.path.isfile(p):
        print("\nFile not found. Try again")
        continue
    break
stops = [str(x.strip()) for x in open(p,'r',encoding="utf-8")]
#
# Drop words with few characters
#
word_length_c = input("\nMinimal number of characters you want to have in a word between 2 and 9: ")
print("You entered: " + word_length_c)
#
# Define a progress bar for some tasks
def progress_bar(iteration, total, prefix='', suffix='', length=30, fill='█'):
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
#
# Label matrices -- function
#
def labeling(df):
    """
    Function to label matrices in pandas
    """
    dft = df.T
    dft.columns = [corp_labels]
    df_base = dft.T
    return df_base
#
## Better removing of stopwords and junk from corpus
#
def remove_stopwords(corpus):
    """
    Function to remove stopwords from given list of stopwords
    """
    output_array=[]
    for sentence in corpus:
        temp_list=[]
        for word in sentence.lower().split():
            if word.lower().strip('"') not in stops:
                temp_list.append(word)
        output_array.append(' '.join(temp_list))
    return output_array
#
def remove_digits(lst):
    """
    Function to remove digits in list of strings
    """
    pattern = '[0-9]'
    lst = [re.sub(pattern, '', i) for i in lst]
    return lst
#
def remove_space(lst_space):
    """
    Function to remove extra spaces around strings
    """
    pattern_d = ' +'
    lst_space = [re.sub(pattern_d, ' ', i) for i in lst_space]
    return lst_space

def remove_dots(lst_dots):
    """
    Function to remove extra dots
    """
    pattern_e = r'\.\.+'
    lst_dots = [re.sub(pattern_e, ' ', i) for i in lst_dots]
    return lst_dots

def remove_url(lst_url):
    """
    Function to remove urls
    """
    pattern_f = r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))'''
    lst_url = [re.sub(pattern_f, '',i) for i in lst_url]
    return lst_url
#
# Reading the corpus into list with generator function -- List of strings
#
def generate_corpus(files):
    return [open(file, encoding="utf8").read() for file in sorted(glob.glob(files))]

corp_gen = generate_corpus(in_files)
corp_woa = remove_stopwords(corp_gen)
corp_woc = remove_digits(corp_woa)
corp_wod = remove_url(corp_woc)
corp_woe = remove_dots(corp_wod)
corp_join = [' '.join(word for word in sent.split() if not (word.startswith('www') or word.startswith('http')) and word not in stops) for sent in corp_woe]
corp_wo = remove_space(corp_join)

# Deleting unneeded corpus
del corp_woa
del corp_woc
del corp_wod
del corp_woe
del corp_join
gc.collect()
#
# Make the main corpus corpus_wo -- remove punctuation without removing blank spaces
corpus_woa = [re.sub(r'\W+',' ', i) for i in corp_wo]
corpus_wo = [" ".join([word for word in sentences.split(" ") if len(word)>=int(word_length_c)]) for sentences in corpus_woa]
# Need complete corpus as real list later
corpus_re = list(re.sub("[^a-zA-Z'.,;:!?-]+",' ', i) for i in corp_gen)
del corpus_woa
del corp_gen
gc.collect()
#
print("\n PROCESSING YOUR TEXTS \n")
#
# Make a list of lists of sentences from corpus for word2vec, split, use generator for tokenizer
#
def sent_tokenize(corp):
    sentences = re.split('[.!?]', corp)
    sentences = (sent.strip(' ') for sent in sentences)
    return [_f for _f in sentences if _f]

tokenized_data_r = [sent_tokenize(text) for text in corpus_wo]

# Need real list for word2vec

tokenized_data = list(str(liist).split(' ') for liist in tokenized_data_r)

print("\n Done ")

print("\n TUNING PARAMETERS ")

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

print("""
It might be a good idea to discard words rarely or too frequently appearing in your corpus.
Hereafter, you can choose to do that and tune your analysis, or you can go with
the default application. For your information: MTA defaults to an analysis with
words appearing at least in 2 documents, but which do not appear in more than
95% of the documents.
""")

defaultornot = input("\nDo you want to tune your analysis? (yes/no): ")
print("You entered: " + defaultornot)

if defaultornot == 'no' or defaultornot == 'n':
    vectorize = TfidfVectorizer(min_df=2, max_df=0.95, encoding='utf-8', analyzer='word', ngram_range=(1,1), stop_words=stops)
    lda_vectorize = CountVectorizer(min_df=2, max_df=0.95, encoding='utf-8', analyzer='word', ngram_range=(1,1), stop_words=stops)
elif defaultornot == 'yes' or defaultornot == 'y':

    # Words occuring once in the corpus in percent

    min_df = input("""\n
    Give a value in percent (f.ex. 10% must be written 0.1) for very
    rare words that you want to discard from the analysis.

    For example 0.01 is for words appearing in less than 1% of your documents.
    MTA will not take up words appearing in less than 1% of
    your documents: """)
    print("\nYou entered: " + min_df)

    max_df = input("""\n
    Give a value in percent (f.ex. 92% must be written 0.92) for very frequent words
    that you want to discard from the analysis.

    For example 0.92 is for words appearing in more than 92% of your documents.
    MTA will ignore words that appear in mostly all (exactly
    more than 92% of) your documents: """)
    print("\nYou entered: " + max_df)


    vectorize = TfidfVectorizer(min_df=float(min_df), max_df=float(max_df), encoding='utf-8', analyzer='word', ngram_range=(1,1), stop_words=stops)
    lda_vectorize = CountVectorizer(min_df=float(min_df), max_df=float(max_df), encoding='utf-8', analyzer='word', ngram_range=(1,1), stop_words=stops)

## Create matrices from corpus with label for NMF topic modelling
#
try:
    tf_matrix = vectorize.fit_transform(corpus_wo)
except ValueError: # raised if not enough texts
    print("""
          We have a value error because you have not enough texts.
          """)
    pass
tf_names = vectorize.get_feature_names_out()
dense = tf_matrix.todense()
dense_a = numpy.asarray(dense) # new with new numpy
vocab = numpy.array(vectorize.get_feature_names_out())
df_one = pandas.DataFrame(dense, columns=tf_names)
df_median = labeling(df_one)
# Delete unneeded matrices

del df_one
gc.collect()
#
# Create matrix for LDA topic modelling
#
lda_matrix = lda_vectorize.fit_transform(corpus_wo)
lda_names = lda_vectorize.get_feature_names_out()
#
#
## Choose Language for the plots and fonts
#
plt_user = input("""\n
We want to output nice plots in your language. Please select:

- de for german
- fr for french
- en for english :\n\n """)
print("You entered: " + plt_user)

plt_font = input("""\n
We want to output nice fonts for your plots -- they must be installed on your
computer. Please select:

- a for arial
- b for baskerville
- c for caslon
- g for garamond
- h for helvetica (a kind of arial)
- l for liberation serif (a kind of times new roman)
- m for courier new
- t for times new roman:\n\n """)
print("You entered: " + plt_font)
#
#
# Parameters for the size of the font in plots
#
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
#
# Functions for MTA
#
# Parameters for the fonts of the regular matplotlib plots
def font_plt(plt_user_f):
    if plt_user_f == "a":
        plt.rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
    elif plt_user_f == "b":
        plt.rc('font',**{'family':'Libre Baskerville'})
    elif plt_user_f == "c":
        plt.rc('font',**{'family':'Big Caslon'})
    elif plt_user_f == "g":
        plt.rc('font',**{'family':'Adobe Garamond Pro'})
    elif plt_user_f == "h":
        plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    elif plt_user_f == "l":
        plt.rc('font',**{'family':'Liberation Serif'})
    elif plt_user_f == "m":
        plt.rc('font',**{'family':'monospace','monospace':['Courier New']})
    elif plt_user_f == "t":
        plt.rc('font',**{'family':'serif','serif':['Times New Roman']})
#
# Function to search occurence of terms in list of words related to topics
#
def findItem(theList, item):
    return [(ind, theList[ind].index(item)) for ind in range(len(theList)) if item in theList[ind]]
#
# Function metrics -- calculates a given number of clusters and output the
# optimal number of clusters for best model
## Kmeans++ for evaluation of best topics and metrics for validation
## Metrics: Silhouette, Davies Bouldin score, BIC and Log-Likelihood
#
from scipy.cluster.hierarchy import cophenet, dendrogram, linkage
from scipy.spatial.distance import cosine, pdist
from sklearn import decomposition
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.metrics import silhouette_score,davies_bouldin_score,v_measure_score,calinski_harabasz_score
from sklearn.decomposition import LatentDirichletAllocation

# Function for metrics and cophenet correlation

def km_metrics_all(km_sse,km_sc,km_ch,km_db,coph_corr,coph_corr_lda):
   for i in range(2,num_c):
      km = KMeans(n_clusters=i, random_state=0,init="k-means++").fit(X_scaled)
      km_sse[i] = km.inertia_
      km_sc[i] = silhouette_score(X_scaled, km.fit_predict(X_scaled))
      km_ch[i] = calinski_harabasz_score(X_scaled, km.labels_)
      km_db[i] = davies_bouldin_score(X_scaled,km.labels_)
      progress_bar(i, num_c-1, prefix='Progress:', suffix='Complete', length=50)
   # Delete unneeded objects
   del km

   for i in range(2,num_c):
      nmf = decomposition.NMF(n_components=i, random_state=1)
      doctopic = nmf.fit_transform(tf_matrix)
      linkage_doc = linkage(doctopic, 'ward')
      coph_corr.append(cophenet(linkage_doc, pdist(doctopic)))

   for i in range(2,num_c):
      lda_sktl = LatentDirichletAllocation(n_components=i, evaluate_every=-1, learning_method='online', n_jobs=-1, learning_offset=50., random_state=100, batch_size=128)
      doctopic_lda = lda_sktl.fit_transform(lda_matrix)
      linkage_doc_lda = linkage(doctopic_lda, 'ward')
      coph_corr_lda.append(cophenet(linkage_doc_lda, pdist(doctopic_lda)))

   return km_sse, km_sc, km_ch, km_db, coph_corr, coph_corr_lda

# Function to get turning points in metrics -- linear variant because those arrays are small in general

def turnp(lst):
   lst_n = []
   for i in range(1, len(lst)-1):
      if lst[i-1] > lst[i] < lst[i+1] or lst[i-1] < lst[i] > lst[i+1]:
         lst_n.append(i)
   return lst_n

# Function to return the results of turning points

def check_lst(lst):
   if lst == None or len(lst) == 0:
      print("No optimal number of topics found")
   else:
      print("Optimal number of topics, from better to worst:", *lst, sep=" ")
   return lst

# Function to model the topics with NMF

def nmf_tm(numb_top):
    num_top_words = len(df_median.columns)
    print("\nTopic-Model with NMF\n")
    nmf = decomposition.NMF(n_components=numb_top, random_state=1)
    doctopic = nmf.fit_transform(tf_matrix)
    topicwords = nmf.components_
    linkage_doc = linkage(doctopic, 'ward')
    coph_corr = cophenet(linkage_doc, pdist(doctopic))
    return num_top_words, nmf, doctopic, topicwords, linkage_doc, coph_corr

# Function to model the topics with LDA

def lda_tm(numb_tlda):
    print("\n Topic-Model with LDA\n")
    lda_sktl = LatentDirichletAllocation(n_components=numb_tlda, evaluate_every=-1, learning_method='online', n_jobs=-1, learning_offset=50., random_state=100, batch_size=128)
    lda_sktl.fit_transform(lda_matrix)
    doctopic_lda = lda_sktl.fit_transform(lda_matrix)
    topicwords_lda = lda_sktl.components_
    linkage_doc_lda = linkage(doctopic_lda, 'ward')
    coph_corr_lda = cophenet(linkage_doc_lda, pdist(doctopic_lda))
    return lda_sktl, doctopic_lda, topicwords_lda, linkage_doc_lda, coph_corr_lda

# Function for the 20 best words depicting topics with NMF

def top_w(tw_list):
    for topic in nmf.components_:
        word_idx = numpy.argsort(topic)[::-1][0:num_top_words]
        tw_list.append([tf_names[i] for i in word_idx])
    topic_words_df = pandas.DataFrame(tw_list)
    topic_words_df_T = topic_words_df.T.head(20).add_prefix('Topic')
    print("\n20 most important words per topics\n")
    print(topic_words_df_T)
    file = open(nmf_topic_words, "a")
    topic_words_df_T.to_csv(nmf_topic_words + datetime.datetime.now().strftime("_%d_%m_%Y_%H_%M_%S") + '.csv', index=False)
    file.close()
    return tw_list,topic_words_df_T

# Function for the 20 best words depicting topics with LDA

def top_wlda(tw_list_lda):
    for topiclda in lda_sktl.components_:
        word_idx = numpy.argsort(topiclda)[::-1][0:num_top_words]
        tw_list_lda.append([vocab[i] for i in word_idx])
    topic_words_lda_df = pandas.DataFrame(tw_list_lda)
    topic_words_lda_df_T = topic_words_lda_df.T.head(20).add_prefix('Topic_')
    print("\n20 most important words per topics\n")
    print(topic_words_lda_df_T)
    file = open(lda_topic_words, "a")
    topic_words_lda_df_T.to_csv(lda_topic_words + datetime.datetime.now().strftime("_%d_%m_%Y_%H_%M_%S") + '.csv', index=False)
    file.close()
    return tw_list_lda,topic_words_lda_df_T

# Function for the data to be shown as wordclouds and to print weights of words in topics
#
def wordcl(top_lst_a,top_lst_b):
   for i in topicwords:
      for j in i[:50]:
         top_lst_a.append(j)

   topic_n = []
   for i in topic_words:
      for j in i[:50]:
         top_lst_b.append(j)

   return top_lst_a,top_lst_b

# Function to plot in the language of the user -- Weight of topics

def lang_weight(plt_userlang):

    if plt_userlang == "fr":
        plt.xlabel("Textes")
        plt.ylabel('Poids des Topics')
    elif plt_userlang == "de":
        plt.xlabel("Texte")
        plt.ylabel('Gewicht der Topics')
    elif plt_userlang == "en":
        plt.xlabel("Texts")
        plt.ylabel('Weight of Topics')

# Function to plot in the language of the user -- Distr of topics

def lang_distr(plt_userlang):

    if plt_userlang == "fr":
        plt.xlabel("Textes")
        plt.ylabel('Distribution des Topics dans les textes en %')
    elif plt_userlang == "de":
        plt.xlabel("Texte")
        plt.ylabel('Verteilung der Topics in den Texten in %')
    elif plt_userlang == "en":
        plt.xlabel("Texts")
        plt.ylabel('Distribution of Topics over texts in %')

# Function to plot in the language of the user -- Corr. between topics

def lang_corr(plt_userlang):

    if plt_userlang == "fr":
        plt.xlabel("Textes")
        plt.ylabel('Textes')
    elif plt_userlang == "de":
        plt.xlabel("Texte")
        plt.ylabel('Texte')
    elif plt_userlang == "en":
        plt.xlabel("Texts")
        plt.ylabel('Texts')

# Function to plot in the language of the user -- Comp. between NMF/LDA topics

def lang_comp(plt_userlang):

    if plt_userlang == "fr":
        plt.xlabel("Topics")
        plt.ylabel('Topics')
    elif plt_userlang == "de":
        plt.xlabel("Topics")
        plt.ylabel('Topics')
    elif plt_userlang == "en":
        plt.xlabel("Topics")
        plt.ylabel('Topics')

# Function to plot in the language of the user -- Rolling mean

def lang_dev(plt_userlang):

    if plt_userlang == "fr":
        plt.xlabel("Textes")
        plt.ylabel('Développement des Topics')
    elif plt_userlang == "de":
        plt.xlabel("Texte")
        plt.ylabel('Entwicklung der Topics')
    elif plt_userlang == "en":
        plt.xlabel("Texts")
        plt.ylabel('Development of Topics')

# Function to compare NMF and LDA topics in both models

def comp_nmflda_plot(number_topics):

    if number_topics == num_tof:
        print("""
        As you have taken the same number of topics for NMF and LDA
        algorithms, we compare their semantic similarities and plot them into
        your MTA folder
        """)
        topcomp=[]
        for i in range(len(topic_words[:20])):
            for j in range(len(topic_words_lda[:20])):
                topcomp.append(len(set(topic_words[i][:20]) & set(topic_words_lda[j][:20]))/float(20))
        topcomp_array = numpy.array(topcomp)
        shape = (num_topics,num_tof)
        comptop_final = topcomp_array.reshape(shape)
        df_nmflda = pandas.DataFrame(comptop_final)
        plt.clf()

        mask = numpy.zeros_like(df_nmflda)
        mask[numpy.triu_indices_from(mask)] = True

        with sns.axes_style("white"):
            ax = sns.heatmap(df_nmflda, mask=mask, fmt='.4f', cmap="PiYG", vmin=-1, vmax=1, annot=True, xticklabels=True, yticklabels=True)
            lang_comp(plt_user)
            font_plt(plt_font)
            plt.tick_params(labelsize=8)
            plt.yticks(rotation=0)
            plt.xticks(rotation=90)
            plt.xlabel("LDA-Topics")
            plt.ylabel('NMF-Topics')
            plt.savefig(similar_topics_nmflda + str(num_tof) + datetime.datetime.now().strftime("_%d_%m_%Y_%H_%M_%S") + '.pdf', bbox_inches='tight')
    elif number_topics != num_tof:
        print("""
        You have taken different number for NMF and LDA topic models,
        that's why we can not compare them
        """)
#
#
# =======================
#     MENUS
# =======================

# Clear terminal for best Menu rendering
#
#print(chr(27) + "[2J")
print(chr(27) + "[H" + chr(27) + "[J")
def main_menu():
    print(("-"*30,  "MENU",  "-"*30))
    print("\nTopic analysis\n")
    print("""
            \t1. NMF and LDA topics in your corpus with crossvalidation
        """)
    print("\nFurther analysis after running 1. above\n")
    print("""
            \t2. Evolution of NMF and LDA topics through texts
            \t3. Weight of a given word in NMF and LDA topics per text
        """)
    print("\nContext analysis with Word2Vec\n")
    print("""
            \t4. Similarities between given words and between clusters around these words
            \t0. Exit the program

            \tChoose a menu entry...
        """)
    print(("-"*30,  "MENU",  "-"*30))
#
## Loop to exit to the Menu
#
loop = True
while loop:
    main_menu()
    key_in = int(input("Menu entry: "))
    print("\n"*3)

# =======================
#  MENU NMF-LDA TOPICS
# =======================

    if key_in == 1:

        bnt_f = input("\nDo you want an automatic estimation of the best number of topics? (yes/no): ")
        print("\nYou entered: \n" + bnt_f)
        if bnt_f == "yes" or bnt_f == "y":
            num_c_1 =  int(input("\nMax. number of topics you think you would have in your text: "))
            print("You entered: \n" + str(num_c_1))
            print("\nWe calculate the best number of topics for NMF and LDA. It can take time...\n")
            # Increment 1 to num_c_1 because of indexes of dict beginning with 0
            num_c = num_c_1 + 1
            #
            ### Metrics: Elbow, Silhouette, Calinski Harabasz, Davies Bouldin score -- Save a plot with the results
            ks = list(range(2, num_c))
            X_scaled = dense_a.T
            km_elbow = {}
            km_silhouette = {}
            km_calinski = {}
            km_bouldin= {}
            coph_corr= []
            coph_corr_lda= []

            list(km_metrics_all(km_elbow,km_silhouette,km_calinski,km_bouldin,coph_corr,coph_corr_lda))
            print("\n")

            # Save dict to list, perform turning points and output them to list

            km_elbow_n = list(km_elbow.values())
            km_silhouette_n = list(km_silhouette.values())
            km_calinski_n = list(km_calinski.values())
            km_bouldin_n = list(km_bouldin.values())

            km_el = turnp(km_elbow_n)
            km_el_n = [i+2 for i in km_el]
            print("Elbow scores")
            check_lst(km_el_n)
            km_sil = turnp(km_silhouette_n)
            km_sil_n = [i+2 for i in km_sil]
            print("\nSilhouette scores")
            check_lst(km_sil_n)
            km_cal = turnp(km_calinski_n)
            km_cal_n = [i+2 for i in km_cal]
            print("\nCalinski Harabasz scores")
            check_lst(km_cal_n)
            km_bou = turnp(km_bouldin_n)
            km_bou_n = [i+2 for i in km_bou]
            print("\nDavies Bouldin scores")
            check_lst(km_bou_n)

            # Same for nmf and lda Cophenet correlations

            coph_corr_l = [i[0] for i in coph_corr]
            cophnmf_tp = turnp(coph_corr_l)
            cophnmf_tp_n = [i+2 for i in cophnmf_tp]
            print("\nNMF-Cophenet scores")
            check_lst(cophnmf_tp_n)
            val_cophnmf = []
            for i in cophnmf_tp:
               val_cophnmf.append(coph_corr_l[i])
            print("NMF-Cophenet correlation values for scores: ", (*val_cophnmf,), sep=" ")

            coph_corr_lda_n = [i[0] for i in coph_corr_lda]
            cophlda_tp = turnp(coph_corr_lda_n)
            cophlda_tp_n = [i+2 for i in cophlda_tp]
            print("\nLDA-Cophenet scores")
            check_lst(cophlda_tp_n)
            val_cophlda = []
            for i in cophlda_tp:
               val_cophlda.append(coph_corr_lda_n[i])
            print("LDA-Cophenet correlation values for scores: ", (*val_cophlda,), sep=" ")

            coph_nmf = dict(zip(ks,coph_corr_l))
            coph_lda = dict(zip(ks,coph_corr_lda_n))

            print("\nWe save a plot 'Cluster_Metrics' with all performed test in your MTA folder")

            # Do the plot for all metrics (kmeans++, nmf and lda Cophenet)
            plt.clf()
            font_plt(plt_font)
            fig, axs = plt.subplots(2,3, sharex=True)
            axs[0,0].scatter(x=list(km_elbow.keys()),y=list(km_elbow.values()),s=15,edgecolor='#b58900',alpha=0.5)
            axs[0,0].set_title('Elbow')
            axs[0,1].scatter(x=list(km_silhouette.keys()),y=list(km_silhouette.values()),s=15,edgecolor='#cb4b16',alpha=0.5)
            axs[0,1].set_title('Silhouette')
            axs[0,2].scatter(x=list(coph_nmf.keys()),y=list(coph_nmf.values()),s=15,edgecolor='#cb4b16',alpha=0.5)
            axs[0,2].set_title('Cophenet NMF')
            axs[1,0].scatter(x=list(km_calinski.keys()),y=list(km_calinski.values()),s=15,edgecolor='#dc322f', alpha=0.5)
            axs[1,0].set_title('Calinski Harabasz')
            axs[1,1].scatter(x=list(km_bouldin.keys()),y=list(km_bouldin.values()),s=15,edgecolor='#d33682',alpha=0.5)
            axs[1,1].set_title('Davies Bouldin')
            axs[1,2].scatter(x=list(coph_lda.keys()),y=list(coph_lda.values()),s=15,edgecolor='#cb4b16',alpha=0.5)
            axs[1,2].set_title('Cophenet LDA')
            fig.tight_layout()
            font_plt(plt_font)
            for ax in axs.flat:
                 ax.set(xlabel='Clusters', ylabel='Metrics')
            for ax in axs.flat:
                 ax.label_outer()
            plt.xticks(range(2, num_c))
            plt.locator_params(nbins=10, axis='x')
            if num_c > 20:
                 plt.locator_params(nbins=10, axis='x')
            plt.yticks(fontsize=8)
            plt.savefig(clusters_metrics + datetime.datetime.now().strftime("_%d_%m_%Y_%H_%M_%S") + '.pdf', bbox_inches='tight')

           # Delete unneeded lists/dicts
            del km_elbow
            del km_silhouette
            del km_calinski
            del km_bouldin
            del km_elbow_n
            del km_silhouette_n
            del km_calinski_n
            del km_bouldin_n
            del km_el
            del km_sil
            del km_cal
            del km_bou
            del km_el_n
            del km_sil_n
            del km_cal_n
            del km_bou_n
            del coph_corr
            del cophlda_tp
            del cophlda_tp_n
            del cophnmf_tp
            del cophnmf_tp_n
            del coph_corr_lda
            del coph_corr_l
            del coph_corr_lda_n
            del coph_nmf
            del coph_lda
            del val_cophnmf
            del val_cophlda
            gc.collect()

        elif bnt_f == "no" or bnt_f == "n":
           print("\nNo automatic estimation -- we continue\n")

        num_topics = int(input("\nNumber of topics you want to compute: "))
        print("You entered: " + str(num_topics))

        # NMF topic modelling
        print("\nNMF Topic model\n")
        num_top_words, nmf, doctopic, topicwords, linkage_doc, coph_corr = nmf_tm(num_topics)
        print(("\nCorrelation between topics and texts: %r" % coph_corr[0]))
        print("\nChange the number of topics to improve this correlation\n")

        # 20 best words depicting topics and write them to file

        topic_words = []
        topic_words_df_T = top_w(topic_words)

        # Scale the DTM with sum of components values to one, set labels

        doctopic = doctopic / numpy.sum(doctopic, axis=1, keepdims=True)
        text_labels = numpy.asarray(corp_labels)
        doctopic_orig = doctopic.copy()
        num_groups = len(set(text_labels))
        doctopic_grouped = numpy.zeros((num_groups, num_topics))

        for i, name in enumerate(sorted(set(text_labels))):
            doctopic_grouped[i, :] = numpy.mean(doctopic[text_labels == name, :], axis=0)

        doctopic = doctopic_grouped

        # Write best sentences to file for each topic

        corpus_split = [sent for line in corpus_re for sent in re.split('[?.!...]', line)]

        sent_b=[]
        for t, s in [(t,s) for t in range(len(topic_words)) for s in corpus_split]:
            sum_s = sum(1 for kword in topic_words[t][:50] if kword in s)
            if sum_s > 2:
                sent_b.extend((t,sum_s,s))

        it = iter(sent_b)
        res_lst=list(zip(it,it,it))
        res=sorted(res_lst, key=lambda x: (-x[1], x[0]))
        res_df_endlines = pandas.DataFrame(res, columns =['Topic', 'Frequency of sentence', 'Sentence'])
        res_df = res_df_endlines.replace(r'\n',' ', regex=True)

        file = open(sentnmf, "a")
        res_df.to_csv(sentnmf + datetime.datetime.now().strftime("_%d_%m_%Y_%H_%M_%S") + '.csv', index=False)
        file.close()

        # Delete unneeded lists
        del sent_b
        del res_lst
        del res_df_endlines
        del res_df
        gc.collect()

        # Formating csv file with Word, Topics, Values and invert normalization
        # -- NFM normalize with Froebenius Norm wich makes 0 values be best

        topic_w = []
        topic_n = []
        wordcl(topic_w,topic_n)
        topic_tuple = list(zip(topic_n, topic_w))

        topic_df = pandas.DataFrame(topic_tuple, columns=['Words','Values'])
        topic_df['Norm'] = (topic_df.Values.max() - topic_df.Values) / (topic_df.Values.max() - topic_df.Values.min())
        del topic_df['Values']
        topic_df.rename({'Norm' : 'Values'}, axis=1, inplace=True)
        topic_df_copy = topic_df.copy()

        df_listindex = topic_df_copy.index.values.tolist()
        topic_names = [df_listindex[i:i + 50] for i in range(0,len(df_listindex),50)]
        append_str = 'Topic_'
        index_tn = [i for i in range(len(topic_names))]
        index_tn_a = numpy.repeat(index_tn,50)
        topname = map(str, index_tn_a)
        topicnames = [append_str + sub for sub in topname]

        topic_df_copy['Topics'] = pandas.Series(topicnames, dtype="object")
        topic_words_nmf_df = topic_df_copy.pivot_table(index='Words',columns='Topics', values='Values', aggfunc=numpy.mean)

        file = open(nmf_df, "a")
        topic_df_copy.to_csv(nmf_df + '.csv', sep='\t', encoding='utf-8')
        file.close()

        #file = open(mat_nmf_df, "a")
        #topic_words_nmf_df.to_csv(mat_nmf_df + '.csv', sep=',', encoding='utf-8')
        #file.close()

        # Delete unneeded objects
        del topic_df_copy
        del df_listindex
        del topic_names
        del index_tn
        del index_tn_a
        del topname
        del topic_words_nmf_df
        gc.collect()

        print("""\n
        We save the following files for NMF results:

        - part of topics in each text
        - dominant topics for each text
        - correlations between texts based on topics distribution
        - 20 best words per topic
        - Best sentences per topic (topic number, freq of words in sentence,
          sentence)

        You will find these files in your MTA-Results folder.
        """)

        # Distribution of topics over texts for NMF topics

        df_topic = pandas.DataFrame(doctopic)
        df_topics = labeling(df_topic)
        df_topics_s = df_topics.sort_index(ascending=True)

        plt.clf()
        font_plt(plt_font)


        if len(df_topics_s.index) > 70:
            df_topics.columns = ['Topic_' + str(col) for col in df_topics.columns]
            df_topics['Jahr'] = df_topics.index
            df_topics['Jahr'] = df_topics['Jahr'].map(lambda x:str(x)[2:6])
            df_topics.groupby(['Jahr']).mean().plot(kind='bar', stacked=True, colormap='Paired')
            lang_weight(plt_user)
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop={'size':10})
            plt.tick_params(
                            axis='y',       # changes apply to the y-axis
                            which='both',      # both major and minor ticks are affected
                            bottom=False,      # ticks along the bottom edge are off
                            top=False,         # ticks along the top edge are off
                            labelbottom=False, right='off', left='off', labelleft='off')
            plt.xticks(rotation=90)
            plt.locator_params(nbins=10, axis='x')
            plt.savefig(mult_part_topics_docs_nmf + str(num_topics) + datetime.datetime.now().strftime("_%d_%m_%Y_%H_%M_%S") + '.pdf',  bbox_inches='tight')
        elif len(df_topics_s.index) <= 70:
            df_topics_s.columns = ['Topic_' + str(col) for col in df_topics_s.columns]
            df_topics_s.plot(kind='bar', stacked=True, colormap='Paired')
            lang_distr(plt_user)
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop={'size':10})
            plt.savefig(part_topics_docs_nmf + str(num_topics) + datetime.datetime.now().strftime("_%d_%m_%Y_%H_%M_%S") + '.pdf',  bbox_inches='tight')

        # Delete unneeded objects
        del df_topic
        del df_topics
        gc.collect()
        # Dominant topic per text

        dominant_topic = numpy.argmax(df_topics_s.values, axis=1)
        dominant_t = pandas.Series(dominant_topic)
        df_topics_s['Dominant_Topic_NMF'] = dominant_t.values
        df_domt_nmf = df_topics_s.sort_values('Dominant_Topic_NMF')

        file = open(dom_topics_nmf, "a")
        df_domt_nmf.to_csv(dom_topics_nmf + str(num_topics) + datetime.datetime.now().strftime("_%d_%m_%Y_%H_%M_%S") + '.csv')
        file.close()

        # Delete unneeded objects
        del dominant_topic
        del dominant_t
        del df_domt_nmf
        gc.collect()

        # Correlations between texts based on topics distribution

        df_topic = pandas.DataFrame(doctopic)
        df_topics = labeling(df_topic)
        df_topics_s = df_topics.sort_index(ascending=True)
        df_topics_t = df_topics_s.T
        dfcorr = df_topics_t.corr()
        dfcorr_l = dfcorr[(dfcorr > 0.7) & (dfcorr < 1.0)]

        file = open(corr_texts_nmf, "a")
        dfcorr_l.to_csv(corr_texts_nmf + str(num_topics) + datetime.datetime.now().strftime("_%d_%m_%Y_%H_%M_%S") + '.csv')
        file.close()

        # Delete unneeded objects
        del df_topic
        del df_topics_s
        del df_topics_t
        del dfcorr
        gc.collect()

        if len(dfcorr_l.index) < 70:
            plt.clf()

            mask = numpy.zeros_like(dfcorr_l)
            mask[numpy.triu_indices_from(mask)] = True

            with sns.axes_style("white"):
                ax = sns.heatmap(dfcorr_l, mask=mask, fmt='.2f',cmap="PiYG", vmin=-1, vmax=1, xticklabels=True, yticklabels=True)
                lang_corr(plt_user)
                font_plt(plt_font)
                plt.tick_params(labelsize=8)
                plt.yticks(rotation=0)
                if len(dfcorr_l.index) > 50:
                    plt.tick_params(
                            axis='both',       # changes apply to the x-axis and the y-axis
                            which='both',      # both major and minor ticks are affected
                            bottom=False,      # ticks along the bottom edge are off
                            top=False,         # ticks along the top edge are off
                            labelbottom=False, right='off', left='off', labelleft='off')
                elif len(dfcorr_l.index) < 50:
                    plt.tick_params(labelsize=8)
                    plt.xticks(rotation=90)
                plt.savefig(similar_texts_nmf + str(num_topics) + datetime.datetime.now().strftime("_%d_%m_%Y_%H_%M_%S") + '.pdf', bbox_inches='tight')
        elif len(dfcorr_l.index) >= 80:
            print("""
            We don't print a plot of the correlations between texts
            based on topics since you have more than 80 documents. Please use
            the csv file instead
            """)

        ### Scikit-Learn LDA

        # Choose final number of topics for LDA

        print(("\nYou calculated this number of topics with the NMF algorithm: %d\n" % num_topics))
        top_lda = input("\nDo you want to calculate a topic model with the LDA algorithm (yes/no): ")
        print("You entered: " + str(top_lda))
        if top_lda == "yes" or top_lda == "y":
            from sklearn.decomposition import LatentDirichletAllocation
            num_tof =  int(input("\nGive a number of topics for LDA (same as NMF f.ex., or another one): "))
            print("You entered: " + str(num_tof))
            lda_sktl, doctopic_lda, topicwords_lda, linkage_doc_lda, coph_corr_lda = lda_tm(num_tof)
            print(("\nCorrelation between topics and texts: %r" % coph_corr_lda[0]))
            print("\nChange the number of topics to improve this correlation\n")

            # 20 best words per topic and save to file

            topic_words_lda = []
            top_wlda(topic_words_lda)

            # Print best sentences for LDA

            sent_lda=[]
            for t, s in [(t,s) for t in range(len(topic_words_lda)) for s in corpus_split]:
                sum_lda = sum(1 for kword in topic_words_lda[t][:50] if kword in s)
                if sum_lda > 2:
                    sent_lda.extend((t,sum_lda,s))

            itl = iter(sent_lda)
            res_lda=list(zip(itl,itl,itl))
            reslda=sorted(res_lda, key=lambda x: (-x[1], x[0]))
            reslda_df_endlines = pandas.DataFrame(reslda, columns =['Topic', 'Frequency of sentence', 'Sentence'])
            reslda_df = reslda_df_endlines.replace(r'\n',' ', regex=True)

            file = open(sentlda, "a")
            reslda_df.to_csv(sentlda + datetime.datetime.now().strftime("_%d_%m_%Y_%H_%M_%S") + '.csv', index=False)
            file.close()

            # Delete unneeded objects
            del sent_lda
            del itl
            del res_lda
            del reslda
            del reslda_df_endlines
            del reslda_df
            gc.collect()

            # Save a pandas DF with weights of words in topics

            topic_w_lda = []
            topic_n_lda = []
            wordcl(topic_w_lda,topic_n_lda)

            topic_tuple_lda = list(zip(topic_n_lda, topic_w_lda))
            topic_df_lda = pandas.DataFrame(topic_tuple_lda, columns=['Words','Values'])
            topic_df_lda_copy = topic_df_lda.copy()
            topic_df_lda_copy['Topics'] = pandas.Series(topicnames, dtype="object")

            topic_words_lda_df = topic_df_lda_copy.pivot_table(index='Words', columns='Topics', values='Values', aggfunc=numpy.mean)

            file = open(lda_df, "a")
            topic_df_lda_copy.to_csv(lda_df + '.csv', sep='\t', encoding='utf-8')
            file.close()

            #file = open(mat_lda_df, "a")
            #topic_words_lda_df.to_csv(mat_lda_df + '.csv', sep=',', encoding='utf-8')
            #file.close()

            # Delete unneeded objects
            del topic_df_lda_copy
            del topicnames
            gc.collect()

            print("""\n
            Now, we save the following files for LDA results:

            - part of topics in each text
            - dominant topics for each text
            - correlations between texts based on topics distribution
            - 20 best words per topic
            - Best sentences per topic (topic number, freq of words in sentence,
              sentence)

            You will find these files in your MTA-Results folder.
            """)

            # Distribution of topics over texts for LDA topics

            df_topic_lda = pandas.DataFrame(doctopic_lda)
            df_topics_lda = labeling(df_topic_lda)
            df_topics_s_lda = df_topics_lda.sort_index(ascending=True)

            plt.clf()
            font_plt(plt_font)

            if len(df_topics_s_lda.index) > 70:
                df_topics_lda.columns = ['Topic_' + str(col) for col in df_topics_lda.columns]
                df_topics_lda['Jahr'] = df_topics_lda.index
                df_topics_lda['Jahr'] = df_topics_lda['Jahr'].map(lambda x:str(x)[2:6])
                df_topics_lda.groupby(['Jahr']).mean().plot(kind='bar', stacked=True, colormap='Paired')
                lang_weight(plt_user)
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop={'size':10})
                plt.tick_params(
                                axis='y',       # changes apply to the x-axis and the y-axis
                                which='both',      # both major and minor ticks are affected
                                bottom=False,      # ticks along the bottom edge are off
                                top=False,         # ticks along the top edge are off
                                labelbottom=False, right='off', left='off', labelleft='off')
                plt.xticks(rotation=90)
                plt.locator_params(nbins=10, axis='x')
                plt.savefig(mult_part_topics_docs_lda + str(num_tof) + datetime.datetime.now().strftime("_%d_%m_%Y_%H_%M_%S") + '.pdf', bbox_inches='tight')
            elif len(df_topics_s_lda.index) <= 70:
                df_topics_s_lda.columns = ['Topic_' + str(col) for col in df_topics_s_lda.columns]
                df_topics_s_lda.plot(kind='bar', stacked=True, colormap='Paired')
                lang_weight(plt_user)
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop={'size':10})
                plt.savefig(part_topics_docs_lda + str(num_tof) + datetime.datetime.now().strftime("_%d_%m_%Y_%H_%M_%S") + '.pdf',  bbox_inches='tight')


            # Scale the DTM with sum of components values to one, set labels

            doctopic_lda = doctopic_lda / numpy.sum(doctopic_lda, axis=1, keepdims=True)
            text_labels = numpy.asarray(corp_labels)
            doctopic_orig_lda = doctopic_lda.copy()
            num_groups_lda = len(set(text_labels))
            doctopic_grouped_lda = numpy.zeros((num_groups_lda, num_tof))

            for i, name in enumerate(sorted(set(text_labels))):
                doctopic_grouped_lda[i, :] = numpy.mean(doctopic_lda[text_labels == name, :], axis=0)

            doctopic_lda = doctopic_grouped_lda

            # Dominant topic

            dominant_topic_lda = numpy.argmax(df_topics_s_lda.values, axis=1)
            dominant_t_lda = pandas.Series(dominant_topic_lda)
            df_topics_s_lda['Dominant_Topic_LDA'] = dominant_t_lda.values
            df_domt_lda = df_topics_s_lda.sort_values('Dominant_Topic_LDA')

            file = open(dom_topics_lda, "a")
            df_domt_lda.to_csv(dom_topics_lda + str(num_tof) + datetime.datetime.now().strftime("_%d_%m_%Y_%H_%M_%S") + '.csv')
            file.close()

            # Delete unneeded objects
            del df_topic_lda
            del df_topics_lda
            del df_topics_s_lda
            gc.collect()

            # Correlations between texts based on topics distribution

            df_topic_lda = pandas.DataFrame(doctopic_lda)
            df_topics_lda = labeling(df_topic_lda)
            df_topics_s_lda = df_topics_lda.sort_index(ascending=True)
            df_topics_t_lda = df_topics_s_lda.T
            dfcorr_lda = df_topics_t_lda.corr()

            file = open(corr_texts_lda, "a")
            dfcorr_lda.to_csv(corr_texts_lda + str(num_tof) + datetime.datetime.now().strftime("_%d_%m_%Y_%H_%M_%S") + '.csv')
            file.close()

            # Delete unneeded objects
            del df_topics
            del df_topics_lda
            del df_topics_s_lda
            del df_topics_t_lda
            gc.collect()

            if len(dfcorr_lda.index) < 70:
                plt.clf()
                font_plt(plt_font)

                mask = numpy.zeros_like(dfcorr_lda)
                mask[numpy.triu_indices_from(mask)] = True

                with sns.axes_style("white"):
                    ax = sns.heatmap(dfcorr_lda, mask=mask, fmt='.2f',cmap="PiYG", vmin=-1, vmax=1, xticklabels=True, yticklabels=True)
                    lang_corr(plt_user)
                    plt.tick_params(labelsize=8)
                    plt.yticks(rotation=0)
                    if len(dfcorr_lda.index) > 50:
                        plt.tick_params(
                                axis='both',       # changes apply to the x-axis and the y-axis
                                which='both',      # both major and minor ticks are affected
                                bottom=False,      # ticks along the bottom edge are off
                                top=False,         # ticks along the top edge are off
                                labelbottom=False, right='off', left='off', labelleft='off')
                    elif len(dfcorr_lda.index) < 50:
                        plt.tick_params(labelsize=8)
                        plt.xticks(rotation=90)
                    plt.savefig(similar_texts_lda + str(num_tof) + datetime.datetime.now().strftime("_%d_%m_%Y_%H_%M_%S") + '.pdf', bbox_inches='tight')
            elif len(dfcorr_lda.index) >= 80:

                print("""
                We don't print a plot of the correlations between texts
                based on topics since you have more than 80 documents. Please use
                the csv file instead
                """)

            # Delete corr matrice
            del dfcorr_lda
            gc.collect()

            # Comparison between NMF and LDA topics by same number of topic for
            # each model
            #
            comp_nmflda_plot(num_topics)

# =======================
#  MENU EVOL. TOPICS
# =======================

    elif key_in == 2:

        print("\nEvolution of NMF topics through texts (rolling mean)\n")
        window_size_nmf = int(input("\nGive a window size -- f.ex. 2 = 2 texts used for calculating the statistics: \n"))
        print("You entered: " + str(window_size_nmf))
        df_topic = pandas.DataFrame(doctopic)
        df_topics = labeling(df_topic)
        df_topics_s = df_topics.sort_index(ascending=True)
        df_topics_s.columns = ['Topic_' + str(col) for col in df_topics_s.columns]

        # Calculate rolling mean

        for col in df_topics_s.columns:
            df_topics_s['RM_'+ str(col)] = df_topics_s[str(col)].rolling(window=window_size_nmf,center=False).mean()

        # Plot rolling mean for all topics in all texts

        filter_col = [col for col in list(df_topics_s) if col.startswith('RM_')]
        df_topics_rm = df_topics_s[filter_col]
        plt.clf()
        font_plt(plt_font)

        # Delete unneeded objects
        del df_topic
        del df_topics
        del df_topics_s
        del filter_col
        gc.collect()

        if len(df_topics_rm.index) > 40:
             print("""
                We don't print a plot of the weight of topics for each text
                since you have more than 50 documents. The plot
                would be unreadable.
                """)
        if len(df_topics_rm.index) <= 40:
            df_topics_rm.plot(kind='bar', stacked=True, colormap='Paired')
            lang_distr(plt_user)
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop={'size':10})
            plt.tick_params(
                                axis='y',       # changes apply to the x-axis and the y-axis
                                which='both',      # both major and minor ticks are affected
                                bottom=False,      # ticks along the bottom edge are off
                                top=False,         # ticks along the top edge are off
                                labelbottom=False, right='off', left='off', labelleft='off')
            plt.xticks(rotation=90)
            plt.savefig(rm_texts + str(num_topics) + datetime.datetime.now().strftime("_%d_%m_%Y_%H_%M_%S") + '.pdf', bbox_inches='tight')


        # Rolling mean of Topics by year if data grouped by year

        rm_nmf = input("""
        If the name of your files starts with a year stamp (YYYY-whatever), you can see results grouped by year (yes/no):
        """)
        print("You entered: " + rm_nmf)
        plt.clf()
        font_plt(plt_font)
        if rm_nmf == "yes" or rm_nmf == "y":
            df_topics_rm.index = df_topics_rm.index.map(lambda x:str(x)[2:6])
            df_topics_rm.groupby(df_topics_rm.index).mean().plot(colormap='Paired')
            lang_dev(plt_user)
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop={'size':10})
            plt.tick_params(
                            axis='y',       # changes apply to the y-axis
                            which='both',      # both major and minor ticks are affected
                            bottom=False,      # ticks along the bottom edge are off
                            top=False,         # ticks along the top edge are off
                            labelbottom=False, right='off', left='off', labelleft='off')
            plt.xticks(rotation=90)
            plt.savefig(rm_time + str(num_topics) + datetime.datetime.now().strftime("_%d_%m_%Y_%H_%M_%S") + '.pdf', bbox_inches='tight')

        elif rm_nmf == "no" or rm_nmf == "n":
            print("\nBack to menu\n")

        # Delete unneeded df_rm matrice
        del df_topics_rm
        gc.collect()

        try:
            df_topic_lda
        except NameError:
            print("You don't have performed a LDA analysis")
        else:
            print("\nEvolution of LDA topics through texts (rolling mean)\n")
            window_size_lda = int(input("\nGive a window size -- f.ex. 2 = 2 texts used for calculating the statistics: \n"))
            print("You entered: " + str(window_size_lda))
            df_topic_lda = pandas.DataFrame(doctopic_lda)
            df_topics_lda = labeling(df_topic_lda)
            df_topics_s_lda = df_topics_lda.sort_index(ascending=True)


            df_topics_s_lda.columns = ['Topic_' + str(col_lda) for col_lda in df_topics_s_lda.columns]

            # Calculate rolling mean

            for col_lda in df_topics_s_lda.columns:
                df_topics_s_lda['RM_'+ str(col_lda)] = df_topics_s_lda[str(col_lda)].rolling(window=window_size_lda,center=False).mean()

            # Rolling mean of all Topics in all texts

            filter_col_lda = [col_lda for col_lda in list(df_topics_s_lda) if col_lda.startswith('RM_')]
            df_topics_rm_lda = df_topics_s_lda[filter_col_lda]

            # Delete unneeded objects
            del df_topic_lda
            del df_topics_lda
            del df_topics_s_lda
            del filter_col_lda
            gc.collect()

            plt.clf()
            font_plt(plt_font)
            if len(df_topics_rm_lda.index) > 40:
                print("""
                We don't print a plot of the weight of topics for each text since you have more than 50 documents.
                The plot would be unreadable.
                """)

            if len(df_topics_rm_lda.index) <= 40:
                df_topics_rm_lda.plot(kind='bar', stacked=True, colormap='Paired')
                lang_distr(plt_user)
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop={'size':10})
                plt.savefig(rm_texts_lda + str(num_tof) + datetime.datetime.now().strftime("_%d_%m_%Y_%H_%M_%S") + '.pdf', bbox_inches='tight')

            # Rolling mean of Topics by year if data grouped by year

            rm_lda = input("""
            If the name of your files starts with a year stamp (YYYY-whatever), you can see results grouped by year (yes/no):
            """)
            print("You entered: " + rm_lda)
            plt.clf()
            font_plt(plt_font)
            if rm_lda == "yes" or rm_lda == "y":
                df_topics_rm_lda.index = df_topics_rm_lda.index.map(lambda x:str(x)[2:6])
                df_topics_rm_lda.groupby(df_topics_rm_lda.index).mean().plot(colormap='Paired')
                lang_dev(plt_user)
                plt.tick_params(axis='y',       # changes apply to the x-axis and the y-axis
                                which='both',      # both major and minor ticks are affected
                                bottom=False,      # ticks along the bottom edge are off
                                top=False,         # ticks along the top edge are off
                                labelbottom=False, right='off', left='off', labelleft='off')
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop={'size':10})
                plt.xticks(rotation=90)
                plt.savefig(rm_time_lda + str(num_tof) + datetime.datetime.now().strftime("_%d_%m_%Y_%H_%M_%S") + '.pdf', bbox_inches='tight')

            elif rm_lda == "no" or rm_lda == "n":
                print("\nBack to menu\n")

            # Delete unneeded df rm
            del df_topics_rm_lda
            gc.collect()


# =======================
#  MENU WORD TO TOPICS
# =======================

    elif key_in == 3:

        print("\nWe return the weight of a given word to all NMF topics")
        string_u_nmf = input("\nEnter a word: ")
        print("You entered: " + string_u_nmf)

        index_t = findItem(topic_words, string_u_nmf)
        index_t_list = [int(i[1]) for i in index_t]
        word_topics = []
        for i in index_t_list:
            for j in range(len(topicwords)):
                word_topics.append(topicwords[j][i])
        word_topics_f = word_topics[0::len(index_t_list)+1]

        print("\nValues")

        print(('\n'.join('Topic {}: {}'.format(*k) for k in enumerate(word_topics_f))))

        # Delete unneeded objects
        del index_t
        del index_t_list
        del word_topics
        del word_topics_f
        gc.collect()

        try:
            topic_words_lda
        except NameError:
            print("You don't have performed a LDA analysis")
        else:
            print("\nWe return the weight of a given word to all LDA topics")
            string_u_lda = input("\nEnter a word: ")
            print("You entered: " + string_u_lda)

            index_t_lda = findItem(topic_words_lda, string_u_lda)
            index_t_list_lda = [int(i[1]) for i in index_t_lda]
            word_topics_lda = []
            for i in index_t_list_lda:
                for j in range(len(topicwords_lda)):
                    word_topics_lda.append(topicwords_lda[j][i])
            word_topics_f_lda = word_topics_lda[0::len(index_t_list_lda)+1]

            print("\nValues")

            print(('\n'.join('Topic {}: {}'.format(*k) for k in enumerate(word_topics_f_lda))))

            # Delete unneeded objects
            del index_t_lda
            del index_t_list_lda
            del word_topics_lda
            del word_topics_f_lda
            gc.collect()


# =======================
#   MENU SIMIL WORDS
# =======================

    elif key_in == 4:

        try:
            user_sw = input("""\n
            Input one or more words lower case space separated (e.g. cat dog car etc.): \n
            """)
            user_sw_lst = user_sw.split()
            print("You entered: " + user_sw)

            # Set a value for the average size of a word vector for word2vec model and round the
            # result -- Lenght of unique terms in documents divided by the number of
            # documents
            #
            size = len(set(str(corpus_wo).split(' ')))/len(corp_labels)
            rounded_s = round(size)
            if rounded_s > 300:
                rounded_s == 300
            elif rounded_s > 10 and rounded_s < 300:
                rounded_s == rounded_s
            elif rounded_s < 10:
                rounded_s == 10
            #
            # Making a word2vec model for words embedding and words associations
            #
            print("\n MAKING A WORD2VEC MODEL WITH YOUR TEXTS FOR SIMILARITIES \n")
            import gensim
            from gensim.models import Word2Vec, CoherenceModel
            from gensim import similarities, corpora
            from gensim.similarities.docsim import Similarity
            from gensim.utils import simple_preprocess
            from gensim.corpora.dictionary import Dictionary
            model_yesno = input("\nLoad an existing w2vec model (yes/no)?: ")
            print("You entered: " + model_yesno)

            if model_yesno == "yes" or model_yesno == "y":
                while True:
                    w2vm = input("\nGive absolute path to your model: \n")
                    print("You entered: " + w2vm)
                    if not os.path.isfile(w2vm):
                        print("\nModel not found. Try again")
                        continue
                    break
                w2vmodel = Word2Vec.load(w2vm)
            elif model_yesno == "no" or model_yesno == "n":
                contextwin = input("\nNumber of words around a word to look for associates -- if normal sentences 3-5, with long sentences around 6-10 (max. 20): ")
                print("You entered: " + contextwin)
                num_features = int(rounded_s)
                context = int(contextwin)
                sample = 1e-05
                w2vmodel = Word2Vec(tokenized_data, min_count=5, vector_size=num_features,workers=6, window=context, sample=sample, epochs=5)
                save_yn = input("\nSave your model for further use?(yes/no): ")
                if save_yn == "yes" or save_yn == "y":
                    w2vmodel.save(w2vec)
                elif save_yn == "no" or save_yn == "n":
                    print("\nWe do not save your model")

            # Check the list to keep only items in the w2vec model
            for item in user_sw_lst[:]:
               if item not in w2vmodel.wv:
                  user_sw_lst.remove(item)

            print("\nBest similar words to your word(s) decreasing in importance from left to right\n")
            similar_words = {given_term: [item[0] for item in w2vmodel.wv.most_similar([given_term]) if item[1] > 0.001] for given_term in user_sw_lst}
            print("{:<15} {:<25}".format('Input word(s)','Similar words'))
            for k, v in similar_words.items():
                words = v
                print("{:<15} {:<25}".format(k, ', '.join(words)))

            # Check best files for input words

            swkeys=list(similar_words.keys())
            swval=list(similar_words.values())
            flt_swval=[x for xs in swval for x in xs]
            smerge=swkeys + flt_swval
            dfmedian_n =df_median[df_median.columns.intersection(smerge)]
            dfmedian_n['mean']=dfmedian_n.mean(axis=1)

            bestfilescwords_yesno = input("\nDo you want to save a list of best files for your choosen words (yes/no)?: ")
            if bestfilescwords_yesno == "yes" or bestfilescwords_yesno =="y":
               print("\nValues with min, max and average importance for all files\n")
               print(dfmedian_n["mean"].describe())
               print("""
               You can input a value around average to select more/less important files corresponding
               to value below average (more files with important and less important ones) or above it
               (less files, i.e. almost only the important files).
               """)
               bestfilescwords_value = input("\nYour value: ")
               bestfiles_cwords_df=dfmedian_n.loc[dfmedian_n["mean"] > float(bestfilescwords_value)]
               print("\nTotal number of files in your corpus: " + str(len(df_median.index)))
               print("\nNumber of selected files based on your words: " + str(len(bestfiles_cwords_df.index)))
               print("""\n
               We save the list of your files to disk.
               You can use it to select those files from your corpus and perform
               an analysis only on those files
               """)
               file = open(bestfileswords, "a")
               bestfiles_cwords_df.to_csv(bestfileswords + datetime.datetime.now().strftime("_%d_%m_%Y_%H_%M_%S") + '.csv', columns=[], header=False)
               file.close()
            elif bestfilescwords_yesno == "no" or bestfilescwords_yesno=="n":
               print("\nNo list of files needed, we continue\n")

            # Visualisation of word embeddings: TSNE and scatter plot for given
            # terms
            #
            import matplotlib.cm as cm
            from sklearn.manifold import TSNE
            def tsnescatterplot(model, word):

                arr = numpy.empty((0,int(rounded_s)), dtype='f')
                word_labels = [word]

                # get close words
                close_words = model.wv.most_similar(word)

                # add the vector for each of the closest words to the array
                arr = numpy.append(arr, numpy.array([model.wv[word]]), axis=0)
                for wrd_score in close_words:
                    wrd_vector = model.wv[wrd_score[0]]
                    word_labels.append(wrd_score[0])
                    arr = numpy.append(arr, numpy.array([wrd_vector]), axis=0)

                # find tsne coords for 2 dimensions
                tsne = TSNE(n_components=2, random_state=0, perplexity=7)
                numpy.set_printoptions(suppress=True)
                Y = tsne.fit_transform(arr)

                x_coords = Y[:, 0]
                y_coords = Y[:, 1]
                # display scatter plot
                plt.clf()
                font_plt(plt_font)

                plt.figure(figsize=(12, 6))
                plt.scatter(x_coords, y_coords, clip_on=False)

                for label, x, y in zip(word_labels, x_coords, y_coords):
                    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')

                plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
                plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
                plt.axis("off")
                plt.savefig(sim_words + datetime.datetime.now().strftime("_%d_%m_%Y_%H_%M_%S") + '.pdf', bbox_inches='tight')

            for i in user_sw_lst:
                tsnescatterplot(w2vmodel, i)

        except KeyError as e:
            print(e)
            continue
        #
        ## Cluster or words around given words
        #
        embedding_clusters = []
        word_clusters = []
        for word in user_sw_lst:
            embeddings = []
            words = []
            for sim_word, _ in w2vmodel.wv.most_similar(word, topn=50):
                words.append(sim_word)
                embeddings.append(w2vmodel.wv[sim_word])
            embedding_clusters.append(embeddings)
            word_clusters.append(words)

        embedding_clusters = numpy.array(embedding_clusters)
        n, m, k = embedding_clusters.shape
        tsne_model_sw = TSNE(perplexity=16, n_components=2, init='pca', n_iter=3500, random_state=32)
        embeddings_sw = numpy.array(tsne_model_sw.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)

        def tsne_plot_similar_words(labels, embedding_clusters, word_clusters, a, filename=None):
            font_plt(plt_font)
            plt.figure(figsize=(16, 9))
            colors = cm.rainbow(numpy.linspace(0, 1, len(labels)))
            for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
                x = embeddings[:, 0]
                y = embeddings[:, 1]
                plt.scatter(x, y, color=color, alpha=a, label=label)
                for i, word in enumerate(words):
                    plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2),textcoords='offset points', ha='right', va='bottom', size=8)
            plt.legend(loc=4)
            plt.axis("off")
            plt.savefig(sim_clust_words + datetime.datetime.now().strftime("_%d_%m_%Y_%H_%M_%S") + '.pdf', bbox_inches='tight')

        tsne_plot_similar_words(user_sw_lst, embeddings_sw, word_clusters, 0.7)


    elif key_in == 0:
        loop=False # This will make the while loop to end as not value of loop is set to False
    else:
        # Any integer inputs other than values 0-10 we print an error message
        input("Wrong option selection. Enter any key to try again.")
