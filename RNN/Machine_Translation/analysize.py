# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 10:56:48 2019
Project task: build machine translation system
Module task: plot lengths list of ids_file
@author: daijun.chen
"""

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from tensorflow.python.platform import gfile
from tools import *
from data import *

# analysize #
filesfrom, _ = getRawFileList(data_dir+'fromids/')
filesto, _ = getRawFileList(data_dir+'toids/')
    
ids_from_file = filesfrom[0]
ids_to_file = filesto[0]
    
analysize_idsfiles(ids_from_file, ids_to_file, plot_histograms=True, plot_scatter=True)

#---------------- Analysize the source and target ids files ------------------#
def analysize_idsfiles(source_ids_file, target_ids_file, plot_histograms=True, plot_scatter=True):
    source_lengths = []
    target_lengths = []
    
    with gfile.GFile(source_ids_file, 'r') as s_file:
        with gfile.GFile(target_ids_file, 'r') as t_file:
            for s_line in s_file:
                source_num_ids = [len(s_line.split())]
                source_lengths.extend(source_num_ids)
            
            for t_line in t_file:
                target_num_ids = [len(t_line.split())]
                target_lengths.extend(target_num_ids)
            
    if plot_histograms:
        plot_hist_lengths('target lengths', 'green', target_lengths)
        plot_hist_lengths('source_lengths', 'blue', source_lengths)

    if plot_scatter:
        plot_scatter_lengths('source lengths v.s. target lengths', 'source', 
                             'target', source_lengths, target_lengths)

#---------------- plot the histgram of lengths list --------------------------#
def plot_hist_lengths(title, color, lengths_list):
    mean = np.mean(lengths_list)
    std = np.std(lengths_list)
    x = lengths_list
    
    n, bins, patches = plt.hist(x, 50, density=True, facecolor=color, alpha=0.5)
    y = mlab.normpdf(bins, mean, std)
    
    plt.plot(bins, y, 'r--')
    plt.title(title)
    plt.xlabel('Length')
    plt.ylabel('Number of sequences')
    plt.xlim(0, max(lengths_list))
    plt.show()

#------------ plot the scatter distribution: target v.s. source --------------#    
def plot_scatter_lengths(title, x_title, y_title, x_lengths_list, y_lenghts_list):
    plt.scatter(x_lengths_list, y_lenghts_list)
    diag_line = list(range(0, max(x_lengths_list)))
    plt.plot(diag_line, diag_line, 'r--') # add a diagonal line: y=x 
    plt.title(title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.xlim(0, max(x_lengths_list))
    plt.ylim(0, max(y_lenghts_list))
    plt.show()
    
    
