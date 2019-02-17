# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 22:25:23 2019
Project: LM_RNN
module: preprocess functions
@author: daijun.chen
"""

#------------------------ get label data from data file ----------------------#
def get_ch_label(file):
    labels = ''

    with open(file, 'rb') as f:
        for label in f:
            labels = labels + label.decode('gb2312')

    return labels

#------------------------ convert text file to vector list -------------------#
def get_ch_label_vec(file, word_order_map):
    words_len = len(word_order_map)
    to_order = lambda word: word_order_map.get(word, words_len)
    
    if file != None:
        file = get_ch_label(file)
    
    labels_vector = list(map(to_order, file))
    
    return labels_vector
    
#------------------------ 



