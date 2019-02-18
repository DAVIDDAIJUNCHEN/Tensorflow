# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 22:25:23 2019
Project: LM_RNN
module: preprocess functions
@author: daijun.chen
"""

#------------------------ normalized elapsed time ----------------------------#
def elapsed(sec):
    if sec < 60:
        return str(sec) + ' secs'
    elif sec < (60*60):
        return str(sec/60) + ' mins'
    else:
        return str(sec/3600) + 'hours'

#------------------------ get label data from data file ----------------------#
def get_ch_label(file):
    labels = ''

    with open(file, 'rb') as f:
        for label in f:
#             print(u'{0}'.format(label))
            labels = labels + label.decode('utf-8')
    return labels

#------------------------ convert text file to vector list -------------------#
def get_ch_label_vec(file, word_order_map, txt_label=None):
    words_len = len(word_order_map)
    to_order = lambda word: word_order_map.get(word, words_len)
    
    if file != None:
        txt_label = get_ch_label(file)
    
    labels_vector = list(map(to_order, txt_label))
    
    return labels_vector
    
#------------------------ 
