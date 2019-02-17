# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 21:43:35 2019
Project: RNN LM
module: preprocess data
@author: daijun.chen
"""

import tensorflow as tf
import numpy as np
from preprocess import get_ch_label, get_ch_label_vec
from collections import Counter

tf.reset_default_graph()

#--------------------- input the data ----------------------------------------#
training_file = 'wordstest.txt'
training_data = get_ch_label(training_file)

print('Loading training data, plz wait a moment')
print(training_data)

counter = Counter(training_data) # output a counter dictionary
words = sorted(counter)  # get the ordered words list
words_len = len(words)  # 67 words 
words_order_map = dict(zip(words, range(words_len))) # {'word0': 0, 'word1':1}

print('In training data, we have {} words in vocabulary'.format(words_len))

wordlabel = get_ch_label_vec(training_file, words_order_map) #convert txt to order

