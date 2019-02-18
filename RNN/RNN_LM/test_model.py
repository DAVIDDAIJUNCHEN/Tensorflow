# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 11:16:35 2019
Project task: build rnn_lm
Module task: test_model 
@author: daijun.chen
"""

import tensorflow as tf
import numpy as np
from preprocess import get_ch_label_vec
from data import words_order_map, words
from model import *

save_dir = 'log/rnn_lm'
saver = tf.train.Saver(max_to_keep=1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.latest_checkpoint(save_dir)
    print('ckpt: ', ckpt)
    
    if ckpt != None:
        saver.restore(sess, ckpt)
    else:
        print('Train model first !')
    
    while True:
        prompt = 'Plz input %s words: ' % n_input
        sentence = input(prompt)
        input_word = sentence.strip()
        
        if len(input_word) != n_input:
            print('The lenght of the input data is: ', len(input_word), 'plz reinput 4 words')
            continue
        try:
            input_word = get_ch_label_vec(None, words_order_map, input_word)
            print(input_word)        
            for i in range(32):
                keys = np.reshape(np.array(input_word), [-1, n_input, 1])
                onehot_pred = sess.run(pred, feed_dict={x: keys})
                onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
                sentence = '%s%s'%(sentence, words[onehot_pred_index])
                input_word = input_word[1: ]
                input_word.append(onehot_pred_index)
            print(sentence)            
        except:
            print('I did not learn it well')
            break
