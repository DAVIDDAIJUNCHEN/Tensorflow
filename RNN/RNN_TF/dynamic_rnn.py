# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 16:47:04 2019
Project task: Familiar with RNN in TF
Module task: dynamic RNN for variable-length sequences
@author: daijun.chen
"""

import tensorflow as tf
import numpy as np

tf.reset_default_graph()

# input data #
X = np.random.randn(2, 4, 5)
X[1,1:] = 0
seq_lengths = [4, 1]

# rnn cell 
lstm = tf.contrib.rnn.BasicLSTMCell(num_units=3, state_is_tuple=True)
gru = tf.contrib.rnn.GRUCell(3)

outputs, last_states = tf.nn.dynamic_rnn(lstm, X, seq_lengths, dtype=tf.float64)
gruoutputs, grulast_states = tf.nn.dynamic_rnn(gru, X, seq_lengths, dtype=tf.float64)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    lstmout, lstmsta, gruout, grusta = sess.run([outputs, last_states, gruoutputs, grulast_states])
    
    print('Whole sequence: \n', lstmout[0])
    print('Short sequence: \n', lstmout[1])
    print('LSTM state: ', len(lstmsta), '\n', lstmsta[1])
    print('GRU short seq: \n', gruout[1])
    print('GRU state: ', len(grusta), '\n', grusta[1])

