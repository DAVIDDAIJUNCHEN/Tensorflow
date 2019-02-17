# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 15:42:17 2019
Project task: build RNN LM
Module task: build model  
@author: daijun.chen
"""

import tensorflow as tf
from data import words_len

#------------------------------- Build RNN_LM model --------------------------#
# define hyper parameters #
learning_rate = 0.001
training_iters = 10000
display_step = 1000
n_input = 4  # input seq length 

n_hidden1 = 256
n_hidden2 = 512
n_hidden3 = 512

# define placeholder #
x = tf.placeholder('float32', [None, n_input, 1]) # 4 consective words 
wordy = tf.placeholder('float32', [None, words_len]) # one-hot representation

# define network structure #
x1 = tf.reshape(x, [-1, n_input]) # (, 4)
x2 = tf.split(x1, n_input, 1) # [(:,1), (:,1), (:,1), (:,1)]

LSTMCell1 = tf.contrib.rnn.LSTMCell(n_hidden1)
LSTMCell2 = tf.contrib.rnn.LSTMCell(n_hidden2)
LSTMCell3 = tf.contrib.rnn.LSTMCell(n_hidden3)

rnn_cell = tf.contrib.rnn.MultiRNNCell([LSTMCell1, LSTMCell2, LSTMCell3]) # define cell

outputs, states = tf.contrib.rnn.static_rnn(cell=rnn_cell, inputs=x2, dtype=tf.float32)
pred = tf.contrib.layers.fully_connected(outputs[1], words_len, activation_fn=None)

# define optimizer #
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=wordy))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).miniminze(loss)

# evaluate model #
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(wordy, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

