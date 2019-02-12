# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 12:08:19 2019
Project task: ASR with BiRNN
module task: Build model
@author: daijun.chen
"""

import tensorflow as tf
from input_data import n_input, n_context

tf.reset_default_graph()

#--------------------------- build placeholder -------------------------------#
MFCC_feature = n_input + 2*n_input*n_context

# [batch_size, max_len, MFCC_features]
input_tensor = tf.placeholder(tf.float32, [None, None, MFCC_feature], name='input')
# text of audio, sparse matrix placeholder
targets = tf.sparse_placeholder(tf.int32, name='targets')
# length of current sequence 
seq_length = tf.placeholder(tf.int32, [None], name='seq_length')
# keep probability
keep_dropout = tf.placeholder(tf.float32)

#------------------------- define variable on cpu ----------------------------#
def variable_on_cpu(name, shape, initializer):

    # /cpu:0 device
    with tf.device('/cpu:0'):
         var = tf.get_variable(name=name, shape=shape, initializer=initializer)

    return var

#--------------------------- build BiRNN model -------------------------------#
# batch_x is the input_tensor #
def BiRNN(batch_x, seq_length, num_input, num_context, num_character, keep_dropout):
    # [batch_size, max_len, MFCC_features] #
    batch_x_shape = tf.shape(batch_x)
    # transfer to time series [max_len, batch_size, MFCC_feature] #
    batch_x = tf.transpose(batch_x, [1, 0, 2])
    # transfer to 2-dimension for inputing #
    MFCC_feature = num_input + 2*num_input*num_context
    batch_x = tf.reshape(batch_x, [-1, MFCC_feature])

    # define model hyper parameters #
    b_stddev = 0.046875
    h_stddev = 0.046875
    
    num_hidden = 1024
    num_hidden_1 = 1024
    num_hidden_2 = 1024
    num_hidden_3 = 2*1024
    num_cell_dim = 2014 # 4th layer cell dimensions
    num_hidden_5 = 1024

    
    keep_dropout_rate = 0.95
    relu_clip = 20    
        
    # 1st layer
    with tf.name_scope('fc1'):
        b1 = variable_on_cpu('b1', shape=[num_hidden_1], tf.random_normal_initializer(stddev=b_stddev))
        h1 = variable_on_cpu('h1', shape=[MFCC_feature, num_hidden_1], tf.random_normal_initializer(stddev=h_stddev))
        z1 = tf.matmul(batch_x, h1) + b1
        layer_1 = tf.minimum(tf.nn.relu(z1), relu_clip)
        layer_1 = tf.nn.dropout(layer_1, keep_dropout)
    
    # 2nd layer
    with tf.name_scope('fc2'):
        b2 = variable_on_cpu('b2', shape=[num_hidden_2], tf.random_normal_initializer(stddev=b_stddev))
        h2 = variable_on_cpu('h2', shape=[num_hidden_1, num_hidden_2], tf.random_normal_initializer(stddev=h_stddev))
        z2 = tf.matmul(layer_1, h2) + b2 
        layer_2 = tf.minimum(tf.nn.relu(z2), relu_clip)
        layer_2 = tf.nn.dropout(layer_2, keep_dropout)
        
    # 3rd layer
    with tf.name_scope('fc3'):
        b3 = variable_on_cpu('b3', shape=[num_hidden_3], tf.random_normal_initializer(stddev_b_stddev))
        h3 = variable_on_cpu('h3', shape=[num_hidden_2, num_hidden_3], tf.random_normal_initializer(stddev=h_stddev))
        z3 = tf.matmul(layer_2, h3) + h3
        layer_3 = tf.minimum(tf.nn.relu(z3), relu_clip)
        layer_3 = tf.nn.dropout(layer_3, keep_dropout)
        
    # BiRNN layer (4th layer)
    with tf.name_scope('lstm'):
        # Forward direction cell
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(num_cell_dim, forget_bias=1.0, state_is_tuple=True)
        lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, input_keep_prob=keep_dropout)
        
        # Backward direction cell 
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(num_cell_dim, forget_bias=1.0, state_is_tuple=True)
        lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, input_keep_prob=keep_dropout)
        
        # layer_3: [n_step, batch_size, 2*num_cell_dim]
        layer_3 = tf.reshape(layer_3, [-1, batch_x_shape[0], num_hidden_3])
        
        outputs, outputs_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, 
                                                                  inputs=layer_3, dtype=tf.float32,
                                                                  time_major=True, seq_length=seq_length)        
        outputs = tf.concat(outputs, 2)
        outputs = tf.reshape(outputs, [-1, 2*num_cell_dim])
        
    # 5th layer
    with tf.name_scope('fc5'):
        num_hidden_4 = 2*num_cell_dim
        b5 = variable_on_cpu('b5', [num_hidden_5], tf.random_noraml_initializer(stddev=b_stddev))
        h5 = variable_on_cpu('h5', [num_hidden_4, num_hidden_5], tf.random_normal_initializer(stddev=h_stddev))
        z5 = tf.matmul(outputs, h5) + b5
        layer_5 = tf.minmum(tf.nn.relu(z5), relu_clip)
        layer_5 = tf.nn.dropout(layer_5, keep_dropout)
        
        
    # 6th layer
    with tf.name_scope('fc6'):
        b6 = variable_on_cpu('b6', [num_character], tf.random_normal_initializer(stddev=b_stddev))
        h6 = variable_on_cpu('h6', [num_hidden_5, num_character], tf.random_normal_initializer(stddev=h_stddev))
        layer_6 = tf.matmul(layer_5, h6) + b6
        
    layer_6 = tf.reshape(layer_6, [-1, batch_x_shape[0], num_character])
    
    return layer_6
        
