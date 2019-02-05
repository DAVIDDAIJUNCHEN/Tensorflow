# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 16:53:15 2019
Project task: familiar with softmax function & cross entropy 
@author: daijun.chen
"""

import tensorflow as tf
import numpy as np

#-------------------------- Softmax function ---------------------------------#
labels = [[0, 0, 1], [0, 1, 0]]
label1s = [2, 1]
label2s = [[.4, .1, .5], [.3, .6, .1]]
logits = [[2, 0.5, 6], [0.1, 0, 3]]

logits_scaled = tf.nn.softmax(logits)
logits_scaled2 = tf.nn.softmax(logits_scaled)

result1 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
result2 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits_scaled)
result3 = - tf.reduce_sum(labels*tf.log(logits_scaled), 1)
result4 = - tf.reduce_sum(labels*tf.log(logits_scaled2), 1)
result5 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label1s, logits=logits)
result6 = tf.nn.softmax_cross_entropy_with_logits(labels=label2s, logits=logits)


with tf.Session() as sess:
    print('scaled=', sess.run(logits_scaled))
    print('scaled2=', sess.run(logits_scaled2))
    print('rel1=', sess.run(result1), '\n')    
    print('rel2=', sess.run(result2), '\n')
    print('rel3=', sess.run(result3), '\n')
    print('rel4=', sess.run(result4), '\n')
    print('rel5=', sess.run(result5), '\n')
    print('rel6=', sess.run(result6), '\n')
    
#-------------------------------- cross entropy ------------------------------#
global_step = tf.Variable(0, trainable=False)
initial_learning_rate = 0.1
learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step=global_step
                                           , decay_steps=10, decay_rate=0.9)
opt = tf.train.GradientDescentOptimizer(learning_rate)
add_global = global_step.assign_add(1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    print('0', sess.run(learning_rate))
    
    for i in range(20):
        g, rate = sess.run([add_global, learning_rate])
        
        print(g, rate)
        
        
