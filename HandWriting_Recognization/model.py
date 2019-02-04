# -*- coding: utf-8 -*-
"""
Project task: handwriting recognization
module task: build MLP model 
Created on Sun Feb  3 16:29:41 2019
@author: daijun.chen
"""

from data import *
import tensorflow as tf
import numpy as np

# reset the graph #
tf.reset_default_graph()

# build model #
x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

parameters = {'w':tf.Variable(tf.random_normal([784, 10])), 'b':tf.Variable(tf.zeros([10]))}

z = tf.matmul(x, parameters['w']) + parameters['b']

pred = tf.nn.softmax(z)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y), axis=0)

learning_rate = 0.01

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

