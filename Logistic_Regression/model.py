import tensorflow as tf
import numpy as np

# define structure y = 2x #
X = tf.placeholder('float')
Y = tf.placeholder('float')

w = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.zeros([1]), 'intercept')

Z = tf.multiply(X, w) + b

# define loss function & optimizer#
loss = tf.reduce_mean(tf.square(Y - Z))

learning_rate = 0.01

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


