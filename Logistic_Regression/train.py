import tensorflow as tf
from  model import *
from data import *
import numpy as np


# Train model #
init = tf.global_variables_initializer()

training_epochs = 100

display_step = 5

with tf.Session() as sess:
    sess.run(init) 
    plotdata = {'batchsize:': [], 'loss:': []}

    for epoch in range(training_epochs):
        for (x, y) in zip(Train_X, Train_Y):
            feed = {X:x, Y:y}
            sess.run(optimizer, feed_dict=feed)
            
        if epoch % display_step == 0:
            feed = {X:x, Y:y}
            loss1 = sess.run(loss, feed_dict=feed)
            w1 = sess.run(w)
            b1 = sess.run(b)
            print('Epoch: {},  Loss: {}, w={} and b={}'.format(epoch+1, loss1, w1, b1))
    print('Training finished!')
    print('Loss = ', sess.run(loss, feed_dict={X:Train_X, Y:Train_Y}), 'w=', sess.run(w), 'b=', sess.run(b))




