# -*- coding: utf-8 -*-
"""
Project task: handwriting recognization
module task: test MLP model 
Created on Sun Feb  3 16:29:41 2019
@author: daijun.chen
"""

import tensorflow as tf
import numpy as np

save = tf.train.Saver()

print('Start restoring model')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    save.restore(sess, './log/model.ckpt')
    
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
    
    output = tf.argmax(pred, 1)
    batch_xs, batch_ys = mnist.train.next_batch(2)
    outputval, predv = sess.run([output, pred], feed_dict={x:batch_xs})
    
    print(outputval, predv, batch_ys)
    
    im = batch_xs[0]
    im = im.reshape(-1, 28)
    pylab.imshow(im)
    pylab.show()
    
    im = batch_xs[1]
    im = im.reshape(-1, 28)
    pylab.imshow(im)
    pylab.show()
    
    
    
    
    
