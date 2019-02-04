# -*- coding: utf-8 -*-
"""
Project task: handwriting recognization
module task: train MLP model 
Created on Sun Feb  3 16:29:41 2019
@author: daijun.chen
"""

import tensorflow as tf
import numpy as np
from data import *
from model import *

training_epoches = 100
display_epoch = 5
batch_size = 100

saver = tf.train.Saver()
save_dir = 'log/model.ckpt'

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(training_epoches):
        num_batches = int(mnist.train.num_examples/batch_size)
        
        for batch in range(num_batches):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            feed = {x:batch_x, y:batch_y}
            sess.run(optimizer, feed_dict=feed)
            loss1 = sess.run(loss, feed_dict=feed)
            avg_loss = loss1/num_batches
            
        if epoch % display_epoch==0:
            print('Epoch: {}'.format(epoch+1), 'loss={:9f}'.format(avg_loss))    
                
    print('Training is finished !')
    
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))        
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    print('Accuracy:', accuracy.eval({x: mnist.test.images, y:mnist.test.labels}))
    
    saver.save(sess, save_path=save_dir)    
    print('The model is saved in %s', save_dir)
    
    
    
    
