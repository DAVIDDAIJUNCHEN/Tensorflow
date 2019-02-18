# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 15:44:29 2019
Porject task: build RNN LM
module task: train model
@author: daijun.chen
"""

import time
import numpy as np
import tensorflow as tf
import random as random
from preprocess import elapsed
from data import wordlabel, training_data, words_size, words
from model import *

save_dir = 'log/rnn_lm/'
saver = tf.train.Saver(max_to_keep=1)

training_iteration = 1000000  # clairify the total epochs 
display_step = 1000

start_time = time.time()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # global variable initialization
    
    step = 0
    offset = random.randint(0, n_input+1)
    end_offset = n_input + 1
    acc_total = 0
    loss_total = 0
    
    ckpt = tf.train.latest_checkpoint(save_dir) # find the latest checkpoint
    print('ckpt: ', ckpt )
    
    start_epoch = 0
    
    if ckpt != None:
        saver.restore(sess, ckpt)
        ind = ckpt.find('-') # find index of '-'
        start_epoch = start_epoch + int(ckpt[(ind+1):])
        print('start of epoch is {}'.format(start_epoch))
        step = start_epoch
    
    while step < training_iteration:
        if offset > (len(training_data) - end_offset):
            offset = random.randint(0, n_input + 1)  # shorten the offset 
        
        inwords = [[wordlabel[i]] for i in range(offset, offset+n_input)]
        inwords = np.reshape(np.array(inwords), [-1, n_input, 1]) # seq with len n_input
        
        out_one_hot = np.zeros([words_size], dtype=float) # [0,0,0,0,0], len=voc_size
        out_one_hot[wordlabel[offset+n_input]] = 1.0 # one_hot 
        out_one_hot = np.reshape(out_one_hot, [1, -1]) # row vector
        
        feed = {x:inwords, wordy:out_one_hot} # sequence and row one_hot vector
        _, acc, lossval, one_hot_pred = sess.run([optimizer, accuracy, loss, pred],
                                                feed_dict=feed)
        
        loss_total += lossval
        acc_total += acc
        
        if (step+1) % display_step == 0:
            print('Iter = ' + str(step+1) + ', average loss = ' + '{:.6f}'.format(
                    loss_total/display_step) + ', average accuracy = {:.2f}%'.format(
                            100*acc_total/display_step))
            acc_total = 0
            loss_total = 0
            in2 = [words[wordlabel[i]] for i in range(offset, offset + n_input)]
            out2 = words[wordlabel[offset + n_input]]
            out_pred = words[int(tf.arg_max(one_hot_pred, 1).eval())]
            print('%s - [%s] vs [%s]' % (in2, out2, out_pred))
            saver.save(sess, save_dir + 'rnnwordtest.ckpt', global_step = step)
            
        step += 1
        offset += 1
        
    print('Finished !')
 
    saver.save(sess, save_dir + 'rnnwordtest.ckpt', global_step = step)
    
    print('Elapsed time:', elapsed(time.time() - start_time))
        
