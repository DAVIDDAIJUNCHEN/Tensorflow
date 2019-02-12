# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 16:05:55 2019

@author: daijun.chen
"""

import tensorflow as tf
import numpy as np
import time
from input_data import labels, batch_size, next_batch
from input_data import *
from build_model import *

#-------------------------- define session -----------------------------------#
epochs = 100
save_dir = 'log/asrtest/'
saver = tf.train.Saver(max_to_keep=1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.latest_checkpoint(save_dir)
    print('latest checkpoint:', ckpt)

    start_epo = 0
    if ckpt!=None:
        saver.restore(sess, ckpt)
        ind = ckpt.find('-')
        start_epo = int(ckpt[(ind+1):])
        print(start_epo)
    
    print('Run training epoch')
    
    train_start = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()    
        if epoch < start_epo:
            continue
        print('epoch start: {} '.format(epoch), ' total_epochs={}'.format(epochs))

        n_batches_per_epoch = int(np.ceil(len(labels) / batch_size))
        
    print('total_loop ', n_batches_per_epoch, 'in one epoch, ', batch_size, ' items in one loop')
    
    train_loss = 0.0
    train_label_error_rate = 0.0
    next_idx = 0
    
    for batch in range(n_batches_per_epoch):
        next_idx, source, source_lengths, sparse_labels = next_batch(labels, next_idx, batch_size)
        feed = {input_tensor:source, targets:sparse_labels, seq_length:source_lengths,
                                                            keep_dropout:keep_dropout_rate}
        batch_loss, _ = sess.run([avg_loss, optimizer], feed_dict=feed)
        train_loss += batch_loss
    
        if (batch+1)%20 == 0:
            print('loop: ', batch, 'train cost:', train_loss/(batch+1))
            feed2 = {input_tensor: source, targets: sparse_labels,seq_length: source_lengths,keep_dropout:1.0}

            d,train_ler = sess.run([decoded[0],label_error_rate], feed_dict=feed2)
            dense_decoded = tf.sparse_tensor_to_dense( d, default_value=-1).eval(session=sess)
            dense_labels = sparse_tuple_to_texts_ch(sparse_labels, words)
            
            counter =0
            print('Label err rate: ', train_ler)
            for orig, decoded_arr in zip(dense_labels, dense_decoded):
            # convert to strings
                decoded_str = ndarray_to_text_ch(decoded_arr,words)
                print(' file {}'.format( counter))
                print('Original: {}'.format(orig))
                print('Decoded:  {}'.format(decoded_str))
                counter=counter+1
                break
                
    epoch_duration = time.time() - epoch_start
    
    log = 'Epoch {}/{}, train_cost: {:.3f}, train_ler: {:.3f}, time: {:.2f} sec'
    print(log.format(epoch ,epochs, train_loss,train_ler,epoch_duration))
    saver.save(sess, save_dir+"asr.cpkt", global_step=epoch)
    
train_duration = time.time() - train_start
print('Training complete, total duration: {:.2f} min'.format(train_duration / 60))

