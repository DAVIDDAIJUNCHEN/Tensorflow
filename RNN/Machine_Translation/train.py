# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 21:33:12 2019
Project task: Build a machine translation model
Module task: train the model 
@author: daijun.chen
"""

import os
import time
import math
import sys
import numpy as np
import tensorflow as tf
import tools
import data
import seq2seq_model 

# reset the tensor graph
tf.reset_default_graph()

# set hyperparameters 
steps_per_checkpoint = 200

max_train_data_size = 0

dropout = 0.9
grad_clip = 5.0
batch_size = 60

num_layers = 2
learning_rate = 0.5
lr_decay_factor = 0.99

# translation steps  
hidden_size = 100
checkpoint_dir = './checkpoints/'

_buckets = [(20, 20), (40, 40), (60, 60)]

vocab_en_file = os.path.join(data.data_dir, data.en_voc_filename)
vocab_ch_file = os.path.join(data.data_dir, data.ch_voc_filename)

idfile_en = data.data_dir + 'fromids/'
idfile_ch = data.data_dir + 'toids/'


#---------------- get en & ch vocabulary and ids file info -------------------#
def translate_info():
    # print out en and ch vocabulary info
    vocab_en, reverse_vocab_en = tools.initialize_vocabulary(vocab_en_file)
    vocab_ch, reverse_vocab_ch = tools.initialize_vocabulary(vocab_ch_file)
    
    size_vocab_en = len(vocab_en)
    size_vocab_ch = len(vocab_ch)
    
    print('Size of English vocabulary: {}'.format(size_vocab_en))
    print('Size of Chinese vocabulary: {}'.format(size_vocab_ch))
    
    # get the source and target ids train file path    
    filesfrom, _ = tools.getRawFileList(idfile_en)
    filesto, _ = tools.getRawFileList(idfile_ch)
    source_trainfile_path = filesfrom[0]
    target_tranfile_path = filesto[0]

    # will add source and target ids test file path
    return size_vocab_en, size_vocab_ch, reverse_vocab_en, reverse_vocab_ch, source_trainfile_path, target_tranfile_path
    
# 
def train_main():
    size_vocab_en, size_vocab_ch, reverse_vocab_en, reverse_vocab_ch, source_trainfile_path, target_trainfile_path = translate_info()
    
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    
    print('The relative ckpt dir is %s .' % checkpoint_dir)
    
    # start session to train model 
    with tf.Session() as sess:
        model = create_Model(sess, False, size_vocab_en, size_vocab_ch)
        print('Using bucket sizes: ', _buckets)
        
        # source&target: test data = train data 
        source_testfile_path = source_trainfile_path
        target_testfile_path = target_trainfile_path
   
        print('source train file path: ', source_trainfile_path)
        print('source test file path: ', source_testfile_path)
        print('target train file path: ', target_trainfile_path)
        print('target test file path: ', target_testfile_path)        
        
        # buckets source and target data set #
        train_set = readData(source_trainfile_path, target_trainfile_path, max_train_data_size)
        test_set = readData(source_testfile_path, target_testfile_path, max_train_data_size)
        
        train_bucket_sizes = [len(train_set[b]) for b in range(len(_buckets))]
        test_bucket_sizes = [len(test_set[b]) for b in range(len(_buckets))]
        
        total_size_train = float(sum(train_bucket_sizes))
        total_size_test = float(sum(test_bucket_sizes))
        
        print('Bucket sizes of train data set: ', train_bucket_sizes)
        print('Bucket sizes of test data set: ', test_bucket_sizes)
        # accumulated ratio 
        train_buckets_scale = [sum(train_bucket_sizes[:(i+1)])/total_size_train for i in range(len(train_bucket_sizes))]
        test_buckets_scale = [sum(test_bucket_sizes[:(i+1)])/total_size_test for i in range(len(test_bucket_sizes))]
        
        step_time, loss = (.0, .0)
        current_step = 0
        prev_losses = []
        
        while True:
            # find the smallest id st. percentile > rand #
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in range(len(train_buckets_scale)) if train_buckets_scale[i] > random_number_01])
            
        # start training #
        start_time = time.time()
        encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id)
        _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, False)
        step_time += (time.time() - start_time) / steps_per_checkpoint
        loss += step_loss / steps_per_checkpoint
        current_step += 1
        
        if current_step % steps_per_checkpoint ==0:
            perplexity = math.exp(loss) if loss < 300 else float('inf')
            print('global step %d learning rate %.4f step-time %.2f perplexity'
                  '%.2f' % (model.global_step.eval(), model.learning_rate.eval(), step_time, perplexity))
            if len(prev_losses) > 2 and loss > max(prev_losses[-3:]):
                sess.run(model.learning_rate_decay_op)
                
            prev_losses.append(loss)
            
            checkpoint_path = os.path.join(checkpoint_dir, 'seq2seqtest.ckpt')
            model.saver.save(sess, checkpoint_path, global_step=model.global_step)
            step_time = .0
            loss = .0
            
            for bucket_id in range(len(_buckets)):
                if len(test_set[bucket_id]) == 0:
                    print(' eval: empty bucket %d' % (bucket_id))
                    continue
                encoder_inputs, decoder_inputs, target_weights = model.get_batch(test_set, bucket_id)
                
                _, eval_loss, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
                eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
                print(' eval: bucket %d perplexity %.2f' % (bucket_id, eval_ppx))
                
                inputstr = tools.ids2texts(reversed([en[0] for en in encoder_inputs]), reverse_vocab_en)
                print('inputs', inputstr)
                print('outputs', tools.ids2texts([en[0] for en in decoder_inputs]), reverse_vocab_ch)
            
                outputs = [np.argmax(logit, axis=1)[0] for logit in output_logits]
                if tools.EOS_ID in outputs:
                    outputs = outputs[:outputs.index(tools.EOS_ID)]
                    print('results ', tools.ids2texts(outputs, reverse_vocab_ch))
                
                sys.stdout.flush()
            
#-------------- create seq2seq model and initialize model in session ---------#
def create_Model(session, forward_only, size_vocab_source, size_vocab_target):
    
    '''Create seq2seq machine translation model and initialize parameters'''
    model = seq2seq_model.Seq2SeqModel(
            size_vocab_source,
            size_vocab_target,
            _buckets,
            hidden_size,
            num_layers,
            dropout,
            grad_clip,
            batch_size,
            learning_rate,
            lr_decay_factor,
            forward_only,
            dtype=tf.float32
            )

    print('model is finished !')
    
    ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    
    if ckpt!=None:
        model.saver.restore(session, ckpt)
        print('Resotring model from {0}'.format(ckpt))
    else:
        print('Building model from scratch.')
        session.run(tf.global_variables_initializer())
    
    return model 

#-----------------  
def readData(source_path, target_path, max_size=None):
    data_set = [[] for _ in _buckets]
    
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        with tf.gfile.GFile(target_path, mode="r") as target_file:
            source = source_file.readline()
            target = target_file.readline()
            counter = 0
            
            while source and target and (not max_size or counter < max_size):
                counter += 1
                if counter % 100000 == 0:
                    print("  reading data line %d" % counter)
                    sys.stdout.flush()
                source_ids = [int(s) for s in source.split()]
                target_ids = [int(t) for t in target.split()]
                target_ids.append(tools.EOS_ID)
                # from small bucket to large bucket, judge which bucket is belongs to     
                for bucket_id, (source_size, target_size) in enumerate(_buckets):
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids])
                        break
                # read the next line
                source, target = source_file.readline(), target_file.readline()

    return data_set

# main of train function #
train_main()


