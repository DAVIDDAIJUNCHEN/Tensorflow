# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 22:56:19 2019
Project task: ASR with BiRNN
module task: get data
@author: daijun.chen
"""

import numpy as np
from collections import Counter 
from auxfunctions import get_wavs_labels
from auxfunctions import pad_sequences
from auxfunctions import sparse_tuple_from
from auxfunctions import get_audio_and_transcriptch
#from auxfunctions import 

#--------------------------- wav and label path ------------------------------#
wav_path='C:/Users/daijun.chen/Desktop/深度学习之TensorFlow配套资源/Tensorflow_Tutor/.spyproject/Chap9/thchs30-standalone/wav/train'
label_file='C:/Users/daijun.chen/Desktop/深度学习之TensorFlow配套资源/Tensorflow_Tutor/.spyproject/Chap9/thchs30-standalone/doc/trans/train.word.txt'

wav_files, labels = get_wavs_labels(wav_path, label_file)

print(wav_files[0], labels[0])
print('wav: ', len(wav_files), 'label: ', len(labels))

#--------------------------- generate batch function -------------------------#
# word counting with Counter function #
all_words = []

for label in labels:
    all_words += [word for word in label]

counter = Counter(all_words) # word dictionary
words = sorted(counter) # vocabulary
words_size = len(words) # 2666 words in vocabulary
word_num_map = dict(zip(words, range(words_size)))

#--------------------------- take a batch of samples -------------------------#
n_input = 26
n_context = 9  # context number tuning parameter 
batch_size = 8 # tuning parameter

def next_batch(labels, start_idx=0, batch_size=1, wav_files=wav_files):
    file_size = len(labels)
    end_idx = min(file_size, start_idx + batch_size)
    idx_list = range(start_idx, end_idx)
    txt_labels = [labels[i] for i in idx_list]
    wav_files = [wav_files[i] for i in idx_list]
    (source, audio_len, target, transcript_len) = get_audio_and_transcriptch(None,
                                                     wav_files, n_input, n_context,
                                                     word_num_map, txt_labels)
    
    start_idx += batch_size 
    # if start_idx > file_size, then start from last one
    if start_idx >= file_size:
        start_idx = -1
    
    # make the sequences have the same length
    source, source_lengths = pad_sequences(source)
    # convert dense matrix to sparse matrix
    sparse_labels = sparse_tuple_from(target)
    
    return start_idx, source, source_lengths, sparse_labels

next_batch(labels, batch_size=2)        
    

