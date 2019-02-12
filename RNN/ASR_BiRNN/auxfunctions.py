# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 22:23:32 2019
Project task: ASR with BiRNN 
Module task: tool functions 
@author: daijun.chen
"""

import os
import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc

#----------------------------- Get wav files and labels ----------------------#
'''Get wav files and labels'''
def get_wavs_labels(wav_path, label_file):

    wav_files = []
    for dirpath, dirnames, filenames in os.walk(wav_path):
        for filename in filenames:
            if filename.endswith('.wav') or filename.endswith('.WAV') or filename.endswith('.Wav'):
                filename_path = os.sep.join([dirpath, filename])
                if os.stat(filename_path).st_size < 240000:
                    continue
                wav_files.append(filename_path)
    
    labels_dict = {}
    with open(label_file, 'rb') as f:
        for label in f:
            label = label.strip(b'\n')
            label_id = label.split(b' ', 1)[0]
            label_text = label.split(b' ', 1)[1]
            labels_dict[label_id.decode('ascii')] = label_text.decode('utf-8')

    labels = []
    new_wav_files = []
    
    for wav_file in wav_files:
        wav_id = os.path.basename(wav_file).split('.')[0]
        if wav_id in labels_dict:
            labels.append(labels_dict[wav_id])
            new_wav_files.append(wav_file)
            
    return new_wav_files, labels


#---------------------------- get audio & transcripts ------------------------#
def get_audio_and_transcriptch(txt_files, wav_files, num_input, num_context,
                               word_num_map, txt_labels=None):
    audio = []
    audio_len = []
    transcript = []
    transcript_len = []
    
    # get txt_labels #
    if txt_files != None:
        txt_labels = txt_files
    
    for txt_obj, wav_file in zip(txt_labels, wav_files):
        # load audio & convert to features #
        audio_data = audiofile2input_vector(wav_file, num_input, num_context)
        audio_data = audio_data.astype('float32') # convert to mfcc

        audio.append(audio_data)
        audio_len.append(np.int32(len(audio_data)))
        
        # load text transcription & convert to numerical array #
        target = []
        if txt_files != None:
            target = get_ch_label_v(txt_obj, word_num_map) # convert to vector
        else:
            target = get_ch_label_v(None, word_num_map, txt_obj)
            
        transcript.append(target)
        transcript_len.append(np.int32(len(target)))
    
    audio = np.asarray(audio)
    audio_len = np.asarray(audio_len)
    transcript = np.asarray(transcript)
    transcript_len = np.asanyarray(transcript_len)
    
    return audio, audio_len, transcript, transcript_len
    

#------------------------ audiofile2input_vector() ------------------------# 
def audiofile2input_vector(audio_filename, numcep, numcontext):
    # load wav files #
    fs, audio = wav.read(audio_filename)
    
    # get mfcc coefficients #
    origin_inputs = mfcc(audio, samplerate=fs, numcep=numcep)
    
    origin_inputs = origin_inputs[::2] # take rows by skipping 
    
    train_inputs = np.array([], np.float32)
    train_inputs.resize((origin_inputs.shape[0], numcep + 2*numcep*numcontext))
    
    # pre-fix post fix context #
    empty_mfcc = np.array([])
    empty_mfcc.resize((numcep)) # ?
    
    # prepare train_inputs with past and future contexts #
    time_slices = range(train_inputs.shape[0])
    context_past_min = time_slices[0] + numcontext # the 1st one with full past context
    context_future_max = time_slices[-1] - numcontext # the last one with full future context
    
    for time_slice in time_slices: # from 1st to last one 
        need_empty_past = max(0, (context_past_min - time_slice))
        empty_source_past = list(empty_mfcc for empty_slots in range(need_empty_past))
        data_source_past = origin_inputs[max(0, time_slice - numcontext):time_slice]
        assert(len(empty_source_past)+len(data_source_past) == numcontext)

        if need_empty_past:
            past = np.concatenate((empty_source_past, data_source_past))
        else:
            past = data_source_past
        
        need_empty_future = max(0, (time_slice - context_future_max))
        empty_source_future = list(empty_mfcc for empty_slots in range(need_empty_future))
        data_source_future = origin_inputs[(time_slice + 1):(time_slice + numcontext + 1)]
        assert(len(empty_source_future)+len(data_source_future) == numcontext)
        
        if need_empty_future:
            future = np.concatenate((data_source_future, empty_source_future))
        else:
            future = data_source_future
        
        past = np.reshape(past, numcontext*numcep)
        now = origin_inputs[time_slice]
        future = np.reshape(future, numcontext*numcep)
        
        train_inputs[time_slice] = np.concatenate((past, now, future))
        assert(len(train_inputs[time_slice]) == numcep + 2*numcep*numcontext)
        
    train_inputs = (train_inputs - np.mean(train_inputs)) / np.std(train_inputs)
    
    return train_inputs
        
        
#----------------------------- get_ch_label_v() ------------------------------#
def get_ch_label_v(txt_file, word_num_map, txt_label=None):
    
    word_size = len(word_num_map)
    
    word2num = lambda word: word_num_map.get(word, word_size)
    
    if txt_file != None:
        txt_label = get_ch_label(txt_file)
        
    labels_vector = list(map(word2num, txt_label))
    
    return labels_vector
    
        
def get_ch_label(txt_file):

    labels = ''
    with open(txt_file, 'rb') as f:
        for label in f:
            labels += label.decode('gb2312')
            
    return labels

#--------------------------- pad_sequences() ---------------------------------#
def pad_sequences(sequences, max_len=None, dtype=np.float32, padding='post', truncating='post', value=0.0):
    lengths = np.asarray([len(s) for s in sequences], dtype=np.int32)
    num_samples = len(sequences)
    
    # define the maximum length of sequence #
    if max_len == None:
        max_len = np.max(lengths)
    
    sample_shape = ()
    
    # define the sample_shape as the shape of first non-empty sequnece #
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break
    
    x = (np.ones((num_samples, max_len) + sample_shape) * value).astype(dtype)
    
    for idx, s in enumerate(sequences):
        if len(s) == 0: # empty list
            continue
        if truncating=='pre': # pre truncating sequences with max_len
            trunc = s[-max_len:]
        elif truncating == 'post': # post truncating sequences with max_len
            trunc = s[:max_len] 
        else:
            raise ValueError('Truncating type must be "pre" or "post"')

        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('position %s has different shape compared with expected shape', idx)

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise(ValueError('Padding type must be "pre" or "post"'))
            
    return x, lengths

#--------------------------- sparse_tuple_from ------------------------------#
def sparse_tuple_from(sequences, dtype=np.float32):
    '''convert dense matrix to sparse matrix'''
    indices = []
    values = []
    
    for idx, seq in enumerate(sequences):
        indices.extend(zip([idx] * len(seq), range(len(seq))))
        values.extend(seq)
        
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), indices.max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape    

#---------------------------- word vector to text ----------------------------#
def sparse_tuple2texts_ch(tuple, words):
    









