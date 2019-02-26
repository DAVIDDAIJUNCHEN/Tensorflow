# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 09:42:16 2019
Project task: build machine translation system
Module task: processing tools 
@author: daijun.chen
"""

import os as os
import re
import jieba
import collections
import random as rand
import numpy as np

from tensorflow.python.platform import gfile

#------------ create vocabulary and convert file folder to id folder ---------# 
def create_dicts(data_dir, en_voc_filename, ch_voc_filename, from_data_dir, to_data_dir, voc_size):
    # make vocabulary file path
    en_voc_file = os.path.join(data_dir, en_voc_filename)
    ch_voc_file = os.path.join(data_dir, ch_voc_filename)
    
    # create English dict #
    en_train_data, en_count, en_dict, en_reverse_dict, en_text = create_vocabulary(
            data_dir, en_voc_file, from_data_dir, voc_size, Isch=False, normalize_digits=True)

    print('Word size of English training data {}'.format(len(en_train_data)))
    print('Word size of English dictionary {}'.format(len(en_dict)))
    
    # create Chinese dict #
    ch_train_data, ch_count, ch_dict, ch_reverse_dict, ch_text = create_vocabulary(
            data_dir, ch_voc_file, to_data_dir, voc_size, Isch=True, normalize_digits=True)
    
    print('Word size of English training data {}'.format(len(ch_train_data)))
    print('Word size of English dictionray {}'.format(len(ch_dict)))

    # initialize vocabulary： get {'a':0, 'b':1} voc dir #
    en_vocabulary, en_rev_vocabulary = initialize_vocabulary(en_voc_file)
    ch_vocabulary, ch_rev_vocabulary = initialize_vocabulary(ch_voc_file)

    # convert text dir to ids dir #
    if not os.path.exists(data_dir+'fromids/'):
        os.makedirs(data_dir+'fromids/')
    text_dir2id_dir(from_data_dir, data_dir+'fromids/', en_vocabulary, normalize_digits=True, Isch=False)

    if not os.path.exists(data_dir+'toids/'):
        os.makedirs(data_dir+'toids/')
    text_dir2id_dir(to_data_dir, data_dir+'toids/', ch_vocabulary, normalize_digits=True, Isch=True)

    
#------------------  
def create_vocabulary(data_dir, voc_file, raw_data_dir, max_voc_size, Isch=False, normalize_digits=True):
    # get tokenized texts list and size list  
    texts, _texts_size = get_ch_path_text(raw_data_dir, Isch, normalize_digits)

    print(texts[0], len(texts))
    print('Number of rows: ', len(_texts_size), _texts_size)

    all_words = []
    
    for label in texts:
        print('length of words in a single sentence: {}'.format(len(label)))
        all_words += [word for word in label]

    print('length of words in all files: {}'.format(len(all_words)))
    
    # get training_label: [1,3,4,], sequence of word orders, 
    _train_data, _count, _dictionary, _reverse_dict = build_dataset(all_words, max_voc_size)
    
    if not gfile.Exists(voc_file):
        print('Need to create vocabulary %s from data %s', (voc_file, data_dir))
        #if len(reverse_dict) > max_voc_size:
        
        with gfile.GFile(voc_file, mode='w') as vocab_file:
            for w in _reverse_dict:
                print(_reverse_dict[w])
                vocab_file.write(_reverse_dict[w] + '\n')
    else:
        print('We had vocabulary! Go ahead!')
            
    return _train_data, _count, _dictionary, _reverse_dict, _texts_size 

#-------------- get tokenized training data & data sizes lists ---------------#
def get_ch_path_text(raw_data_dir, Isch=False, normalize_digits=True):
    text_files, _ = getRawFileList(raw_data_dir) #get file paths and names list
    labels = []
    training_data_sizes = [0]
    
    if len(text_files) == 0:
        print('Erros: there is no files in the dir ', raw_data_dir)
    
    print('There are {} '.format(len(text_files)), ' files, the 1st one is ', text_files[0])
    
    rand.shuffle(text_files) # shuffle the file paths list
    
    for text_file in text_files:
        # get the tokenized training data and data sizes lists 
        training_data, training_data_sizes = get_ch_label(text_file, Isch, normalize_digits)
        training_tokenized = np.array(training_data)
        training_tokenized = np.reshape(training_tokenized, [-1, ])
        labels.append(training_tokenized)
        
        training_data_size = np.array(training_data_sizes) + training_data_sizes[-1]
        training_data_sizes.extend(list(training_data_size))
        
    return labels, training_data_sizes
    
#----------------- Get relative files paths list and names list --------------#
def getRawFileList(raw_data_dir):
    files = []
    names = []
    
    for f in os.listdir(raw_data_dir):
        if not f.endswith('~') or not f=="":
            files.append(os.path.join(raw_data_dir, f))
            names.append(f)
            
    return files, names

#------------------ get tokenized text and text size in one file -------------#    
def get_ch_label(text_file, Isch=False, normalize_digits=False):
    labels = []
    labels_size = []
    
    with open(text_file, 'rb') as f:
        for label in f:
            linestr = label.decode('utf-8')
            if normalize_digits: # 
                linestr = re.sub('\d+', _NUM, linestr)

            notoken = basic_tokenizer(linestr)

            if Isch:             # if Chinese, then use jieba tokenization
                notoken = tokenize(notoken)
            else:                # if English, then just split with space
                notoken = notoken.split() 
            
            labels.extend(notoken)  # extend labels list with tokenized str
            labels_size.append(len(labels)) # append len of labels 

    return labels, labels_size

#------------------------ get rid of the signs, like 。 ， ‘ ’ -----------------#
def basic_tokenizer(string):
    _WORD_SPLIT = "([.,!?\"':;)(])"
    _CHWORD_SPLIT = '、|。|，|‘|’'
    str1 = ""
    for i in re.split(_CHWORD_SPLIT,  string):
        str1 = str1 + i
    str2 = ""
    for i in re.split(_WORD_SPLIT ,  str1):
        str2 = str2 + i
    return str2    

#------ tokenize a string and output a list containing tokenized words -------#
def tokenize(word_str):
    str_token = jieba.cut(word_str)
    str_token = ' '.join(str_token)
    list_token = str_token.split()
    
    return list_token

#----------------- 
# system defined signs
_PAD = '_PAD'  # placeholder signs in bucket mechenism 
_GO = '_Go'    # head bit when decode inputs     
_EOS = '_EOS'  # end of outputs
_UNK = '_UNK'  # unknown characters in dict
_NUM = '_NUM'  # signs replacing the number in text 

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

#--------------- 
def build_dataset(words_list, num_words):
    count = [[_PAD, -1], [_GO, -1], [_EOS, -1], [_UNK, -1]]
    counter = collections.Counter(words_list)     # convert word_list to Counter 
    counter_top= counter.most_common(num_words)   # pick the top num_words items
    count.extend(counter_top)  # counter_top is a list [('a', a_num), ('b', b_num)]
    
    dictionary = {}
    
    for word, _ in count:   # construct the word-rank mapping 
        dictionary[word] = len(dictionary)        # {'a':1, 'b':2, 'c':3}
        
    data = []
    unk_count = 0
    
    for word in words_list:
        if word in dictionary:          # if word in cutted dict
            index = dictionary[word]    # get the index of word
        else:
            index = 0                   # if word is not in cutted dict
            unk_count += 1              # let the word be unk      
        data.append(index)

    count[0][1] = unk_count
    
    reverse_dict = dict(zip(dictionary.values(), dictionary.keys()))        
        
    return data, count, dictionary, reverse_dict # data: [2,3,5,3]
    
#------- generate vocabulary with order number --{'a':0, 'b':1}---------------#
def initialize_vocabulary(voc_file):
    if gfile.Exists(voc_file):
        _reverse_vocabulary = []
        
        with gfile.GFile(voc_file, 'r') as f:
            _reverse_vocabulary.extend(f.readlines())
            _reverse_vocabulary = [line.strip() for line in _reverse_vocabulary]

        _vocabulary = dict([(j, i) for i, j in enumerate(_reverse_vocabulary)])            

        return _vocabulary, _reverse_vocabulary
    else:
        raise ValueError('%s does not contain vocabulary file', voc_file)

#---------------- convert text files in folder to id files -------------------# 
def text_dir2id_dir(raw_data_dir, id_dir, vocabulary, normalize_digits=True, Isch=False):
    # get file paths and file names 
    filepaths, filenames = getRawFileList(raw_data_dir)
    if len(filenames) == 0:
        raise ValueError('No text file in folder %s' % raw_data_dir)
    else:
        for text_file, name in zip(filepaths, filenames):
            id_file = id_dir + name
            text_file2ids_file(text_file, id_file, vocabulary, normalize_digits, Isch)
            print('Text file: %s ==> Id file: %s ' % (text_file, id_file))
    
#---------------- convert single data file to ids file -----------------------# 
def text_file2ids_file(origin_file, target_file, vocabulary, normalize_digits=True, Isch=False):
    if not gfile.Exists(target_file):
        print('Start creating data in file %s' % target_file)
        
        with gfile.GFile(origin_file, 'r') as data_file:
            
            with gfile.GFile(target_file, 'w') as ids_file:

                for line in data_file:
                    print(line)
                    token_ids = sentence2ids(line, vocabulary, normalize_digits, Isch)
                    ids_file.write(' '.join([str(tok) for tok in token_ids]) + '\n')

#---------------- convert single sentence to ids sequence --------------------#
def sentence2ids(sentence, vocabulary, normalize_digits=True, Isch=False):
    if normalize_digits == True:
        sentence = re.sub('\d+', _NUM, sentence) # replace digits string with _NUM
    notoken = basic_tokenizer(sentence)  # strip the sentence with ‘ ’ 。 ，

    if Isch == True:
        notoken = tokenize(notoken)
    else:
        notoken = notoken.split()
    # get ids sequence for the seq of tokenized words       
    ids_sentence = [vocabulary.get(word, UNK_ID) for word in notoken]

    return ids_sentence
        
