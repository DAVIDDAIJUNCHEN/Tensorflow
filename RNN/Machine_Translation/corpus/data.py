i# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 20:06:05 2019
Project task: seq2seq machine translation
module task: process data 
@author: daijun.chen
"""

import tensorflow as tf
from tools import create_dicts

# load parallel corpus and generate dicts #
data_dir = 'corpus/'
from_data_dir = 'corpus/from/'
to_data_dir = 'corpus/to/'

en_voc_filename = 'en_voc.txt' # name of english vocabulary file
ch_voc_filename = 'ch_voc.txt' # name of chinese vocabulary file

voc_size = 40000

create_dicts(data_dir, en_voc_filename, ch_voc_filename, from_data_dir, to_data_dir, voc_size)







