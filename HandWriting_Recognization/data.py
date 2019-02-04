# -*- coding: utf-8 -*-
"""
Project task: handwriting recognization
module task: input mnist data set  
Created on Sun Feb  3 16:29:41 2019
@author: daijun.chen
"""

# 
from tensorflow.examples.tutorials.mnist import input_data
import pylab

mnist = input_data.read_data_sets('MNIST_DATA/', one_hot=True)
train_im0 = mnist.train.images[1]
train_im0 = train_im0.reshape(-1, 28)
train_label0 = mnist.train.labels[1]

print('The first image in training set')
print('The first label in training set {}'.format(train_label0))

pylab.imshow(train_im0)

