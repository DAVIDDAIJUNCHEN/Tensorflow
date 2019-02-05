# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 12:53:40 2019
Project: Multilayer Linear Perceptron (MLP)
module task: Activation function class
@author: daijun.chen
"""

import tensorflow as tf
import numpy as np

class Activation(object):
    '''Need to input the linear transformation output '''
    def __init__(self, input):
        self.input = input
        
    
    '''Relu'''
    def relu(self):
        return tf.max(self.input, 0)
    
    '''Sigmoid'''
    def sigmoid(self):
        return 1/(1 + tf.exp(-self.input))
    
    '''Tanh'''
    def tanh(self):
        return 2*self.sigmoid() -1
    
    '''Softplus'''
    def softplus(self):
        return tf.log(1+tf.exp(self.input))
    
    '''Noisy relu'''
    def noisy_relu(self):
        return tf.max(0, tf.random_normal(1)+self.input)
    
    '''Leaky relu'''
    def leaky_relu(self, leak=0.01):
        return tf.maximum(self.input, leak*self.input)
    
    '''Elus'''
    def elus(self, leak):
        return tf.maximum(self.input, leak*(tf.exp(self.input)-1.))
    
    '''Swish'''
    def swish(self, beta):
        return self.input/(1.+tf.exp(-self.input*beta))

    '''relu6'''
    def relu6(self):
        return self.maximum(self.input, 6.)
            
