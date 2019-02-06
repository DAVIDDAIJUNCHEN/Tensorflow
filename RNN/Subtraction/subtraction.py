# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 23:48:35 2019
Project task: Subtraction by RNN 
Module task: Basic function
@author: daijun.chen
"""

#-----------------------------------------------------------------------------#
import numpy as np
import copy

seed = 0
np.random.seed(seed)

def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

def sigmoid2derivative(output):
    return output*(1 - output)

bin_digit = 8
max_num = pow(2, bin_digit)
int2bin = {}
binary = np.unpackbits(np.array([range(max_num)], dtype=np.uint8).T, axis=1)

for i in range(max_num):
    int2bin[i] = binary[i]
    
alpha = 0.01
input_dim = 2
hidden_dim = 16
output_dim = 1

synapse_0 = (2*np.random.random((input_dim, hidden_dim)) - 1)*0.05 # [input_dim, hidden_dim]
synapse_1 = (2*np.random.random((hidden_dim, output_dim)) - 1)*0.05 
synapse_h = (2*np.random.random((hidden_dim, hidden_dim)) - 1)*0.05

synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)
 
sample_size = 1000000

for j in range(sample_size):
    a_int = np.random.randint(max_num)
    b_int = np.random.randint(max_num/2)
    if a_int < b_int:
        tmp_int = a_int
        a_int = b_int
        b_int = tmp_int
    
    a = int2bin[a_int]
    b = int2bin[b_int]
    
    c_int = a_int - b_int
    c = int2bin[c_int]
    
    d = np.zeros_like(c)
    overallError = 0
    layer_2_deltas = []
    layer_1_values = []
    
    layer_1_values.append(np.ones(hidden_dim)*0.1)
    

    for position in range(bin_digit):
        X = np.array([[a[bin_digit - position -1], b[bin_digit - position -1]]])
        y = np.array([[c[bin_digit - position -1]]]).T
        layer_1 = sigmoid(np.dot(X, synapse_0) + np.dot(layer_1_values[-1], synapse_h))
        layer_2 = sigmoid(np.dot(layer_1, synapse_1))

        layer_2_error = y - layer_2
        layer_2_deltas.append((layer_2_error)*sigmoid2derivative(layer_2)) 
        overallError += np.abs(layer_2_error[0])
    
        d[bin_digit-position-1] = np.round(layer_2[0][0])
        layer_1_values.append(copy.deepcopy(layer_1))
    
    future_layer_1_delta = np.zeros(hidden_dim)


    for position in range(bin_digit):
        X = np.array([[a[position], b[position]]])
        layer_1 = layer_1_values[-position - 1]
        prev_layer_1 = layer_1_values[-position - 2]
        layer_2_delta = layer_2_deltas[-position -1]
        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) +
                     layer_2_delta.dot(synapse_1.T))*sigmoid2derivative(layer_1)
        synapse_1_update = np.atleast_2d(layer_1).T.dot(layer_2_delta)
        synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        synapse_0_update += X.T.dot(layer_1_delta)
    
        future_layer_1_delta = layer_1_delta
    
    synapse_0 += synapse_0_update*alpha
    synapse_1 += synapse_1_update*alpha
    synapse_h += synapse_h_update*alpha
    synapse_0_update *= 0
    synapse_1_update *= 0
    synapse_h_update *= 0


    if (j%800) == 0:
        print('Total error:', str(overallError))
        print('Pred:', str(d))
        print('True:', str(c))
        out = 0
        for index, x in enumerate(reversed(d)):
            out += x*pow(2, index)
        print(str(a_int)+' - '+str(b_int)+ ' = ' + str(out))
        print('-------------------------------------------')

