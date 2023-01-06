# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 00:49:35 2022

@author:shiva
"""

import numpy as np

def sigmoid(a):
    return 1/(1 + np.exp(-a))


# sigmoid for backprop
def sigmoid_grad(x):
    return (1.0 - sigmoid(x) * sigmoid(x))


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(x)/np.sum(np.exp(x), axis=-1, keepdims=True)
