# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 17:38:42 2022

@author: shiva
"""

import numpy as np
import matplotlib.pyplot as plt

from mnist import Mnist
from two_layer_net import TwoLayerNet

#const definition
mnist_pkl_filename = 'nidamanuri_mnist_nn_model.pkl'

#load mnist datasets
mnist = Mnist()
(x_train, t_train), (x_test, t_test) = mnist.load_data(normalize=True, one_hot_label=True)
                                        

#hyperparameters

iterations = 10000
batch_size = 50
learning_rate = 0.1

hidden_size = 1000
network = TwoLayerNet(input_size = mnist.img_size, hidden_size = hidden_size, output_size = 10)

network.fit(iterations, x_train, t_train, x_test, t_test, batch_size, learning_rate=learning_rate,
            backprop=True)

network.save_model(mnist_pkl_filename)

#visualize loss
plt.figure()
x = np.arange(len(network.train_losses))
plt.plot(x, network.train_losses, label='train loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

#visualize acc
plt.figure()
markers = {'train': 'o', 'test':'s'}
x = np.arange(len(network.train_accs))
plt.plot(x, network.train_accs, label='train accuracy')
plt.plot(x, network.test_accs, label='test accuracy', linestyle='--')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
