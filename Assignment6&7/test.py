#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 19:02:09 2022

@author: shiva
"""
import matplotlib.pyplot as plt
from mnist import Mnist
import numpy as np
import sys
import cv2
from two_layer_net import TwoLayerNet

#const definition
mnist_pkl_filename = 'nidamanuri_mnist_nn_model.pkl'

#%%
mnist=Mnist()
if len(sys.argv)>1:
    img_idx=int(sys.argv[1])

#%%

import glob

path = glob.glob("*.PNG")
fnl_image=[]
predicted_nums=[]
labels=[]
image2display=[]
for img in path:
    n=cv2.imread(img)
    n=cv2.resize(n, (28, 28))
    predicted_nums.append(img[:1])
    labels.append(img[:1])
    image2display.append(n)
    img_array = np.asarray(n)
    image = cv2.cvtColor(img_array , cv2.COLOR_BGR2GRAY) #(28, 28)
    image  = cv2.resize(image, (28, 28 ))
    image = image / 255
    image = image.reshape(1, 784)
    fnl_image.append(image)
plt.imshow(image2display[img_idx], cmap=plt.get_cmap('gray'))
plt.show()

(x_train, t_train), (x_test, t_test) = mnist.load_data(normalize=True, one_hot_label=True)
                                        
#hyperparameters

iterations = 10000
batch_size = 100
learning_rate = 0.1

hidden_size = 500
network = TwoLayerNet(input_size = mnist.img_size, hidden_size = hidden_size, output_size = 10)

network.load_model(mnist_pkl_filename)
y = TwoLayerNet.predict(network, fnl_image[img_idx])
predicted_num=np.argmax(y)
certainity=np.max(y)

print('Image {} labeled {} in the test set is predited as {}.'.format(img_idx,labels[img_idx],predicted_nums[img_idx]))
