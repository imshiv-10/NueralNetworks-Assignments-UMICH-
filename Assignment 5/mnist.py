# URL request python library
import os.path
import urllib.request
from gzip import GzipFile
import pickle
import matplotlib.pyplot as plot
import numpy as np
import gzip
import sys


# %% mnist class
class Mnist():
    # attributes of a class
    img_size = 784  # 28*28
    img_dim = (1, 28, 28)
    train_dim = 60000
    test_num = 10000
    weights_file_name = 'sample_weight.pkl'

    def __init__(self):
        self.url_base = 'http://yann.lecun.com/exdb/mnist/'
        self.key_file = {
            'train_img': 'train-images-idx3-ubyte.gz',
            'train_label': 'train-labels-idx1-ubyte.gz',
            'test_img': 't10k-images-idx3-ubyte.gz',
            'test_label': 't10k-labels-idx1-ubyte.gz'
        }
        self.network = None

    # Downloading all mnist datasets in to the assignment5 folder
    def download_mnistDatasets(self):
        for value in self.key_file.values():
            if (os.path.exists(value)):
                print('{} file already exists....'.format(value))
            else:
                print(' downloading {}.....'.format(value))
                urllib.request.urlretrieve(self.url_base + value, value)
                print('Done....')

                # load images

    def load_images(self, filename):
        with gzip.open(filename, 'rb') as f:
            images = np.frombuffer(f.read(), np.uint8, offset=16)
        images = images.reshape(-1, self.img_size)
        print('Done with loading images', filename)
        return images

    # load labels
    def load_labels(self, filename):
        with gzip.open(filename, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
        print('Done with loading labels', filename)
        return labels

    def init_network(self):
        with open(self.weights_file_name, 'rb') as f:
            self.network = pickle.load(f)

    def sigmoid(self, a):
        return 1 / (1 + np.exp(-a))

    def softmax(self, a):
        c = np.max(a)
        exp_a = np.exp(a - c)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        return y

    def predict(self, x):
        w1, w2, w3 = self.network['W1'], self.network['W2'], self.network['W3']
        b1, b2, b3 = self.network['b1'], self.network['b2'], self.network['b3']
        a1 = np.dot(x, w1) + b1
        z1 = self.sigmoid(a1)

        a2 = np.dot(z1, w2) + b2
        z2 = self.sigmoid(a2)

        a3 = np.dot(z2, w3) + b3
        # z3 = sigmoid(a3)
        y = self.softmax(a3)
        return y




