
# %% creating object for mnist class
mnist_test = mnist()
mnist_test.download_mnistDatasets()

# x_train = mnist_test.load_images(key_file['train_img'])
# y_train = mnist_test.load_labels(key_file['train_label'])

x_test = mnist_test.load_images(mnist_test.key_file['test_img'])
y_test = mnist_test.load_labels(mnist_test.key_file['test_label'])

print(len(sys.argv))
print(sys.argv)

img_idx = 100
if len(sys.argv) > 1:
    img_idx = int(sys.argv[1])

img = x_test[img_idx]
label = y_test[img_idx]

img2show = img.reshape(mnist_test.img_dim[1], mnist.img_dim[2])
plot.imshow(img2show, cmap="gray")
plot.show()

mnist_test.init_network()
y = mnist_test.predict(img)
predicted_num = np.argmax(y)
certainty = np.max(y)

print("image #{} labeled {} in the test set is predicted as {} with certainity ".format(img_idx, label, y, certainty))





# URL request python library
import os.path
import urllib.request
from gzip import GzipFile
import matplotlib.pyplot as plot
import pickle

import numpy as np

import gzip

url_base = 'http://yann.lecun.com/exdb/mnist/'
key_file = {
    'train_img': 'train-images-idx3-ubyte.gz',
    'train_label': 'train-labels-idx1-ubyte.gz',
    'test_img': 't10k-images-idx3-ubyte.gz',
    'test_label': 't10k-labels-idx1-ubyte.gz'
}


# pickle_filename = 'mnist.pkl'

# Downloading all mnist datasets in to the assignment5 folder
def download_mnistDatasets():
    for value in key_file.values():
        if (os.path.exists(value)):
            print('{} file already exists....'.format(value))
        else:
            print(' downloading {}.....'.format(value))
            urllib.request.urlretrieve(url_base + value, value)
            print('Done....')


img_size = 784  # 28*28
img_dim = (1, 28, 28)
train_dim = 60000
test_num = 10000
weights_file_name = 'sample_weight.pkl'


# load images
def load_images(filename):
    with gzip.open(filename, 'rb') as f:
        images = np.frombuffer(f.read(), np.uint8, offset=16)
    images = images.reshape(-1, img_size)
    print('Done with loading images', filename)
    return images


# load labels
def load_labels(filename):
    with gzip.open(filename, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print('Done with loading labels', filename)
    return labels


if __name__ == '__main__':
    download_mnistDatasets()
    x_train = load_images(key_file['train_img'])
    y_train = load_labels(key_file['train_label'])

    x_test = load_images(key_file['test_img'])
    y_test = load_labels(key_file['test_label'])

    img_idx = 0
    img = x_train[img_idx]
    label = y_train[img_idx]

    img = x_test[img_idx]
    label = y_test[img_idx]

    img = img.reshape(img_dim[1], img_dim[2])
    plot.imshow(img, cmap="gray")

    # load pre-trained model


def init_network():
    with open(weights_file_name, 'rb') as f:
        network = pickle.load(f)
    return network


# network is the loaded network
# assuming network paraameterer were loaded already
# x is input (784)


def sigmoid(a):
    return 1 / (1 + np.exp(-a))


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def predict(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)

    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)

    a3 = np.dot(z2, w3) + b3
    # z3 = sigmoid(a3)
    y = softmax(a3)
    return y


# %%
img_idx = 100
img = x_test[img_idx]
label = y_test[img_idx]

img2show = img.reshape(img_dim[1], img_dim[2])
plot.imshow(img2show, cmap="gray")

network = init_network()
y = predict(network, img)
predicted_num = np.argmax(y)
certainty = np.max(y)

print("image {} labeled {} in the test set is predicted as {} with certainity ".format(img_idx, label, y, certainty))

# load_images('train-images-idx3-ubyte.gz')
# load_labels('t10k-labels-idx1-ubyte.gz')