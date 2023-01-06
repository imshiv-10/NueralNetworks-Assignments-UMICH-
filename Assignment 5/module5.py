from mnist import Mnist
import sys
import matplotlib.pyplot as plot
import numpy as np


# %% creating object for mnist class
mnist_test = Mnist()
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

img2show = img.reshape(mnist_test.img_dim[1], mnist_test.img_dim[2])
plot.imshow(img2show, cmap="gray")
plot.show()

mnist_test.init_network()
y = mnist_test.predict(img)
predicted_num = np.argmax(y)
certainty = np.max(y)

print("image #{} labeled {} in the test set is predicted as {} with certainity ".format(img_idx, label, y, certainty))

