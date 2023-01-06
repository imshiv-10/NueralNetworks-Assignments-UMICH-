# %%
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# %%
# y = w x + b
# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

randomIndx = np.random.randint(60000)
print(f"Random Index: {randomIndx}")
plt.imshow(x_train[randomIndx].reshape(28,28), cmap = 'gray')
plt.title(y_train[randomIndx])
print(f"Shape of image: {np.shape(x_train[randomIndx])}")
# %%
# y = w x + b
# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# a two-layer net using the Dense layer and Sequential Model.
model = keras.Sequential([
    layers.Dense(512 , activation="relu"),
    layers.Dense(10, activation ="softmax")
])

# compile the network and training the neural network.
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Scale images to the [0, 1] range.
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2]) 
x_train = x_train / 255.0
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
x_test = x_test / 255.0


model.fit(x_train, y_train , epochs = 20 , batch_size= 2048)
model.save('kwon_mn_mnist')

# Testin the network with the first 12 samples in the test set.
test_digits = x_test[:12]
# Using predict() function to test.
predictions = model.predict(test_digits)

# %%
# evaluate the model 
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1]) 

# %%
print(predictions)
tempList = []

#this loop iterates through y_test and converts a format so we can compare to our predictions
for row in y_test:
    tempClass = np.argmax(row)
    tempList.append(tempClass)


y_testClasses = np.array(tempList)
print(y_testClasses)

# %%
wrongPredictionIndexes = []
i = 0
for actual, prediction in zip('y_testClasses', predictions):
    if actual != prediction:
        wrongPredictionIndexes.append(i)
    i = i + 1

# %%
# plotting 12 tests
randomIntList = np.random.randint(len(wrongPredictionIndexes), size = 12)
plot =1
plt.figure()
for randomNum in randomIntList:
    plt.subplot(3,4,plot)
    imageIndex = wrongPredictionIndexes[randomNum]
    plt.title(f'i: {imageIndex}, l:{ y_test[imageIndex]}, p:{predictions[imageIndex].argmax()}')
    plt.subplots_adjust(hspace=0.85)
    plt.imshow(x_test[imageIndex].reshape(28,28), cmap = 'gray')
    plot = plot+1
plt.show()


randomIntList = np.random.randint(len(wrongPredictionIndexes), size = 12)
plot =1
plt.figure()
for randomNum in randomIntList:
    plt.subplot(3,4,plot)
    imageIndex = wrongPredictionIndexes[randomNum]
    plt.title(f'i: {imageIndex}, l:{ y_test[imageIndex]}, p:{predictions[imageIndex].argmax()}')
    plt.subplots_adjust(hspace=0.85)
    plt.imshow(x_test[imageIndex].reshape(28,28), cmap = 'gray')
    plot = plot+1
plt.show()

# %%



