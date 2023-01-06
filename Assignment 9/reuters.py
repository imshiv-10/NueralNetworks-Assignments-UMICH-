#import os 
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Import packages
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import reuters
#from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical

# default values
NUM_WORDS = 10000
BATCH_SIZE = 512

class Reuters:
    def __init__(self, num_words = NUM_WORDS):
        self.x_train, self.y_train, \
        self.x_test, self.y_test = self._load(num_words)
        self.model = self._model()
        

    def _load(self, num_words):
        (x_train_raw, y_train_raw), (x_test_raw, y_test_raw) \
            = reuters.load_data(num_words=num_words)

        # vectorize reviews
        x_train = self._vectorize_sequences(x_train_raw, num_words)
        y_train = np.asarray(y_train_raw).astype('float32')

        x_test = self._vectorize_sequences(x_test_raw, num_words)
        y_test = np.asarray(y_test_raw).astype('float32')
        y_train=to_categorical(y_train)
        y_test=to_categorical(y_test)
        #x_train, x_val, y_train, y_val \
        #    = train_test_split(x_train_vec, y_train, test_size=0.2)

        return x_train, y_train, x_test, y_test


    def _model(self):
        model = keras.Sequential([
            layers.Dense(64, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(46, activation="softmax")
        ])

        # model compilation
        model.compile(optimizer="rmsprop",
                        loss="categorical_crossentropy",
                        metrics=["accuracy"])
        return model


    def plot_loss(self, history):
        # Plotting the training and validation loss
        history_dict = history.history
        loss_values = history_dict["loss"]
        val_loss_values = history_dict["val_loss"]
        epochs = range(1, len(loss_values) + 1)
        plt.figure(1)
        plt.plot(epochs, loss_values, "r", label="Training loss")
        plt.plot(epochs, val_loss_values, "r--", label="Validation loss")
        plt.title("Training and validation loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()


    def plot_accuracy(self, history):
        history_dict = history.history
        acc = history_dict["accuracy"]
        val_acc = history_dict["val_accuracy"]
        epochs = range(1, len(acc) + 1)
        plt.figure(2)
        plt.plot(epochs, acc, "b", label="Training acc")
        plt.plot(epochs, val_acc, "b--", label="Validation acc")
        plt.title("Training and validation accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()


    def train(self, epochs=20):
        if self.model is None: 
            print('[INFO] model is not defined.')
            return
        
        history = self.model.fit(self.x_train, self.y_train, \
                                 epochs = epochs, validation_split = 0.2,
                                 batch_size = BATCH_SIZE)
        self.plot_loss(history)
        self.plot_accuracy(history)
        return history
        
        
    def evaluate(self):
        score = self.model.evaluate(self.x_test, self.y_test)
        print(f'[INFO] Test loss: {score[0]}')
        print(f'[INFO] Test accuracy: {score[1]}')
        
        
    def _vectorize_sequences(self, sequences, dimension=10000):
        results = np.zeros((len(sequences), dimension))
        for i, sequence in enumerate(sequences):
            for j in sequence:
                results[i, j] = 1.
        return results
    def to_one_hot(labels, dimension=46):
        results = np.zeros((len(labels), dimension))
        for i,label in enumerate(labels):
            results[i,label] = 1
        return results