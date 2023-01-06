#import os 
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Import packages
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import boston_housing
#from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# default values
NUM_WORDS = 10000
BATCH_SIZE = 512

class Boston_Housing:
    def __init__(self):
        self.x_train, self.y_train, \
        self.x_test, self.y_test = self._load()
        self.model = self._model()
        

    def _load(self):
        (x_train, y_train), (x_test, y_test) \
            = boston_housing.load_data()

        # vectorize reviews
        mean=x_train.mean(axis=0)
        x_train-=mean
        std=x_train.std(axis=0)
        x_train/=std
        x_test-=mean
        x_test/=std
        
        return x_train, y_train, x_test, y_test


    def _model(self):
        model = keras.Sequential([
            layers.Dense(64, activation="relu",input_shape=(self.x_train.shape[1],)),
            layers.Dense(64, activation="relu"),
            layers.Dense(1)
        ])

        # model compilation
        model.compile(optimizer="rmsprop",
                        loss="mse",
                        metrics=["mae"])
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
    def k_fold(self):
          k = 3
          num_val_samples = len(self.x_train) // k
          num_epochs = 40
          all_scores = []
          for i in range(self, k):
             print(f'Processing fold # {i}')
             val_data = self.x_train[i * num_val_samples: (i+1) * num_val_samples]
             val_targets = self.y_train[i * num_val_samples: (i+1) * num_val_samples]
             partial_train_data = np.concatenate(
                                     [self.x_train[:i * num_val_samples],
                                     self.x_train[(i+1) * num_val_samples:]],
                                     axis=0)
             partial_train_targets = np.concatenate(
                                     [self.y_train[:i * num_val_samples],
                                     self.y_train[(i+1)*num_val_samples:]],
                                     axis=0)
             model = self.train()
             model.fit(partial_train_data,
                       partial_train_targets,
                       epochs=num_epochs,
                       batch_size=16,
                       verbose=0)
             val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
             all_scores.append(val_mae)