
import os, shutil, pathlib
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt
import numpy as np


class CatsDogsPre:
    def __init__(self):
        self.original_dir = pathlib.Path("train")
        self.new_base_dir = pathlib.Path("cats_vs_dogs")
        self.train_dataset = image_dataset_from_directory(self.new_base_dir / "train", image_size=(180, 180), batch_size=32)
        self.validation_dataset = image_dataset_from_directory(self.new_base_dir / "validation", image_size=(180, 180), batch_size=32)
        self.test_dataset = image_dataset_from_directory(self.new_base_dir / "test", image_size=(180, 180), batch_size=32) 
        self.conv_base = keras.applications.vgg16.VGG16(
                    weights="imagenet",
                    include_top=False,
                    input_shape=(180, 180, 3))
        self.train_features, self.train_labels =  self.get_features_and_labels(self.train_dataset)
        self.val_features, self.val_labels =  self.get_features_and_labels(self.validation_dataset)
        self.test_features, self.test_labels =  self.get_features_and_labels(self.test_dataset)
        
    def make_subset(self, subset_name, start_index, end_index):
        for category in ("cat", "dog"):
            self.dir = self.new_base_dir / subset_name / category
            if os.path.exists(self.dir):
                print(f'{self.dir} exists!')
                continue
            os.makedirs(self.dir)
            fnames = [f"{category}.{i}.jpg" for i in range(start_index, end_index)]
            for fname in fnames:
                shutil.copyfile(src=self.original_dir /fname, dst=self.dir / fname)
    def get_features_and_labels(self, dataset):
        all_features = []
        all_labels = []
        for images, labels in dataset:
            preprocessed_images = keras.applications.vgg16.preprocess_input(images)
            features = self.conv_base.predict(preprocessed_images)
            all_features.append(features)
            all_labels.append(labels)
        return np.concatenate(all_features), np.concatenate(all_labels)
                
    def model(self):
            inputs = keras.Input(shape=(5, 5, 512))
            x = layers.Flatten()(inputs)
            x = layers.Dense(256)(x)
            x = layers.Dropout(0.5)(x)
            outputs = layers.Dense(1, activation="sigmoid")(x)
            self.model = keras.Model(inputs, outputs)
            self.model.compile(loss="binary_crossentropy",
                          optimizer="rmsprop",
                          metrics=["accuracy"])
            
            self.callbacks = [
                keras.callbacks.ModelCheckpoint(
                  filepath="feature_extraction.keras",
                  monitor="val_loss")]
        
    def train(self):
        if(os.path.exists('feature_extraction.keras')):
            self.model = keras.models.load_model('feature_extraction.keras')
            self.history = self.model.evaluate(self.test_dataset)
        else: 
            self.model()
            self.history = self.model.fit(self.train_features, self.train_labels, epochs=20, validation_data=(self.val_features, self.val_labels), callbacks=self.callbacks)
              
    def plot(self):
        accuracy = self.history.history["accuracy"]
        val_accuracy = self.history.history["val_accuracy"]
        loss = self.history.history["loss"]
        val_loss = self.history.history["val_loss"]
        epochs = range(1, len(accuracy) + 1)
        plt.plot(epochs, accuracy, "bo", label="Training accuracy")
        plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
        plt.title("Training and validation accuracy")
        plt.legend()
        plt.figure()
        plt.plot(epochs, loss, "bo", label="Training loss")
        plt.plot(epochs, val_loss, "b", label="Validation loss")
        plt.title("Training and validation loss")
        plt.legend()
        plt.show()
     
    def predict(self):
        pred = self.model.predict(self.test_features)
        print(pred)