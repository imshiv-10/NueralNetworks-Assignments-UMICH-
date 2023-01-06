
import os, shutil, pathlib
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt

class CatsDogs:
    def __init__(self):
        self.original_dir = pathlib.Path("train")
        
        self.new_base_dir = pathlib.Path("cats_vs_dogs")
        
        self.make_subset("train", start_index = 0, end_index=1000)
        self.make_subset("validation", start_index = 1000, end_index=1500)
        self.make_subset("test", start_index=1500, end_index=2500)
        
        self.train_dataset = image_dataset_from_directory(self.new_base_dir / "train", image_size=(180, 180), batch_size=32)
        self.validation_dataset = image_dataset_from_directory(self.new_base_dir / "validation", image_size=(180, 180), batch_size=32)
        self.test_dataset = image_dataset_from_directory(self.new_base_dir / "test", image_size=(180, 180), batch_size=32) 
        
        
        
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
                
    def model(self):           
            inputs = keras.Input(shape=(180, 180, 3))
            x = layers.Rescaling(1./255)(inputs)
            x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
            x = layers.MaxPooling2D(pool_size=2)(x)
            x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
            x = layers.MaxPooling2D(pool_size=2)(x)
            x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
            x = layers.MaxPooling2D(pool_size=2)(x)
            x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
            x = layers.MaxPooling2D(pool_size=2)(x)
            x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
            x = layers.Flatten()(x)
            outputs = layers.Dense(1, activation="sigmoid")(x)
            self.model = keras.Model(inputs=inputs, outputs=outputs)
            
            self.model.compile(loss="binary_crossentropy",
            optimizer="rmsprop",
            metrics=["accuracy"])
            
            self.callbacks = [keras.callbacks.ModelCheckpoint(filepath="convnet_from_scratch.keras",
                                                save_best_only=True,
                                                monitor="val_loss")]
        
    def train(self):
        if(os.path.exists('convnet_from_scratch.keras')):
            self.model = keras.models.load_model('convnet_from_scratch.keras')
            self.history = self.model.evaluate(self.test_dataset)
        else: 
            self.model()
            self.history = self.model.fit(self.train_dataset, epochs=30, validation_data=self.validation_dataset, callbacks=self.callbacks)
              
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
        pred = self.model.predict(self.test_dataset)
        print(pred)