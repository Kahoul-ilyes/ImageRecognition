import tensorflow as tf
from tensorflow.keras import datasets, layers, models

from pathlib import Path

MODEL_PATH = './models/'


class Model:
    def __init__(self):
        self.model_dir = './models/'
        self.model = None

    def create(self, image_width, image_height):
        self.model = models.Sequential([
            layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(image_height, image_width, 3)),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dense(1)
        ])

    def compile(self):
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                           metrics=['accuracy'])

    def train(self, train_images, train_labels, epochs, val_images, val_labels):
        return self.model.fit(
            train_images,
            train_labels,
            epochs=epochs,
            validation_data=(val_images, val_labels),
        )

    def evaluate(self, val_images, val_labels):
        return self.model.evaluate(val_images, val_labels, verbose=2)

    def save(self, name):
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        self.model.save(self.model_dir + name)

    def load_previous_model(self, name):
        try:
            self.model = tf.keras.models.load_model(self.model_dir + name)
        except:
            self.model = None
