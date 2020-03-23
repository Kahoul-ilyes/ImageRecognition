import tensorflow as tf
from tensorflow.keras import layers, models

from pathlib import Path

MODEL_PATH = './models/'


class Model:
    def __init__(self):
        self.model_dir = './models/'
        self.model = None
        self.prediction_model = None
        self.class_names = []

    def create(self, image_width, image_height):
        self.model = models.Sequential()

        self.model.add(
            layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(image_height, image_width, 3)))
        self.model.add(layers.MaxPooling2D())
        self.model.add(layers.Dropout(0.2))

        self.model.add(layers.Conv2D(32, 3, padding='same', activation='relu'))
        self.model.add(layers.MaxPooling2D())

        self.model.add(layers.Conv2D(64, 3, padding='same', activation='relu'))
        self.model.add(layers.MaxPooling2D())
        self.model.add(layers.Dropout(0.2))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(512, activation='relu'))
        self.model.add(layers.Dense(2))

    def compile(self):
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

    def train(self, train_images, train_labels, steps_per_epoch, epochs, val_images, val_labels, validation_steps,
              batch_size):
        return self.model.fit(
            train_images,
            train_labels,
            epochs=epochs,
            validation_data=(val_images, val_labels),
            validation_steps=validation_steps
        )

    def save(self, name):
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        self.model.save(self.model_dir + name)

    def load_previous_model(self, name):
        try:
            self.model = tf.keras.models.load_model(self.model_dir + name)
        except:
            self.model = None

    def predictions(self, images):
        if self.prediction_model is None:
            self.prediction_model = tf.keras.Sequential([
                self.model,
                tf.keras.layers.Softmax()
            ])
        return self.prediction_model.predict(images)
