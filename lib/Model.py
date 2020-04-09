import tensorflow as tf
from tensorflow.keras import layers, models

from pathlib import Path

from tensorflow_core.python.keras import regularizers

MODEL_PATH = './models/'


class Model:
    def __init__(self, class_names):
        self.model_dir = './models/'
        self.model = None
        self.prediction_model = None
        self.class_names = class_names

    def create(self, image_width, image_height):
        self.model = models.Sequential()

        self.model.add(
            layers.Conv2D(16, 3, kernel_regularizer=regularizers.l2(0.0001),
                          padding='same', activation='relu', input_shape=(image_height, image_width, 3)))
        self.model.add(layers.MaxPooling2D())
        self.model.add(layers.Dropout(0.3))

        self.model.add(layers.Conv2D(32, 3, kernel_regularizer=regularizers.l2(0.0001),
                                     padding='same', activation='relu'))
        self.model.add(layers.MaxPooling2D())

        self.model.add(layers.Conv2D(64, 3, kernel_regularizer=regularizers.l2(0.0001),
                                     padding='same', activation='relu'))
        self.model.add(layers.MaxPooling2D())
        self.model.add(layers.Dropout(0.3))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                                    activation='relu'))
        self.model.add(layers.Dense(len(self.class_names)))

    def compile(self):
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

    def train(self, train_images, train_labels, epochs, val_images, val_labels, validation_steps):
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
        return self.model.predict(images)
