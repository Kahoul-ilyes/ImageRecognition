from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt

from lib.ImageHandler import ImageHandler
from lib.Model import Model
from lib.Display import Display

EPOCHS = 16
IMG_HEIGHT = 150
IMG_WIDTH = 150

if __name__ == '__main__':
    (train_images, train_labels), (val_images, val_labels) = ImageHandler.load_from_url(
        'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip',
        'cats_and_dogs_filtered',
        IMG_WIDTH,
        IMG_HEIGHT
    )

    model = Model()

    model.load_previous_model('main.model')

    if model.model:
        print(model.model)
    else:
        model.create(IMG_WIDTH, IMG_HEIGHT)
        model.compile()
        history = model.train(
            train_images,
            train_labels,
            EPOCHS,
            val_images,
            val_labels
        )
        model.save('main.model')

        Display.print_graphic(
            range(EPOCHS),
            history.history['accuracy'],
            history.history['val_accuracy'],
            history.history['loss'],
            history.history['val_loss'],
        )

