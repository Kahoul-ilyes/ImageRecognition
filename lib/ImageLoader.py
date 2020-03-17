from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np

AUTOTUNE = tf.data.experimental.AUTOTUNE
import pathlib

BATCH_SIZE = 128


class ImageLoader:
    def __init__(self):
        self.class_names = None
        self.steps_per_epoch = None
        self.validation_steps = None

    def generate_train_data(self, data_path, image_width, image_height, class_names):
        data_generator = ImageDataGenerator(rescale=1. / 255)

        return next(data_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                       directory=data_path,
                                                       shuffle=True,
                                                       target_size=(image_height, image_width),
                                                       classes=class_names,
                                                       class_mode='sparse'))

    def generate_validation_data(self, data_path, image_width, image_height, class_names):
        data_generator = ImageDataGenerator(rescale=1. / 255)

        return next(data_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                       directory=data_path,
                                                       target_size=(image_height, image_width),
                                                       classes=class_names,
                                                       class_mode='sparse'))

    def load_from_url(self, url, dir_name, image_width, image_height):
        data_dir = tf.keras.utils.get_file(
            origin=url,
            fname=dir_name, untar=True)
        data_dir = pathlib.Path(data_dir)
        train_dir = data_dir.joinpath('train')
        validation_dir = data_dir.joinpath('validation')
        self.class_names = list(
            [item.name for item in train_dir.glob('*') if item.name != "LICENSE.txt" and item.name != ".DS_Store"])

        train_image_count = len(list(train_dir.glob('*/*.jpg')))
        self.steps_per_epoch = int(np.ceil(train_image_count / BATCH_SIZE))

        val_image_count = len(list(validation_dir.glob('*/*.jpg')))
        self.validation_steps = int(np.ceil(val_image_count / BATCH_SIZE))

        return self.generate_train_data(train_dir, image_width, image_height,
                                        self.class_names), self.generate_validation_data(validation_dir, image_width,
                                                                                         image_height, self.class_names)
