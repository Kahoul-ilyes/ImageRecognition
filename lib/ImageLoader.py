from __future__ import absolute_import, division, print_function, unicode_literals

import os

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np

AUTOTUNE = tf.data.experimental.AUTOTUNE
import pathlib


class ImageLoader:
    def __init__(self):
        self.class_names = None
        self.steps_per_epoch = None
        self.validation_steps = None
        self.image_width = None
        self.image_height = None

    def get_label(self, file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        print(parts[-2])
        return parts[-2] == self.class_names

    def decode_img(self, img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
        return tf.image.resize(img, [self.image_width, self.image_height])

    def process_path(self, file_path):
        label = self.get_label(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)
        return img, label

    def generate_training_dataset(self, path, batch_size, shuffle_buffer_size=1000):
        list_ds = tf.data.Dataset.list_files(str(path / '*/*'))
        ds = list_ds.map(self.process_path, num_parallel_calls=AUTOTUNE)

        ds = ds.shuffle(buffer_size=shuffle_buffer_size)

        # Repeat forever
        ds = ds.repeat()

        ds = ds.batch(batch_size)

        # `prefetch` lets the dataset fetch batches in the background while the model
        # is training.
        ds = ds.prefetch(buffer_size=AUTOTUNE)

        return next(iter(ds))

    def generate_validation_dataset(self, path, batch_size, shuffle_buffer_size=1000):
        list_ds = tf.data.Dataset.list_files(str(path / '*/*'))
        ds = list_ds.map(self.process_path, num_parallel_calls=AUTOTUNE)

        # Repeat forever
        ds = ds.repeat()

        ds = ds.batch(batch_size)

        return next(iter(ds))

    def load_from_url(self, url, dir_name, image_width, image_height, batch_size):
        data_dir = tf.keras.utils.get_file(
            origin=url,
            fname=dir_name, untar=True)
        data_dir = pathlib.Path(data_dir)
        train_dir = data_dir.joinpath('train')
        validation_dir = data_dir.joinpath('validation')
        self.image_width = image_width
        self.image_height = image_height
        self.class_names = list(
            [item.name for item in train_dir.glob('*') if item.name != "LICENSE.txt" and item.name != ".DS_Store"])

        return self.generate_training_dataset(train_dir, batch_size), self.generate_validation_dataset(validation_dir, batch_size)
        # train_image_count = len(list(train_dir.glob('*/*.jpg')))
        # self.steps_per_epoch = int(np.floor(train_image_count / batch_size))
        #
        # val_image_count = len(list(validation_dir.glob('*/*.jpg')))
        # self.validation_steps = int(np.floor(val_image_count / batch_size))
        #
        # return self.generate_train_data(train_dir, image_width, image_height,
        #                                 self.class_names, batch_size), self.generate_validation_data(validation_dir,
        #                                                                                              image_width,
        #                                                                                              image_height,
        #                                                                                              self.class_names,
        #                                                                                              batch_size)
