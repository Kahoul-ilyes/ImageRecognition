from __future__ import absolute_import, division, print_function, unicode_literals

import os

import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE
import pathlib


class ImageLoader:
    def __init__(self, train_dir, validation_dir):
        self.class_names = None
        self.validation_steps = None
        self.image_width = None
        self.image_height = None
        self.train_dir = train_dir
        self.validation_dir = validation_dir

    def get_label(self, file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
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

    def process_path_with_data_augmentation(self, file_path):
        label = self.get_label(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)

        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)

        return img, label

    def generate_training_dataset(self, path, batch_size, shuffle_buffer_size=1000):
        list_ds = tf.data.Dataset.list_files(str(path / '*/*.jpeg'))

        ds = list_ds.map(self.process_path_with_data_augmentation, num_parallel_calls=AUTOTUNE)

        ds = ds.shuffle(buffer_size=shuffle_buffer_size)

        # Repeat forever
        ds = ds.repeat()

        ds = ds.batch(batch_size)

        # `prefetch` lets the dataset fetch batches in the background while the model
        # is training.
        ds = ds.prefetch(buffer_size=AUTOTUNE)

        return next(iter(ds))

    def generate_validation_dataset(self, path, batch_size):
        list_ds = tf.data.Dataset.list_files(str(path / '*/*.jpeg'))
        ds = list_ds.map(self.process_path, num_parallel_calls=AUTOTUNE)

        # Repeat forever
        ds = ds.repeat()

        ds = ds.batch(batch_size)

        return next(iter(ds))

    def load_from_directory(self, data_dir, image_width, image_height, batch_size):
        data_dir = pathlib.Path(data_dir)
        full_train_dir = data_dir.joinpath(self.train_dir)
        full_validation_dir = data_dir.joinpath(self.validation_dir)
        self.image_width = image_width
        self.image_height = image_height
        self.class_names = list(
            [item.name for item in full_train_dir.glob('*') if item.name != "LICENSE.txt" and item.name[0] != '.'])

        return self.generate_training_dataset(full_train_dir, batch_size), self.generate_validation_dataset(
            full_validation_dir, batch_size)

    def load_from_url(self, url, dir_name, image_width, image_height, batch_size):
        data_dir = tf.keras.utils.get_file(
            origin=url,
            fname=dir_name, untar=True)
        return self.load_from_directory(data_dir, image_width, image_height, batch_size)
