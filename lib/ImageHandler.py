from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os

BATCH_SIZE = 128


def _generate_data(data_path, image_width, image_height):
    data_generator = ImageDataGenerator(rescale=1. / 255)
    return next(data_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                   directory=data_path,
                                                   shuffle=True,
                                                   target_size=(image_height, image_width),
                                                   class_mode='binary'))


class ImageHandler:

    @staticmethod
    def load_from_url(url, dir_name, image_width, image_height):
        path_to_zip = tf.keras.utils.get_file(dir_name + '.zip', origin=url, extract=True)
        data_path = os.path.join(os.path.dirname(path_to_zip), dir_name)
        train_dir = os.path.join(data_path, 'train')
        validation_dir = os.path.join(data_path, 'validation')

        return _generate_data(train_dir, image_width, image_height), _generate_data(validation_dir, image_width,
                                                                                    image_height)
