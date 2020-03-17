from __future__ import absolute_import, division, print_function, unicode_literals

from lib.ImageLoader import ImageLoader
from lib.Model import Model
from lib.Display import Display
import numpy as np

EPOCHS = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150
CLASS_NAME = ['cat', 'dog']

if __name__ == '__main__':
    imageHandler = ImageLoader()
    (train_images, train_labels), (val_images, val_labels) = imageHandler.load_from_url(
        'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip',
        'cats_and_dogs_filtered',
        IMG_WIDTH,
        IMG_HEIGHT
    )
    model = Model(imageHandler.class_names)

    model.load_previous_model('main.model')

    if model.model is None:
        model.create(IMG_WIDTH, IMG_HEIGHT)
        model.compile()
        history = model.train(
            train_images,
            train_labels,
            imageHandler.steps_per_epoch,
            EPOCHS,
            val_images,
            val_labels,
            imageHandler.validation_steps
        )
        model.save('main.model')
        Display.print_graphic(
            range(EPOCHS),
            history.history['accuracy'],
            history.history['val_accuracy'],
            history.history['loss'],
            history.history['val_loss'],
        )

# Display.plot_images_predictions(3, 5, predictions, train_labels, train_images, model.class_names)

# model.save('main.model')

# if model.model:
#     print(model.model)
# else:
#     model.create(IMG_WIDTH, IMG_HEIGHT)
#     model.compile()
#     history = model.train(
#         train_images,
#         train_labels,
#         EPOCHS,
#         val_images,
#         val_labels
#     )
#     model.save('main.model')

# print(list(model.predict(val_images[:5])))
# Display.plot_images(val_images[:5])
