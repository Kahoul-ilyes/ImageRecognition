from __future__ import absolute_import, division, print_function, unicode_literals

from lib.ImageLoader import ImageLoader
from lib.Model import Model
from lib.Display import Display

BATCH_SIZE = 128
EPOCHS = 20
IMG_HEIGHT = 150
IMG_WIDTH = 150

if __name__ == '__main__':
    imageHandler = ImageLoader()
    (train_images, train_labels), (val_images, val_labels) = imageHandler.load_from_url(
        'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip',
        'cats_and_dogs_filtered',
        IMG_WIDTH,
        IMG_HEIGHT,
        BATCH_SIZE
    )
    model = Model()

    model.load_previous_model('main.model')

    if model.model is None:
        model.create(IMG_WIDTH, IMG_HEIGHT)
        model.compile()
        history = model.train(
            train_images,
            train_labels,
            EPOCHS,
            val_images,
            val_labels,
            imageHandler.validation_steps,
        )
        model.save('main.model')
        Display.print_graphic(
            range(EPOCHS),
            history.history['accuracy'],
            history.history['val_accuracy'],
            history.history['loss'],
            history.history['val_loss'],
        )
    predictions = model.predictions(val_images)
    Display.plot_images_predictions(5, 3, predictions, val_labels.numpy(), val_images, imageHandler.class_names)

