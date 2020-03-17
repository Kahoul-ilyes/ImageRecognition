import matplotlib.pyplot as plt
import numpy as np


class Display:

    @staticmethod
    def plot_image(i, predictions_array, true_label, img, class_names):
        predictions_array, true_label, img = predictions_array, true_label[i], img[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        plt.imshow(img, cmap=plt.cm.binary)

        predicted_label = np.argmax(predictions_array)
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'

        plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                             100 * np.max(predictions_array),
                                             class_names[true_label]),
                   color=color)

    @staticmethod
    def plot_value_array(i, predictions_array, true_label):
        predictions_array, true_label = predictions_array, true_label[i]
        plt.grid(False)
        plt.xticks(range(10))
        plt.yticks([])
        thisplot = plt.bar(range(10), predictions_array, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)

        thisplot[predicted_label].set_color('red')
        thisplot[true_label].set_color('blue')

    @staticmethod
    def plot_images_predictions(rows, columns, predictions, labels, images, class_names):

        num_images = rows * columns
        plt.figure(figsize=(2 * 2 * columns, 2 * rows))
        for i in range(num_images):
            plt.subplot(rows, 2 * columns, 2 * i + 1)
            Display.plot_image(i, predictions[i], labels, images, class_names)
            plt.subplot(rows, 2 * columns, 2 * i + 2)
            Display.plot_value_array(i, predictions[i], labels)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def print_graphic(
            epochs_range,
            accuracy,
            validation_accuracy,
            loss,
            validation_loss
    ):
        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, accuracy, label='Training Accuracy')
        plt.plot(epochs_range, validation_accuracy, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, validation_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.savefig('graphic.png')
