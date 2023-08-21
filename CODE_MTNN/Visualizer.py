import matplotlib.pyplot as plt
import tensorflow as tf

class Visualizer:
    def __init__(self) -> None:
        pass

    @staticmethod
    def present_dataset(dataset: tf.data.Dataset):
        class_names = dataset.class_names
        plt.figure(figsize=(10, 10))
        for images, labels in dataset.take(1):
            for i in range(9):
                plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(class_names[labels[i]])
                plt.axis("off")

    @staticmethod
    def present_data_augmentation(augmentation: tf.keras.Sequential,
                                  dataset: tf.data.Dataset):
        for image, _ in dataset.take(1):
            plt.figure(figsize=(10, 10))
            first_image = image[0]
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                augmented_image = augmentation(tf.expand_dims(first_image, 0))
                plt.imshow(augmented_image[0] / 255)
                plt.axis('off')

    @staticmethod
    def present_training_history(history: tf.keras.callbacks.History):
        acc = history.history["accuracy"]
        val_acc = history.history["val_accuracy"]
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        epochs = range(1, len(acc)+1)
        plt.plot(epochs, acc, "bo", label="Training accuracy")
        plt.plot(epochs, val_acc, "b", label="Validation accuracy")
        plt.title("Training and validation accuracy")
        plt.legend()
        plt.figure()
        plt.plot(epochs, loss, "bo", label="Training loss")
        plt.plot(epochs, val_loss, "b", label="Validation loss")
        plt.title("Training and validation loss")
        plt.legend()
        plt.show()