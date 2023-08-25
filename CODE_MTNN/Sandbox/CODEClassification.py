import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from Visualizer import Visualizer

TRAIN_DATASET_PATH = 'data\\CODE\\train'
VAL_DATASET_PATH = 'data\\CODE\\val'
TEST_DATASET_PATH = 'data\\CODE\\test'

# IMG_SIZE = (1920, 960)
IMG_SIZE = (224, 224)
BATCH_SIZE : int = 32
NUM_CLASSES : int = 17
IMG_SHAPE = IMG_SIZE + (3,)

train_dataset : tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(TRAIN_DATASET_PATH, 
                                                                              batch_size=BATCH_SIZE, 
                                                                              image_size=IMG_SIZE, 
                                                                              shuffle=True)

val_dataset : tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(VAL_DATASET_PATH, 
                                                                            batch_size=BATCH_SIZE, 
                                                                            image_size=IMG_SIZE, 
                                                                            shuffle=True)

test_dataset : tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(TEST_DATASET_PATH, 
                                                                             batch_size=BATCH_SIZE, 
                                                                             image_size=IMG_SIZE, 
                                                                             shuffle=True)

print(f"Classes found in the train_dataset: {train_dataset.class_names}")
Visualizer.present_dataset(train_dataset)

print(f"Classes found in the val_dataset: {val_dataset.class_names}")
Visualizer.present_dataset(val_dataset)

print(f"Classes found in the test_dataset: {test_dataset.class_names}")
Visualizer.present_dataset(test_dataset)

# AUTOTUNE = tf.data.AUTOTUNE

# train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
# val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)
# test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal')
])

Visualizer.present_data_augmentation(data_augmentation, train_dataset)

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
# Create the base model from the pre-trained model MobileNet V2

base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
print(feature_batch.shape)

base_model.trainable = False

base_model.summary()

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

prediction_layer = tf.keras.layers.Dense(17, activation="softmax")
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

inputs = tf.keras.Input(shape=(IMG_SHAPE))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[tf.keras.metrics.CategoricalAccuracy])

model.summary()

initial_epochs = 10

loss0, accuracy0 = model.evaluate(val_dataset)

print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

history = model.fit(train_dataset,
                    epochs=initial_epochs,
                    validation_data=val_dataset)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

print("Closing the app!")