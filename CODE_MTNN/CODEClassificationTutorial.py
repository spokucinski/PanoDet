# Import dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

(train_raw, val_raw, test_raw), ds_info = tfds.load('oxford_iiit_pet', # name of cars dataset in tfds
                                           split=['train[:90%]', # Perform train/val split (90/10)
                                                  'train[90%:]',
                                                  'test[:10%]'], # Use only 10% of test set for testing
                                           shuffle_files=True, # Shuffle the order of images
                                           as_supervised=True, # Returns (image, label)
                                           with_info=True # To retrieve dataset info and label names
                                           )

# Display examples
tfds.show_examples(train_raw, ds_info)

# Get number of classes
num_classes = ds_info.features['label'].num_classes
print('Number of car classes:', num_classes)


# Get number of train, val, and test examples
num_train_examples = tf.data.experimental.cardinality(train_raw).numpy()
num_val_examples = tf.data.experimental.cardinality(val_raw).numpy()
num_test_examples = tf.data.experimental.cardinality(test_raw).numpy()

print('Number of training samples:', num_train_examples)
print('Number of validation samples:', num_val_examples)
print('Number of test samples:', num_test_examples)

# Get the distribution of class labels (in integers) in the train and validation sets
def get_value_counts(ds):
    label_list = []
    for images, labels in ds: 
        label_list.append(labels.numpy())
        label_counts = pd.Series(label_list).value_counts(sort=True)
    print(label_counts)

get_value_counts(train_raw)
get_value_counts(val_raw)
# Function within the dataset documentation object that converts class integer to label name
get_label_name = ds_info.features['label'].int2str
# Build the custom function to display image and label name
def view_single_image(ds):
    image, label = next(iter(ds))
    print('Image shape: ', image.shape)
    plt.imshow(image)
    _ = plt.title(get_label_name(label))
view_single_image(train_raw)
IMG_SIZE = 224

train_ds = train_raw.map(lambda x, y: (tf.image.resize(x, (IMG_SIZE, IMG_SIZE)), y))
val_ds = val_raw.map(lambda x, y: (tf.image.resize(x, (IMG_SIZE, IMG_SIZE)), y))
test_ds = test_raw.map(lambda x, y: (tf.image.resize(x, (IMG_SIZE, IMG_SIZE)), y))

def one_hot_encode(image, label):
    label = tf.one_hot(label, num_classes)
    return image, label
train_ds = train_ds.map(one_hot_encode)
val_ds = val_ds.map(one_hot_encode)
test_ds = test_ds.map(one_hot_encode)
print(train_ds)

data_augmentation = keras.Sequential(
                [tf.keras.layers.RandomFlip('horizontal'), 
                #  layers.RandomRotation(factor=(-0.025, 0.025)),
                #  layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
                #  layers.RandomContrast(factor=0.1),
                 ])

for image, label in train_ds.take(1): # Iterate and retrieve one set of image and label from the train_ds generator object
    plt.figure(figsize=(10, 10))
    for i in range(6):  # Display six augmented images in a grid of 3 x 2
        ax = plt.subplot(3, 2, i+1)
        aug_img = data_augmentation(tf.expand_dims(image, axis=0))
        plt.imshow(aug_img[0].numpy().astype('uint8')) # Retrieve raw values from augmented image
        plt.title(get_label_name(int(label[0]))) # Using get_label_name function to retrieve label name
        plt.axis("off")

BATCH_SIZE = 64

# Batch the data and use prefetching to optimize loading speed
train_ds = train_ds.batch(batch_size=BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.batch(batch_size=BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.batch(batch_size=BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

base_model = keras.applications.ResNet50V2(
                    include_top=False, # Exclude ImageNet classifier at the top
                    weights='imagenet',
                    input_shape=(IMG_SIZE, IMG_SIZE, 3)
                    )

# Freeze the base_model
base_model.trainable = False

# Setup inputs with shape of our images
inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

# Apply augmentation
x = data_augmentation(inputs)

# Apply the specific pre-processing function for ResNet v2
x = keras.applications.resnet_v2.preprocess_input(x)

# Keep the base model batch normalization layers in inference mode (instead of training mode)
x = base_model(x, training=False)

# Rebuild top layers
x = tf.keras.layers.GlobalAveragePooling2D()(x) # Average pooling operation
x = tf.keras.layers.BatchNormalization()(x) # Introduce batch norm
x = tf.keras.layers.Dropout(0.2)(x)  # Regularize with dropout

# Flattening to final layer - Dense classifier with 196 units (multi-class classification)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

# Instantiate final Keras model with updated inputs and outputs
model = keras.Model(inputs, outputs)

model.summary()

model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.CategoricalCrossentropy(),
              metrics=[keras.metrics.CategoricalAccuracy()]
             )

EPOCHS = 15

history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, verbose=1)

def plot_metric_hist(hist):
    plt.plot(hist.history['categorical_accuracy'])
    plt.plot(hist.history['val_categorical_accuracy'])
    plt.title('Categorical accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

plot_metric_hist(history)

result = model.evaluate(test_ds)

dict(zip(model.metrics_names, result))

for layer in model.layers[-15:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), # Set a very low learning rate
              loss=keras.losses.CategoricalCrossentropy(),
              metrics=[keras.metrics.CategoricalAccuracy()]
             )

EPOCHS = 10

history_2 = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, verbose=1)

plot_metric_hist(history_2)

result = model.evaluate(test_ds)

dict(zip(model.metrics_names, result))