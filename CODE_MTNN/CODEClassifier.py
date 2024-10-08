import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import wandb

from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
from tensorflow import keras
from keras import layers
from ModelBuilder import ModelBuilder
from Visualizer import Visualizer
from datetime import datetime

os.chdir('CODE_MTNN')

MODELS = ['vgg19'] #['vgg16', 'vgg19', 'NASNetLarge', 'EfficientNetB7', 'EfficientNetV2L', 'ConvNeXtXLarge', 'MobileNetV2']
WEIGHTS = 'imagenet'

BATCH_SIZE : int = 8
NUM_CLASSES : int = 5
EPOCHS : int = 25

TRAIN_DATASET_PATH = 'data\\CODE\\train'
VAL_DATASET_PATH = 'data\\CODE\\val'
TEST_DATASET_PATH = 'data\\CODE\\test'

NUM_CHANNELS = (3,)
IMG_SIZE = (960, 1920)
HALF_IMG_SIZE = (480, 960)
THIRD_IMG_SIZE = (320, 640)
IMG_SIZE = IMG_SIZE
IMG_SIZE_DIM = IMG_SIZE + NUM_CHANNELS

USE_DATA_AUG : bool = False
SHOW_DATASET_PREVIEW = False
VISUALIZE_RESULTS : bool = False

for MODEL in MODELS:
    exp_name = f'{MODEL}_{WEIGHTS}_{EPOCHS}_{BATCH_SIZE}_{datetime.utcnow().isoformat()}'
    wandb.init(project="CODE_5C_Classifier",
            name=exp_name
            )

    train_dataset : tf.data.Dataset
    train_dataset = tf.keras.utils.image_dataset_from_directory(TRAIN_DATASET_PATH, 
                                                                batch_size=BATCH_SIZE, 
                                                                image_size=IMG_SIZE, 
                                                                shuffle=True)

    val_dataset : tf.data.Dataset
    val_dataset = tf.keras.utils.image_dataset_from_directory(VAL_DATASET_PATH, 
                                                            batch_size=BATCH_SIZE, 
                                                            image_size=IMG_SIZE, 
                                                            shuffle=True)

    test_dataset : tf.data.Dataset
    test_dataset = tf.keras.utils.image_dataset_from_directory(TEST_DATASET_PATH, 
                                                            batch_size=BATCH_SIZE, 
                                                            image_size=IMG_SIZE, 
                                                            shuffle=True)

    if SHOW_DATASET_PREVIEW :
        print(f"Classes found in the train_dataset: {train_dataset.class_names}")
        Visualizer.present_dataset(train_dataset)

        print(f"Classes found in the val_dataset: {val_dataset.class_names}")
        Visualizer.present_dataset(val_dataset)

        print(f"Classes found in the test_dataset: {test_dataset.class_names}")
        Visualizer.present_dataset(test_dataset)

    inputs = keras.Input(shape=IMG_SIZE_DIM)
    x = inputs
    if USE_DATA_AUG:
        data_augmentation = keras.Sequential(
        [
            layers.RandomTranslation(height_factor=0, 
                                     width_factor=1, 
                                     fill_mode="wrap"),
            #layers.RandomContrast(0.25),
            #tf.keras.layers.experimental.preprocessing.RandomBrightness(0.25)
        ])
        Visualizer.present_data_augmentation(data_augmentation, train_dataset)
        x = data_augmentation(inputs)

    x = ModelBuilder.preprocess_conv_base_input(model=MODEL,
                                                input=x)

    conv_base = ModelBuilder.build_conv_base(model=MODEL,
                                            weights=WEIGHTS, 
                                            include_top=False, 
                                            input_shape=IMG_SIZE_DIM,
                                            inputs=x)
    conv_base.trainable = False
    conv_base.summary()
    x = conv_base(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.summary()

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    callbacks = [
                    keras.callbacks.ModelCheckpoint(filepath=f"models/{MODEL}", 
                                                    save_best_only=True, 
                                                    monitor="val_loss"),
                    keras.callbacks.TensorBoard(log_dir="logs"),
                    keras.callbacks.EarlyStopping(monitor='loss', patience=3),
                    WandbMetricsLogger(),
                    WandbModelCheckpoint(filepath="wandbModels", 
                                         monitor="val_accuracy", 
                                         save_freq="epoch")
                ]

    history = model.fit(train_dataset, 
                        epochs=EPOCHS, 
                        batch_size=BATCH_SIZE, 
                        validation_data=val_dataset, 
                        callbacks=callbacks)



    if VISUALIZE_RESULTS:
        Visualizer.present_training_history(history=history)

    # get the labels 
    predictions = np.array([])
    labels =  np.array([])
    for x, y in val_dataset:
        predictions = np.concatenate([predictions, np.argmax(model.predict(x), axis=-1)])
        labels = np.concatenate([labels, y.numpy()])

    predictions[:10]
    #array([0., 4., 3., 0., 3., 4., 2., 4., 4., 0.])

    labels[:10]
    #array([0., 4., 3., 0., 3., 4., 1., 2., 4., 0.])
    m = tf.keras.metrics.Accuracy()
    m(labels, predictions).numpy()


    test_loss, test_acc = model.evaluate(test_dataset)
    print(f"Test accuracy: {test_acc:.3f}")

    print("Closing the app!")

    wandb.finish()