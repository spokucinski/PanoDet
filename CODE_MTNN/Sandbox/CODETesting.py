import matplotlib.pyplot as plt
import numpy as np
import os
#import tensorflow as tf
from PIL import Image
#import tensorflowjs as tfjs

from tensorflow import keras
from keras import layers

test_model = keras.models.load_model("C:\\Users\\Sebastian\\Documents\\PanoDet\\CODE_MTNN\\models\\vgg19")
#tfjs.converters.save_keras_model(test_model, "models/trainedVGG16JS")
image = np.array(Image.open("C:\\Users\\Sebastian\\Documents\\PanoDet\\CODE_MTNN\\test_images\\Pano3.jpg").resize((1920, 960)))
plt.imshow(image)
image = image[np.newaxis, :, :, :]
predictions = test_model.predict(image)
print(predictions)