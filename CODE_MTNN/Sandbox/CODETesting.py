import matplotlib.pyplot as plt
import numpy as np
import os
#import tensorflow as tf
from PIL import Image
import tensorflowjs as tfjs

from tensorflow import keras
from keras import layers

test_model = keras.models.load_model("models/trainedVGG16.keras")
tfjs.converters.save_keras_model(test_model, "models/trainedVGG16JS")
image = np.array(Image.open("test_images/Pano3.jpg").resize((640, 320)))
plt.imshow(image)
image = image[np.newaxis, :, :, :]
predictions = test_model.predict(image)
print(predictions)