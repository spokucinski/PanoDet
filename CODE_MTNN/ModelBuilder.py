from typing import Tuple
import tensorflow as tf
from tensorflow import keras
from keras import layers

class ModelBuilder:

    @staticmethod
    def build_conv_base(model: str, 
                        weights: str, 
                        include_top: bool,
                        input_shape: Tuple[int, int, int],
                        inputs) -> keras.Model:
        
        conv_base: keras.Model
        match model:
            case "vgg16":
                return keras.applications.vgg16.VGG16(weights=weights, 
                                                      include_top=include_top, 
                                                      input_shape=input_shape)
            case "vgg19":
                return keras.applications.vgg19.VGG19(weights=weights, 
                                                      include_top=include_top, 
                                                      input_shape=input_shape)
            case "efficientNetV2L":
                return keras.applications.EfficientNetV2L(weights=weights,
                                                          include_top=include_top,
                                                          input_shape=input_shape)
            
            case "ConvNeXtXLarge":
                return keras.applications.ConvNeXtXLarge(weights=weights,
                                                        include_top=include_top,
                                                        input_shape=input_shape)

            case "custom":
                result = layers.Rescaling(1./255)(inputs)
                result = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(inputs)
                result = layers.MaxPooling2D(pool_size=2)(inputs)
                result = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(inputs)
                result = layers.MaxPooling2D(pool_size=2)(inputs)
                result = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(inputs)
                return result
            
    @staticmethod
    def preprocess_conv_base_input(model: str,
                                   input):

        match model:
            case "vgg16":
                return keras.applications.vgg16.preprocess_input(input)
            
            case "vgg19":
                return keras.applications.vgg19.preprocess_input(input)
            
            case "efficientNetV2L":
                return keras.applications.efficientnet_v2.preprocess_input(input)