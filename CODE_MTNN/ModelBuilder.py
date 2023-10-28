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
        if model == "vgg16":
            return keras.applications.vgg16.VGG16(weights=weights, 
                                                    include_top=include_top, 
                                                    input_shape=input_shape)
        elif model == "vgg19":
            return keras.applications.vgg19.VGG19(weights=weights, 
                                                    include_top=include_top, 
                                                    input_shape=input_shape)
        
        elif model == "NASNetLarge":
            return keras.applications.NASNetLarge(weights=weights,
                                                    include_top=include_top,
                                                    input_shape=input_shape)
        
        elif model == "EfficientNetB7":
            return keras.applications.EfficientNetB7(weights=weights,
                                                        include_top=include_top,
                                                        input_shape=input_shape)

        elif model == "EfficientNetV2L":
            return keras.applications.EfficientNetV2L(weights=weights,
                                                        include_top=include_top,
                                                        input_shape=input_shape)
        
        elif model == "ConvNeXtXLarge":
            return keras.applications.ConvNeXtXLarge(weights=weights,
                                                    include_top=include_top,
                                                    input_shape=input_shape)
        
        elif model == "MobileNetV2":
            return keras.applications.MobileNetV2(weights=weights,
                                                    include_top=include_top,
                                                    input_shape=input_shape)

        elif model == "custom":
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

        if model == "vgg16":
            return keras.applications.vgg16.preprocess_input(input)
            
        elif model == "vgg19":
            return keras.applications.vgg19.preprocess_input(input)
        
        elif model == "NASNetLarge":
            return keras.applications.nasnet.preprocess_input(input)
        
        elif model == "EfficientNetV2L":
            return keras.applications.efficientnet_v2.preprocess_input(input)
        
        elif model == "EfficientNetB7":
            return keras.applications.efficientnet.preprocess_input(input)
        
        elif model == "ConvNeXtXLarge":
            return input
        
        elif model == "MobileNetV2":
            return keras.applications.mobilenet_v2.preprocess_input(input)