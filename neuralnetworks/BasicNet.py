# -*- coding: utf-8 -*-
"""
Based on the online guide https://www.pyimagesearch.com/2021/05/22/a-gentle-guide-to-training-your-first-cnn-with-keras-and-tensorflow/
@author: Philippe Savard
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras import backend as K

class BasicNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
        
        model.add(Conv2D(64,(5,5)), padding="same", input_shape=(inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding="valid"))
        
        model.add(Conv2D(128,(3,3)), padding="same", input_shape=(inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding="valid"))
        
        model.add(Conv2D(512,(3,3)), padding="same", input_shape=(inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding="valid"))
        
        model.add(Conv2D(512,(3,3)), padding="same", input_shape=(inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding="valid"))
        
        model.add(Flatten())
        
        model.add(Dense(256))
        model.add(Activation("relu"))
        
        model.add(Dense(512))
        model.add(Activation("relu"))
        
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        
        return model
