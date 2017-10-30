import keras
from keras.regularizers import l2
from keras.preprocessing.ecog import EcogDataGenerator
from keras.layers import Flatten, Dense, Input, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D

import numpy as np
import pdb
import pickle
"""ECoG 1 dimensional (time) model with 1 second input.

"""
def ecog_1d_model(channels=None, weights=None):

    input_tensor = Input(shape=(1, channels, 1000))
    # Block 1
    x = AveragePooling2D((1, 5), name='pre_pool')(input_tensor)
    x = Convolution2D(32, 1, 3, border_mode='same', name='block1_conv1')(x)
    #x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((1, 3), name='block1_pool')(x)

    # Block 2
    x = Convolution2D(64, 1, 3, border_mode='same', name='block2_conv1')(x)
    #x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((1, 3), name='block2_pool')(x)

    # Block 3
    x = Convolution2D(256, 1, 3, border_mode='same', name='block3_conv1')(x)
#    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((1, 3), name='block3_pool')(x)

    # Block 4
    #x = Convolution2D(512, 1, 3, border_mode='same', name='block4_conv1')(x)
    #x = BatchNormalization(axis=1)(x)
    #x = Activation('relu')(x)
    #x = MaxPooling2D((1, 2), name='block4_pool')(x)

    x = Flatten(name='flatten')(x)
    x = Dropout(0.3)(x)
    x1 = Dense(64, W_regularizer=l2(0.01), name='fc1')(x)
    x2 = Dense(64, W_regularizer=l2(0.01), name='fc2')(x)
    x3 = Dense(64, W_regularizer=l2(0.01), name='fc3')(x)
    #x = BatchNormalization()(x)
    x1 = Activation('relu')(x1)
    x2 = Activation('relu')(x2)
    x3 = Activation('relu')(x3)
    x1 = Dropout(0.3)(x1)
    x2 = Dropout(0.3)(x2)
    x3 = Dropout(0.3)(x3)

#    x = Dense(64, W_regularizer=l2(0.01), name='fc2')(x)
#    x = Activation('relu')(x)
  #  x = Dropout(0.5)(x)
    #x = Dropout(0.5)(x)
    x1 = Dense(1, name='predictions1')(x1)
    x2 = Dense(1, name='predictions2')(x2)
    x3 = Dense(1, name='predictions3')(x3)

    # x = BatchNormalization()(x)
    #predictions = Activation('sigmoid')(x)

    # for layer in base_model.layers[:10]:
    #    layer.trainable = False
    model = Model(input=input_tensor, output=[x1,x2,x3])
    if weights is not None:
        model.load_weights(weights)

    return model
