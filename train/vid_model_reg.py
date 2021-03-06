import keras
from keras.regularizers import l2
from keras.preprocessing.ecog import EcogDataGenerator
from keras.layers import Flatten, Dense, Input, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, TimeDistributed

import numpy as np
import pdb
import pickle

def vid_model(weights=None):

    input_tensor = Input(shape=(3, 224, 224 ))
    # Block 1
    x = Convolution2D(128, 5, 5, border_mode='same', name='block1_conv1')(input_tensor)
    # x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2),name='block1_pool')(x)

    # Block 2
    x = Convolution2D(128, 5, 5, border_mode='same', name='block2_conv1')(x)
    # x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2),name='block2_pool')(x)
     # Block 3
    x = Convolution2D(128, 5, 5, border_mode='same', name='block3_conv1')(x)
     # x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    # Block 4
    x = Convolution2D(256, 5, 5, border_mode='same', name='block4_conv1')(x)
    # x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    # Block 5
    x = Convolution2D(512, 9, 9, border_mode='same', name='block5_conv1')(x)
    # x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    # Block 6
    x = Convolution2D(256, 1, 1, border_mode='same', name='block6_conv1')(x)
    # x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    # Block 7
    x = Convolution2D(256, 1, 1, border_mode='same', name='block7_conv1')(x)
    # x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)

    # Block 8
    x = Convolution2D(1, 1, 1, border_mode='same', name='block8_conv1')(x)
    # x = BatchNormalization(axis=1)(x)
    #x = Activation('relu')(x)

    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, W_regularizer=l2(0.01), name='fc1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, W_regularizer=l2(0.01), name='fc2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, name='predictions')(x)
    x = BatchNormalization()(x)
    predictions = Activation('sigmoid')(x)

    # for layer in base_model.layers[:10]:
    #    layer.trainable = False
    model = Model(input=input_tensor, output=predictions)
    if weights is not None:
        model.load_weights(weights)

    return model
