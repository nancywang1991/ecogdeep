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

def vid_model(weights=None, channels=None):

    input_tensor = Input(shape=( 1, 10, 224, 224 ))
    # Block 1
    x = TimeDistributed(Convolution2D(96, 10, 11, 11, border_mode='same', name='block1_conv1', subsample=(4,4)))(input_tensor)
    # x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = TimeDistributed(MaxPooling2D((3, 3), strides=(2,2),name='block1_pool'))(x)

    # Block 2
    x = TimeDistributed(Convolution2D(256, 10, 5, 5, border_mode='same', name='block2_conv1'))(input_tensor)
    # x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = TimeDistributed(MaxPooling2D((3, 3), strides=(2,2),name='block2_pool'))(x)
    # Block 3
    x = TimeDistributed(Convolution2D(384, 10, 3, 3, border_mode='same', name='block3_conv1'))(input_tensor)
    # x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    # Block 3
    x = TimeDistributed(Convolution2D(384, 10, 3, 3, border_mode='same', name='block4_conv1'))(input_tensor)
    # x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    # Block 4
    x = TimeDistributed(Convolution2D(256, 10, 3, 3, border_mode='same', name='block5_conv1'))(input_tensor)
    # x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = TimeDistributed(MaxPooling2D((3, 3), strides=(2, 2), name='block5_pool'))(x)

    x = TimeDistributed(Flatten(name='flatten'))(x)
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
    # x = BatchNormalization()(x)
    predictions = Activation('sigmoid')(x)

    # for layer in base_model.layers[:10]:
    #    layer.trainable = False
    model = Model(input=input_tensor, output=predictions)
    if weights is not None:
        model.load_weights(weights)

    return model
