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
"""ECoG 1 dimensional (time) sequence model with spatial convolution and pooling. This has not been tested yet!!!

"""
def ecog_1d_model(weights=None, channels=None):

    input_tensor = Input(shape=( 5, 1, channels, 200))
    # Block 1
    x = TimeDistributed(MaxPooling2D((1, 5)), name='pre_pool')(input_tensor)
    x = TimeDistributed(Convolution2D(4, 1, 3, border_mode='same'), name='block1_conv1')(x)
    #x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = TimeDistributed(MaxPooling2D((1, 3)), name='block1_pool')(x)

    x = TimeDistributed(Convolution2D(4, 2, 1, border_mode='same'), name='block1_conv1_spa')(x)
    #x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = TimeDistributed(MaxPooling2D((2, 1)), name='block1_pool')(x)

    # Block 2
    x = TimeDistributed(Convolution2D(8, 1, 3, border_mode='same'), name='block2_conv1')(x)
    #x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = TimeDistributed(MaxPooling2D((1, 3)), name='block2_pool')(x)

    x = TimeDistributed(Convolution2D(8, 2, 1, border_mode='same'), name='block2_conv1_spa')(x)
    #x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = TimeDistributed(MaxPooling2D((2, 1)), name='block1_pool')(x)

    # Block 3
    x = TimeDistributed(Convolution2D(16, 1, 3, border_mode='same'), name='block3_conv1')(x)
    #x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = TimeDistributed(MaxPooling2D((1, 3)), name='block3_pool')(x)

    x = TimeDistributed(Convolution2D(16, 2, 1, border_mode='same'), name='block3_conv1_spa')(x)
    #x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = TimeDistributed(MaxPooling2D((2, 1)), name='block1_pool')(x)

    # Block 4
    #x = TimeDistributed(Convolution2D(32, 1, 3, border_mode='same'), name='block4_conv1')(x)
    #x = BatchNormalization(axis=1)(x)
    #x = Activation('relu')(x)
    #x = TimeDistributed(MaxPooling2D((1, 3)), name='block4_pool')(x)

    x = TimeDistributed(Flatten(), name='flatten')(x)
    x = Dropout(0.5)(x)
    x = TimeDistributed(Dense(64, W_regularizer=l2(0.01)), name='fc1')(x)
    #x = BatchNormalization()(x)
    #x = Activation('relu')(x)
    #x = Dropout(0.5)(x)
    #x = Dense(256, W_regularizer=l2(0.01), name='fc2')(x)
    #x = BatchNormalization()(x)
    #x = Activation('relu')(x)
    #x = Dropout(0.5)(x)
    #x = Dense(1, name='predictions')(x)
    # x = BatchNormalization()(x)
    predictions = Activation('sigmoid')(x)

    # for layer in base_model.layers[:10]:
    #    layer.trainable = False
    model = Model(input=input_tensor, output=predictions)
    if weights is not None:
        model.load_weights(weights)

    return model
