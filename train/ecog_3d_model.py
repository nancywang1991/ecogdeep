import keras
from keras.regularizers import l2
from keras.layers import Flatten, Dense, Input, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Convolution3D, MaxPooling3D, AveragePooling3D, TimeDistributed

import numpy as np
import pdb
import pickle
"""ECoG 3 dimensional (time) model with 1 second input.

"""
def ecog_3d_model(channels=None, weights=None):

    input_tensor = Input(shape=(5, 1,10,10, 200))
    # Block 1
    x = TimeDistributed(Convolution3D(4, (1, 1, 3), padding='same', name='block1_conv1'))(input_tensor)
    x = Activation('relu')(x)
    x = TimeDistributed(MaxPooling3D(( 1, 1, 3), name='block1_pool1'))(x)
    x = TimeDistributed(Convolution3D(8, (3, 3, 1), padding='same', name='block1_conv2'))(x)
    x = Activation('relu')(x)
    x = TimeDistributed(MaxPooling3D(( 2, 2, 1), name='block1_pool2'))(x)

    # Block 2


    x = TimeDistributed(Convolution3D(16, (1, 1, 3), padding='same', name='block2_conv1'))(x)
    x = Activation('relu')(x)
    x = TimeDistributed(MaxPooling3D(( 1, 1, 3), name='block2_pool1'))(x)
    x = TimeDistributed(Convolution3D(32, (3, 3, 1), padding='same', name='block2_conv2'))(x)
    x = Activation('relu')(x)

    # Block 3
    x = TimeDistributed(Convolution3D(64, (1, 1, 3), padding='same', name='block3_conv1'))(x)
    x = Activation('relu')(x)
    x = TimeDistributed(MaxPooling3D(( 1, 1, 3), name='block3_pool1'))(x)

    x = TimeDistributed(Flatten(), name='flatten')(x)
    x = Dropout(0.5)(x)
    x = TimeDistributed(Dense(128, kernel_regularizer=l2(0.01)), name='fc1')(x)
    predictions = Activation('sigmoid')(x)

    # for layer in base_model.layers[:10]:
    #    layer.trainable = False
    model = Model(inputs=input_tensor, outputs=predictions)
    if weights is not None:
        model.load_weights(weights)

    return model
