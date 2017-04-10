import keras
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator, center_crop
from keras.layers import Flatten, Dense, Input, Dropout, Activation, merge
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from convnetskeras.convnets import convnet
import pickle

import numpy as np
import pdb

def video_2tower_model(weights=None, alexnet_layer="convpool_5"):
        alexnet_model = convnet('alexnet', weights_path="/home/wangnxr/Documents/ecogdeep/convnets-keras/examples/alexnet_weights.h5")
        base_model = Model(alexnet_model.input, alexnet_model.get_layer(alexnet_layer).output)

        frame_a = Input(shape=(3,227,227))
        frame_d = Input(shape=(3,227,227))


        tower1 = base_model(frame_a)
        tower4 = base_model(frame_d)
        x = merge([tower1, tower4], mode='concat', concat_axis=-1, name="concat")
        x = Dropout(0.5)(x)
        x = Dense(1024, W_regularizer=l2(0.01), name='fc2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, W_regularizer=l2(0.01), name='fc3')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(1, name='predictions')(x)
        #x = BatchNormalization()(x)
        predictions = Activation('sigmoid')(x)

        model = Model(input=[frame_a, frame_d], output=predictions)
        if weights is not None:
            model.load_weights(weights)
        return model



