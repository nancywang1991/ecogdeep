import keras
from keras.regularizers import l2
from keras.preprocessing.ecog import EcogDataGenerator
from keras.layers import Flatten, Dense, Input, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model, load_model
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D

import numpy as np
import pdb
import pickle
import gc
## Data generation
train_datagen = EcogDataGenerator(
        time_shift_range=200,
        gaussian_noise_range=0.01,
        center=False
        #f_lo=2,
        #f_hi=22,
        #fft=True

)

test_datagen = EcogDataGenerator(
        center=True
        #f_lo=2,
        #f_hi=22,
        #fft=True

)

dgdx = train_datagen.flow_from_directory(
        #'/mnt/cb46fd46_5_no_offset/train/',
        '/home/wangnxr/dataset/BCIcompData/train/',
        batch_size=25,
        target_size=(1,64,1000),
	#final_size=(1,64,20),
        class_mode='binary')
dgdx_val = test_datagen.flow_from_directory(
        #'/mnt/cb46fd46_5_no_offset/test/',
        '/home/wangnxr/dataset/BCIcompData/val/',
        batch_size=10,
        shuffle=False,
        target_size=(1,64,1000),
	#final_size=(1,64,20),
        class_mode='binary')

train_generator=dgdx
validation_generator=dgdx_val

## Hyperparameter optimization space


def f_nn(params):
    # Determine proper input shape
    pretrain_model = load_model("/home/wangnxr/model_ecog_1d_1_3_1_2_mj_a0f.h5" )
    base_model = Model(pretrain_model.input, pretrain_model.get_layer("block%i_pool" % params).output)
    base_model = Model(pretrain_model.input, pretrain_model.get_layer("fc1").output)
    input_tensor=Input(shape=(1,64,1000))
    x = base_model(input_tensor) 
    # Block 1
    #x = AveragePooling2D((1,5),  name='pre_pool')(input_tensor)
    #x = Convolution2D(16, 1, 3, border_mode='same', name='block1_conv1')(x)
    #x = Activation('relu')(x)
    #x = MaxPooling2D((1,2),  name='block1_pool')(x)

    # Block 2
    #x = Convolution2D(32, 1, 3,  border_mode='same', name='block2_conv1')(x)
    #x = Activation('relu')(x)
    #x = MaxPooling2D((1,2),  name='block2_pool')(x)

    # Block 3
    #x = Convolution2D(64, 1, 3, border_mode='same', name='block3_conv1')(x)
    #x = Activation('relu')(x)
    #x = MaxPooling2D((1,2),  name='block3_pool')(x)

    # Block 4
    #x = Convolution2D(128, 1, 3, border_mode='same', name='block4_conv1')(x)
    #x = BatchNormalization(axis=1)(x)
    #x = Activation('relu')(x)
    #x = MaxPooling2D((1,2), name='block4_pool')(x)


    #x = Flatten(name='flatten')(x)
    #x = Dropout(0.5)(x)
    #x = Dense(1024, W_regularizer=l2(0.01), name='fc1')(x)
    #x = BatchNormalization()(x)
    #x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(16, W_regularizer=l2(0.01), name='fc2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, name='predictions')(x)
    predictions = Activation('sigmoid')(x)

    #for layer in base_model.layers:
    #	layer.trainable = False

    model = Model(input=input_tensor, output=predictions)
    #print model.get_weights()[0]
    sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9)

    model.compile(optimizer=sgd,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])



    history_callback = model.fit_generator(
            train_generator,
            samples_per_epoch=238,
            nb_epoch=100,
            validation_data=validation_generator,
            nb_val_samples=40)
    pdb.set_trace()
    model.save("/home/wangnxr/BCIcomp_results/model_ecog_1d_imagined_block_dense1_a0f_no_freeze.h5" )
    pickle.dump(history_callback.history,open("/home/wangnxr/BCIcomp_results/history_ecog_1d_imagined_block_dense1_a0f_no_freeze.p", "wb"))

    loss = history_callback.history["val_loss"][-1]
    gc.collect()
    return loss

params_list = [3]
losses = []
for params in params_list:
    losses.append(f_nn(params))
gc.collect()
