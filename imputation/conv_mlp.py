import keras
from keras import backend as K
from ecogdeep.data.preprocessing.ecog_for_imputation import EcogDataGenerator
from keras.layers import Flatten, Dense, Input, Dropout, Activation, merge, TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D

# from keras.imagenet_utils import decode_predictions, preprocess_input, _obtain_input_shape
import numpy as np
import pdb
import pickle
import glob

sbj_to_do = ["a0f", "d65", "a0f_d65"]
for itr in range(1):
    for s, sbj in enumerate(sbj_to_do):
        main_ecog_dir = '/data2/users/wangnxr/dataset/ecog_mni_%s/' % (sbj)

        ## Data generation ECoG
        channels = np.arange(100)
        train_datagen_edf = EcogDataGenerator(
            seq_len=200
        )

        test_datagen_edf = EcogDataGenerator(
            seq_len=200
        )

        dgdx_edf = train_datagen_edf.flow_from_directory(
            '%s/train/' % main_ecog_dir,
            batch_size=24,
            channels=channels,
            pre_shuffle_ind=1
        )

        dgdx_val_edf = test_datagen_edf.flow_from_directory(
            '%s/val/' % main_ecog_dir,
            batch_size=10,
            channels=channels)
        train_generator = dgdx_edf
        validation_generator = dgdx_val_edf

        ecog_series = Input(shape=(1, len(channels), 200))
        x = Convolution2D(4, (1, 3), padding='same', name='block1_conv1')(ecog_series)
        # x = BatchNormalization(axis=1)(x)
        x = Activation('tanh')(x)
        x = MaxPooling2D((1, 3), name='block1_pool')(x)

        # Block 2
        x = Convolution2D(8, (1, 3), padding='same', name='block2_conv1')(x)
        # x = BatchNormalization(axis=1)(x)
        x = Activation('tanh')(x)
        x = MaxPooling2D((1, 3), name='block2_pool')(x)

        # Block 3
        x = Convolution2D(16, (1, 3), padding='same', name='block3_conv1')(x)
        # x = BatchNormalization(axis=1)(x)
        x = Activation('tanh')(x)
        x = Dense(64, name='fc1')(x)
        x = Activation('tanh')(x)
        x = Dense(len(channels), name='predictions')(x)
        # x = BatchNormalization()(x)
        predictions = Activation('sigmoid')(x)

        def selected_loss(input):
            def loss(y_true, y_pred):
                inds = not (y_true==input)
                return K.mean(K.square(y_pred[inds] - y_true[inds]), axis=-1)
            return loss

        model = Model(input=[ecog_series], output=predictions)
        sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9)
        model_savepath = "/home/wangnxr/models/ecog_model_impute_%s_itr_%i" % (sbj, itr)
        model.compile(optimizer=sgd,
                      loss=[selected_loss(input=ecog_series)],
                      metrics=['loss'])
        early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=10, verbose=0, mode='auto')
        checkpoint = ModelCheckpoint("%s_best.h5" % model_savepath, monitor='val_loss', verbose=1,
                                     save_best_only=True, mode='min')
        history_callback = model.fit_generator(
            train_generator,
            samples_per_epoch=len(dgdx_edf.filenames),
            nb_epoch=200,
            validation_data=validation_generator,
            nb_val_samples=len(dgdx_val_edf.filenames), callbacks=[checkpoint, early_stop])

        model.save("%s.h5" % model_savepath)
        pickle.dump(history_callback.history,
                    open("/home/wangnxr/history/%s.p" % model_savepath.split("/")[-1].split(".")[0], "wb"))
