import keras
import keras.backend as K
from keras.preprocessing.ecog import EcogDataGenerator
from keras.preprocessing.image2 import ImageDataGenerator, center_crop
from keras.layers import Flatten, Dense, Input, Dropout, Activation, merge, TimeDistributed, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.models import Model, load_model
from hyperopt import Trials, fmin, tpe, hp, STATUS_OK
from keras.regularizers import l2
from itertools import izip
from keras.callbacks import ModelCheckpoint, EarlyStopping
from ecogdeep.train.ecog_1d_model_seq import ecog_1d_model
from ecogdeep.train.vid_model_seq import vid_model
from sbj_parameters import *

#from keras.imagenet_utils import decode_predictions, preprocess_input, _obtain_input_shape
import numpy as np
import pdb
import pickle
import glob


sbj_to_do = ["cb4", "e5b"]

for itr in xrange(3):
    for s, sbj in enumerate(sbj_ids):
   	if sbj in sbj_to_do:
            main_ecog_dir = '/home/wangnxr/dataset/ecog_vid_combined_%s_day%i/' % (sbj, days[s])
        else:
            continue

        for t, time in enumerate(start_times):
            model_files = glob.glob(
                '/home/wangnxr/models/ecog_model_lstm20_%s_itr_%i_t_%i__weights_*.h5' % (sbj, itr, time))
            ## Data generation ECoG
            channels = channels_list[s]
            train_datagen_edf = EcogDataGenerator(
                time_shift_range=200,
                gaussian_noise_range=0.001,
                center=False,
                seq_len=200,
                start_time=time,
                seq_num = 5,
                seq_st = 200
            )

            test_datagen_edf = EcogDataGenerator(
                time_shift_range=200,
                center=True,
                seq_len=200,
                start_time=time,
                seq_num=5,
                seq_st=200
            )

            dgdx_edf = train_datagen_edf.flow_from_directory(
                '%s/train/' % main_ecog_dir,
                batch_size=24,
                class_mode='binary',
                shuffle=False,
                channels = channels,
                pre_shuffle_ind=1,
                target_size=(1, len(channels), 1000),
                final_size=(1,len(channels),200),
                )

            dgdx_val_edf = test_datagen_edf.flow_from_directory(
                '%s/val/' % main_ecog_dir,
                batch_size=10,
                shuffle=False,
                target_size=(1, len(channels), 1000),
                final_size=(1,len(channels),200),
                channels = channels,
                class_mode='binary')

            if len(model_files) == 0:
                continue
            last_model_ind = np.argmax([int(file.split("_")[-1].split(".")[0]) for file in model_files])
            # pre_shuffle_index = np.random.permutation(len(glob.glob('%s/train/*/*.npy' % main_ecog_dir)))
            ## Data generation ECoG
            channels = channels_list[s]
            model_file = model_files[last_model_ind]
            model = load_model(model_file)

            ecog_series = Input(shape=(5, 1, len(channels), 200))

            base_model_ecog = Model(model.input, model.layers[-7].output)

            ecog_model = ecog_1d_model(channels=len(channels))
            train_generator =  dgdx_edf
            validation_generator =  dgdx_val_edf
            x = base_model_ecog(ecog_series)
            x = Activation('relu')(x)
            x = Dropout(0.5)(x)
            x = Lambda(function=lambda x: K.mean(x, axis=1),
                   output_shape=lambda shape: (shape[0],) + shape[2:])(x)
            x = Dense(1, name='predictions')(x)
            #x = BatchNormalization()(x)
            predictions = Activation('sigmoid')(x)

            for layer in base_model_ecog.layers:
                layer.trainable = False
            model = Model(input=[ecog_series], output=predictions)

            sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9)

            model_savepath = "/home/wangnxr/models/ecog_model_lstm20_%s_itr_%i_t_%i_frozen_" % (sbj,itr,time)
            model.compile(optimizer=sgd,
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
            early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=10, verbose=0, mode='auto')
            checkpoint = ModelCheckpoint("%s_weights_{epoch:02d}.h5" % model_savepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
            history_callback = model.fit_generator(
                train_generator,
                samples_per_epoch=len(dgdx_edf.filenames),
                nb_epoch=40,
                validation_data=validation_generator,
                nb_val_samples=len(dgdx_val_edf.filenames), callbacks=[checkpoint, early_stop])

            model.save("%s.h5" % model_savepath)
            pickle.dump(history_callback.history, open("/home/wangnxr/history/%s.p" % model_savepath.split("/")[-1].split(".")[0], "wb"))
