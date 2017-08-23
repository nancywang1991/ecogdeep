import keras
from keras.preprocessing.ecog import EcogDataGenerator
from keras.preprocessing.image2 import ImageDataGenerator, center_crop
from keras.layers import Flatten, Dense, Input, Dropout, Activation, merge, TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from hyperopt import Trials, fmin, tpe, hp, STATUS_OK
from keras.regularizers import l2
from itertools import izip
from ecogdeep.train.ecog_vid_1d_model_seq import ecog_vid_1d_model
from sbj_parameters import *

import numpy as np
import pdb
import pickle
import glob


sbj_to_do = sbj_ids[:]
for s, sbj in enumerate(sbj_ids):
    if sbj in sbj_to_do:
        main_vid_dir = '/home/wangnxr/dataset/ecog_vid_combined_%s_day%i/' % (sbj, days[s])
        main_ecog_dir = '/home/wangnxr/dataset/ecog_vid_combined_%s_day%i/' % (sbj, days[s])
    else:
        continue
    for itr in range(1):
        for t, time in enumerate(start_times):
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

            # Video data generators
            train_datagen_vid = ImageDataGenerator(
                rescale=1./255,
                random_black=True,
                #random_crop=(224,224),
                keep_frames=frames[t])

            test_datagen_vid = ImageDataGenerator(
                rescale=1./255,
                #center_crop=(224, 224),
                keep_frames=frames[t])

            model = ecog_vid_1d_model(channels=len(channels))

            dgdx_vid = train_datagen_vid.flow_from_directory(
                '/%s/train/' % main_vid_dir,
                img_mode="seq",
                read_formats={'png'},
                target_size=(len(channels), int(200)),
                resize_size=(len(channels), int(200)),
                num_frames=12,
                batch_size=24,
                class_mode='binary',
                shuffle=False,
                pre_shuffle_ind=1)

            dgdx_val_vid = test_datagen_vid.flow_from_directory(
                '/%s/val/' % main_vid_dir,
                img_mode="seq",
                read_formats={'png'},
                target_size=(len(channels), int(200)),
                resize_size=(len(channels), int(200)),
                num_frames=12,
                batch_size=10,
                shuffle=False,
                class_mode='binary')

            def izip_input(gen1, gen2):
                while 1:
                    x1, y1 = gen1.next()
                    x2 = gen2.next()[0]

                    if not x1.shape[0] == x2.shape[0]:
                        pdb.set_trace()
                    combined = []
                    for i in xrange(len(x2)):
                        combined.append(np.hstack([x1[i], x2[i]]))
                    yield combined, y1

            train_generator = izip_input(dgdx_vid, dgdx_edf)
            validation_generator = izip_input(dgdx_val_vid, dgdx_val_edf)
            base_model = Model(model.input, model.get_layer("fc1").output)

            ecog_vid_series = Input(shape=(5,4,len(channels),200))


            x = base_model(ecog_vid_series)

            x = Dropout(0.5)(x)
            x = TimeDistributed(Dense(256, W_regularizer=l2(0.01), name='merge1'))(x)
            #x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Dropout(0.5)(x)
            x = TimeDistributed(Dense(64, W_regularizer=l2(0.01), name='merge2'))(x)
            #x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Dropout(0.5)(x)
            x = LSTM(20, dropout_W=0.2, dropout_U=0.2, name='lstm')(x)
            x = Dense(1, name='predictions')(x)
            #x = BatchNormalization()(x)
            predictions = Activation('sigmoid')(x)
            #predictions = Dense(2, activation='softmax', name='predictions')(x)
            for layer in base_model.layers:
                layer.trainable = True

            model = Model(input=ecog_vid_series, output=predictions)

            sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9)

            model_savepath = "/home/wangnxr/models/ecog_vid_model_lstm_early_fus_%s_itr_%i_t_%i_v2_" % (sbj, itr, time)
            model.compile(optimizer=sgd,
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
            early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=10, verbose=0, mode='auto')
            checkpoint = ModelCheckpoint("%s_chkpt.h5" % model_savepath, monitor='val_acc', verbose=1, save_best_only=True,
                                         mode='max')
            history_callback = model.fit_generator(
                train_generator,
                samples_per_epoch=len(dgdx_vid.filenames),
                nb_epoch=200,
                validation_data=validation_generator,
                nb_val_samples=len(dgdx_val_vid.filenames), callbacks=[checkpoint, early_stop])

            model.save("%s.h5" % model_savepath)
            pickle.dump(history_callback.history,
                        open("/home/wangnxr/history/%s.p" % model_savepath.split("/")[-1].split(".")[0], "wb"))
