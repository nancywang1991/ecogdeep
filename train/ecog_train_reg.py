import keras
from keras.layers import Flatten, Dense, Input, Dropout, Activation, merge
from keras.preprocessing.image_reg import ImageDataGenerator, center_crop
from keras.models import Model
from ecogdeep.train.ecog_1d_model_reg import ecog_1d_model
from keras.preprocessing.ecog_reg_xy import EcogDataGenerator
from ecogdeep.train.vid_model_reg import vid_model
from keras.callbacks import ModelCheckpoint
from sbj_parameters import *

#from keras.imagenet_utils import decode_predictions, preprocess_input, _obtain_input_shape
import numpy as np
import pdb
import pickle
import glob
import time

sbj_to_do = ["a0f", "cb4"]
for s, sbj in enumerate(sbj_ids):
    if sbj in sbj_to_do:
        main_ecog_dir = '/home/wangnxr/dataset_xy_reg/ecog_vid_combined_%s_day%i/' % (sbj, days[s])
    else:
        continue
    for itr in range(1):
        times = [3900]

        ## Data generation ECoG
        channels = channels_list[sbj_ids.index(sbj)]

        train_datagen_edf = EcogDataGenerator(
                    time_shift_range=200,
                    gaussian_noise_range=0.001,
                    center=False,
                    start_time=times,
                )

        test_datagen_edf = EcogDataGenerator(
                    time_shift_range=200,
                    center=True,
                    start_time=times[0],
                )

        dgdx_edf = train_datagen_edf.flow_from_directory(
            #'/mnt/cb46fd46_5_no_offset/train/',
            '%s/train/' % main_ecog_dir,
            batch_size=24,
            target_size=(1,len(channels),1000),
            final_size=(1,len(channels),1000),
            class_mode='binary',
            shuffle=False,
            channels = channels,
            pre_shuffle_ind=1)

        dgdx_val_edf = test_datagen_edf.flow_from_directory(
            #'/mnt/cb46fd46_5_no_offset/test/',
            '%s/val/' % main_ecog_dir,
            batch_size=10,
            shuffle=False,
            target_size=(1,len(channels),1000),
            final_size=(1,len(channels),1000),
            channels = channels,
            class_mode='binary')
        ecog_model = ecog_1d_model(channels=len(channels))
        base_model_ecog = Model(ecog_model.input, ecog_model.get_layer("predictions").output)
        ecog_series = Input(shape=(1,len(channels),1000))

        train_generator = dgdx_edf
        validation_generator = dgdx_val_edf

        predictions = base_model_ecog(ecog_series)

        sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9)

        model = Model(input=ecog_series, output=predictions)

        model_savepath = "/home/wangnxr/models/ecog_model_%s_itr_%i_reg_v5_" % (sbj, itr)
        model.compile(optimizer="rmsprop",
                      loss='mean_squared_error')
        checkpoint = ModelCheckpoint(model_savepath + "_valbest_chkpt.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        history_callback = model.fit_generator(
            train_generator,
            samples_per_epoch=len(dgdx_edf.filenames),
            nb_epoch=100,
            validation_data=validation_generator,
            nb_val_samples=len(dgdx_val_edf.filenames), callbacks=[checkpoint])

        model.save("%s.h5" % model_savepath)
        pickle.dump(history_callback.history, open("/home/wangnxr/history/ecog_model_%s_itr_%i_reg_v5_" % (sbj, itr), "wb"))
    time.sleep(50)
