from __future__ import print_function
import keras
from keras.preprocessing.ecog import EcogDataGenerator
from sklearn.linear_model import SGDClassifier
import numpy as np
import pdb
import pickle
from ecogdeep.train.sbj_parameters import *
sbj_ids = ['a0f', 'e5b', 'd65', "cb4", "c95"]

sbj_to_do = ["c95"]

for s, sbj in enumerate(sbj_ids):
    if sbj in sbj_to_do:
        main_ecog_dir = '/home/wangnxr/dataset/ecog_vid_combined_%s_day%i/' % (sbj, days[s])
    else:
        continue
    for t, time in enumerate(start_times):
        channels = channels_list[s]
        ## Data generation
        train_datagen_edf = EcogDataGenerator(
            start_time=time,
            time_shift_range=200,
            gaussian_noise_range=0.001,
            center=False,
            fft = True
        )

        test_datagen_edf = EcogDataGenerator(
            start_time=time,
            center=True,
            time_shift_range = 200,
            fft=True
        )

        dgdx_edf = train_datagen_edf.flow_from_directory(
            # '/mnt/cb46fd46_5_no_offset/train/',
            '%s/train/' % main_ecog_dir,
            batch_size=24,
            target_size=(1, len(channels), 1000),
            final_size=(1, len(channels), 2),
            class_mode='binary',
            shuffle=False,
            channels=channels,
            pre_shuffle_ind=1)

        dgdx_val_edf = test_datagen_edf.flow_from_directory(
            # '/mnt/cb46fd46_5_no_offset/test/',
            '%s/val/' % main_ecog_dir,
            batch_size=10,
            shuffle=False,
            target_size=(1, len(channels), 1000),
            final_size=(1, len(channels), 2),
            channels=channels,
            class_mode='binary')

        train_generator=dgdx_edf
        validation_generator=dgdx_val_edf
        samples_per_epoch=train_generator.nb_sample
        samples_per_epoch_test=validation_generator.nb_sample
        x = []
        y = []
        for b in xrange(samples_per_epoch / train_generator.batch_size +1):
            temp_data = train_generator.next()
            x.append(temp_data[0])
            y.append(temp_data[1])
        x = np.vstack(x)
        y = np.hstack(y)
        print(x.shape)
        print(y.shape)

        val_x = []
        val_y = []
        for b in xrange(samples_per_epoch / validation_generator.batch_size):
            temp_data = train_generator.next()
            val_x.append(temp_data[0])
            val_y.append(temp_data[1])
        val_x = np.vstack(val_x)
        val_y = np.hstack(val_y)

        model = SGDClassifier(verbose=0,n_jobs=8)
        logfile = open("/home/wangnxr/history/ecog_model_svm_%s_t_%i.txt" % (sbj, time), "wb")
        #test_data = np.vstack([val for val in validation_generator])
        best_score = 0

        model.fit(np.reshape(x, (samples_per_epoch, x.shape[2]*x.shape[3])), y)
        print("training acc: %f" % model.score(np.reshape(x, (samples_per_epoch, x.shape[2]*x.shape[3])),y), file=logfile)
        print("validation acc: %f" % model.score(np.reshape(val_x, (samples_per_epoch, val_x.shape[2] * val_x.shape[3])), val_y),
            file=logfile)
        pickle.dump(model, open("/home/wangnxr/models/ecog_model_svm_%s_t_%i.p" % (sbj, time), "wb"))






