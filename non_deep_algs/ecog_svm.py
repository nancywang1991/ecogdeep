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
    for itr in xrange(3):
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
            nb_epoch=5

            model = SGDClassifier(verbose=0,n_jobs=8)
            logfile = open("/home/wangnxr/history/ecog_model_svm_%s_itr_%i_t_%i.txt" % (sbj, itr, time), "wb")
            #test_data = np.vstack([val for val in validation_generator])
            best_score = 0
            for e in xrange(nb_epoch):
                print("Epoch: %i" % e, file=logfile)
                mean_test_score = []
                for b in xrange(samples_per_epoch/train_generator.batch_size):
                    data = train_generator.next()

                    if len(data[0])<train_generator.batch_size:
                        train_generator.reset()
                        data = train_generator.next()
                    model.partial_fit(np.reshape(data[0], (train_generator.batch_size, data[0].shape[2]*data[0].shape[3])), data[1], classes=(0,1))
                    if b%50==0:
                            print(model.score(np.reshape(data[0], (train_generator.batch_size, data[0].shape[2]*data[0].shape[3])),data[1]), file=logfile)
                            print("Batch %i of %i" % (b, samples_per_epoch/train_generator.batch_size), file=logfile)
                            logfile.flush()
                for b in xrange(samples_per_epoch_test/validation_generator.batch_size):
                    test_data = validation_generator.next()
                    #pdb.set_trace()
                    if len(test_data[0])<validation_generator.batch_size:
                            validation_generator.reset()
                            test_data = validation_generator.next()
                    mean_test_score.append(model.score(np.reshape(test_data[0], (validation_generator.batch_size, test_data[0].shape[2]*data[0].shape[3])), test_data[1]))
                print("Val_acc = %f" % np.mean(np.array(mean_test_score)), file=logfile)
                if np.mean(np.array(mean_test_score)) > best_score:
                    pickle.dump(model, open("/home/wangnxr/models/ecog_model_svm_%s_itr_%i_t_%i_ep_%i.p" % (sbj, itr, time, e), "wb"))
                    best_score = np.mean(np.array(mean_test_score))
                    print("Model saved", file=logfile)





