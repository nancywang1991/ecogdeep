import keras
from keras.preprocessing.ecog import EcogDataGenerator
from sklearn.linear_model import SGDClassifier
import numpy as np
import pdb
import pickle
import pdb
from ecogdeep.train.sbj_parameters import *
import glob


with open("/home/wangnxr/results/ecog_svm_summary_results.txt", "wb") as summary_writer:
    for s, sbj in enumerate(sbj_ids):
        for time in start_times:
            main_ecog_dir = '/home/wangnxr/dataset/ecog_vid_combined_%s_day%i/test/' % (sbj, days[s])
            for itr in xrange(3):
                model_files = glob.glob('/home/wangnxr/models/ecog_model_%s_itr_%i_t_%i_*.h5' % (sbj, itr, time))
                if len(model_files)==0:
                    continue
                last_model_ind = np.argmax([int(file.split("_")[-1].split(".")[0]) for file in model_files])
                #pre_shuffle_index = np.random.permutation(len(glob.glob('%s/train/*/*.npy' % main_ecog_dir)))
                ## Data generation ECoG
                channels = channels_list[s]
                model_file = model_files[last_model_ind]

                test_datagen = EcogDataGenerator(
                        start_time=time,
                        #f_lo=60,
                        #f_hi=90,
                        fft=True,
                        center=False
                )

                dgdx_val_edf = test_datagen.flow_from_directory(
                        #'/mnt/cb46fd46_5_no_offset/test/',
                        '/%s/test/' % main_ecog_dir,
                        batch_size=10,
                        shuffle=False,
                        target_size=(1,len(channels),1000),
                        final_size=(1,len(channels),2),
                        channels=channels,
                        class_mode='binary')

                validation_generator =  dgdx_val_edf
                samples_per_epoch_test = validation_generator.nb_sample
                model = pickle.load(model_file)
                mean_test_score_0 = []
                mean_test_score_1 = []
                for b in xrange(samples_per_epoch_test/validation_generator.batch_size):
                        test_data = validation_generator.next()
                        if len(test_data[0])<validation_generator.batch_size:
                                validation_generator.reset()
                                test_data = validation_generator.next()
                        result = model.predict(np.reshape(test_data[0], (validation_generator.batch_size, test_data[0].shape[2]*test_data[0].shape[3])))
                        for d, data in enumerate(test_data[1]):
                                if data == 0:
                                        mean_test_score_0.append(result[d])
                                else:
                                        mean_test_score_1.append(result[d])
                summary_writer.write("%s" % model_files.split("/")[-1])
                summary_writer.write("accuracy_0: %f" % len(np.where(np.array(mean_test_score_0)==0)[0])/float(len(mean_test_score_0)))
                summary_writer.write("accuracy_1: %f" % len(np.where(np.array(mean_test_score_1) == 1)[0]) / float(len(mean_test_score_1)))




