import keras
from keras.preprocessing.ecog import EcogDataGenerator
from sklearn.linear_model import SGDClassifier
import numpy as np
import pdb
import pickle
import pdb
from ecogdeep.train.sbj_parameters import *
import glob
"""ECoG SVM testing.


Example:
        $ python ecog_svm_test.py

"""
# Save file
with open("/home/wangnxr/results/ecog_svm_summary_results.txt", "wb") as summary_writer:
    for s, sbj in enumerate(sbj_ids):
        for time in start_times:
            main_ecog_dir = '/home/wangnxr/dataset/ecog_vid_combined_%s_day%i/test/' % (sbj, days[s])
            model_files = glob.glob(
                '/home/wangnxr/models/ecog_model_svm_%s_t_%i.p' % (sbj, time))
            if len(model_files)==0:
                continue
            ## Data generation ECoG
            channels = channels_list[s]
            model_file = model_files[0]

            test_datagen = EcogDataGenerator(
                    start_time=time,
                    #f_lo=60,
                    #f_hi=90,
                    fft=True,
                    center=False
            )

            dgdx_val_edf = test_datagen.flow_from_directory(
                    #'/mnt/cb46fd46_5_no_offset/test/',
                    main_ecog_dir,
                    batch_size=10,
                    shuffle=False,
                    target_size=(1,len(channels),1000),
                    final_size=(1,len(channels),2),
                    channels=channels,
                    class_mode='binary')

            validation_generator =  dgdx_val_edf
            samples_per_epoch_test = validation_generator.nb_sample
            model = pickle.load(open(model_file))

            val_x = []
            val_y = []
            for b in xrange(samples_per_epoch_test / validation_generator.batch_size + 1):
                temp_data = validation_generator.next()
                val_x.append(temp_data[0])
                val_y.append(temp_data[1])
            val_x = np.vstack(val_x)
            val_y = np.hstack(val_y)

            mean_test_score_0 = []
            mean_test_score_1 = []

            result = model.predict(np.reshape(val_x, (samples_per_epoch_test, val_x.shape[2] * val_x.shape[3])))
            for d, data in enumerate(val_y):
                if data == 0:
                    mean_test_score_0.append(result[d])
                else:
                    mean_test_score_1.append(result[d])
            summary_writer.write("%s\n" % model_file.split("/")[-1])
            summary_writer.write("accuracy_0: %f\n" % (len(np.where(np.array(mean_test_score_0)==0)[0])/float(len(mean_test_score_0))))
            summary_writer.write("accuracy_1: %f\n" % (len(np.where(np.array(mean_test_score_1) == 1)[0]) / float(len(mean_test_score_1))))




