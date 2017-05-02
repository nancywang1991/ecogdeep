import keras
from keras.preprocessing.ecog import EcogDataGenerator
from sklearn.linear_model import SGDClassifier
import numpy as np
import pdb
import pickle

sbj_ids = ['c95']
days = [7]
start_times = [2800, 3400, 4000]
channels_list = [#np.hstack([np.arange(36), np.arange(37, 65), np.arange(66, 92)]),
                 #np.hstack([np.arange(80), np.arange(81, 85), np.arange(86, 104),np.arange(105, 108), np.arange(110, 111)]), 
		np.arange(86)]
#sbj_ids = ['a0f','e5b','d65', "cb4", "c95"]
#days = [10,9,9,10,7]
#sbj_ids = ['a0f']
#days = [10]
#start_times = [2800, 3400, 4000]
#channels_list = [np.hstack([np.arange(36), np.arange(37, 65), np.arange(66, 92)]),
#                 np.hstack([np.arange(80), np.arange(81, 85), np.arange(86, 104),np.arange(105, 108), np.arange(110, 111)]),
#                 np.arange(82),
#                 np.arange(80),
#                 np.arange(86)]

for s, sbj in enumerate(sbj_ids):
        main_ecog_dir = '/home/wangnxr/dataset/ecog_vid_combined_%s_day%i/' % (sbj, days[s])
        for t, time in enumerate(start_times):
                channels = channels_list[s]
                ## Data generation

                test_datagen = EcogDataGenerator(
                        start_time=time,
                        #f_lo=60,
                        #f_hi=90,
                        fft=True,
                        center=False
                )

                dgdx_val = test_datagen.flow_from_directory(
                        #'/mnt/cb46fd46_5_no_offset/test/',
                        '/%s/test/' % main_ecog_dir,
                        batch_size=10,
                        shuffle=False,
                        target_size=(1,len(channels),1000),
                        final_size=(1,len(channels),2),
                        channels=channels,
                        class_mode='binary')

                validation_generator=dgdx_val
                samples_per_epoch_test=validation_generator.nb_sample
                mean_test_score_0=[]
		mean_test_score_1=[]
                try:
                    model = pickle.load(open("/home/wangnxr/models/svm_ecog_%s_t_%i.p" % (sbj, time)))
                except:
                    continue
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
                print len(np.where(np.array(mean_test_score_0)==0)[0])/float(len(mean_test_score_0))
		print len(np.where(np.array(mean_test_score_1)==1)[0])/float(len(mean_test_score_1))




