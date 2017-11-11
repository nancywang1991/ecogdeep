import keras
import cv2
from keras.preprocessing.ecog_reg_xy import EcogDataGenerator
from keras.layers import Flatten, Dense, Input, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model, load_model
import numpy as np
import pdb
from keras.preprocessing.image_reg import ImageDataGenerator
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from sbj_parameters import *
from matplotlib.colors import rgb_to_hsv

"""Accuracy of test set using ECoG LSTM model.

"""

sbj_to_do = ["cb4"]
start_times = [2800]
for s, sbj in enumerate(sbj_ids):
    if sbj in sbj_to_do:
        main_ecog_dir = '/home/wangnxr/dataset_xy_reg/ecog_vid_combined_%s_day%i/' % (sbj, days[s])
    else:
        continue
    for t, time in enumerate(start_times):
        try:
            model_file =  "/home/wangnxr/models/ecog_model_cb4_itr_0_reg_v11__valbest_chkpt.h5"
            model = load_model(model_file)
        except:
            continue
        # pre_shuffle_index = np.random.permutation(len(glob.glob('%s/train/*/*.npy' % main_ecog_dir)))
        ## Data generation ECoG
        channels = channels_list[s]
        times = [3900,3850,3800,3750,3700,3650,3600,3550,3500,3450,3400]

        test_datagen_edf = EcogDataGenerator(
            time_shift_range=200,
            center=True,
            #seq_len=200,
            start_time=3500,
            #seq_num=5,
            #seq_st=200
        )

        dgdx_val_edf = test_datagen_edf.flow_from_directory(
            '%s/test/' % main_ecog_dir,
            batch_size=10,
            shuffle=False,
            target_size=(1, len(channels), 1000),
            final_size=(1,len(channels),1000),
            channels = channels,
            class_mode='binary')

        def extract_max(heatmap):
            max_point = np.argmax(heatmap)
            return (max_point%56, max_point/56)

        validation_generator = dgdx_val_edf
        #validation_generator = dgdx_val_vid
        #for layer in base_model.layers[:10]:
        #    layer.trainable = False
        #pdb.set_trace()
        files = dgdx_val_edf.filenames
        total_dist = 0
        #results = model.predict_generator(validation_generator, len(files))
        predictions = []
	tests = []
        for b in xrange(10):
            test = validation_generator.next()
            tests.append(np.vstack(test[1]).T)
            prediction = np.hstack(model.predict(test[0]))
            predictions.append(prediction)
                #plt.imshow(np.reshape(prediction[i], (56,56)))
                #plt.savefig("/home/wangnxr/results_tmp/ecog_test_%i_%i.png" % (b,i))
                #plt.imshow(np.reshape(test[1][i], (56,56)))
                #plt.savefig("/home/wangnxr/results_tmp/ecog_orig_%i_%i.png" % (b,i))
                #plt.imshow(cv2.resize(np.ndarray.transpose(test[0][0][i], (1,2,0)), (56,56)))
                #plt.savefig("/home/wangnxr/results_tmp/ecog_input_%i_%i.png" % (b,i))
                #dist= np.array(extract_max(test[1][i])) - np.array(extract_max(prediction[i]))
            #dist = prediction-test[1]
            #print np.mean(np.abs(dist))
            #total_dist += np.sum(np.abs(dist))
        #print total_dist
	print "direction corr:" + str(np.corrcoef(np.vstack(predictions)[:,0], np.vstack(tests)[:,0]))
	print "magnitude corr:" + str(np.corrcoef(np.vstack(predictions)[:,1], np.vstack(tests)[:,1]))


