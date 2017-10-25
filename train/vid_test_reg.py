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

"""Accuracy of test set using ECoG LSTM model.

"""

sbj_to_do = ["a0f"]
start_times = [2800,3400,4000]
for s, sbj in enumerate(sbj_ids):
    if sbj in sbj_to_do:
        main_ecog_dir = '/home/wangnxr/dataset_xy_reg/ecog_vid_combined_%s_day%i/' % (sbj, days[s])
        main_vid_dir = '/home/wangnxr/dataset_xy_reg/ecog_vid_combined_%s_day%i/' % (sbj, days[s])
    else:
        continue
    for t, time in enumerate(start_times):
        try:
            model_file =  "/home/wangnxr/models/vid_model_a0f_itr_0_reg_89_chkpt.h5"
            model = load_model(model_file)
        except:
            continue
        times = [3900, 3850, 3800, 3750, 3700, 3650, 3600, 3550, 3500, 3450, 3400]
        # pre_shuffle_index = np.random.permutation(len(glob.glob('%s/train/*/*.npy' % main_ecog_dir)))
        test_datagen_vid = ImageDataGenerator(
            rescale=1. / 255,
            start_time=times[0],
            center_crop=(224, 224))

        dgdx_val_vid = test_datagen_vid.flow_from_directory(
            '/%s/test/' % main_vid_dir,
            read_formats={'png'},
            target_size=(int(224), int(224)),
            num_frames=12,
            batch_size=10,
            shuffle=False,
            class_mode='binary')

        def extract_max(heatmap):
            max_point = np.argmax(heatmap)
            return (max_point%56, max_point/56)

        validation_generator = dgdx_val_vid
        #validation_generator = dgdx_val_vid
        #for layer in base_model.layers[:10]:
        #    layer.trainable = False
        #pdb.set_trace()
        files = dgdx_val_vid.filenames
        total_dist = 0
        #results = model.predict_generator(validation_generator, len(files))
        for b in xrange(4):
            test = validation_generator.next()
            prediction = model.predict(test[0])
            for i in xrange(10):
                plt.imshow(np.reshape(prediction[i], (56,56)))
                plt.savefig("/home/wangnxr/results_tmp/vid_test_%i_%i.png" % (b,i))
                plt.imshow(np.reshape(test[1][i], (56,56)))
                plt.savefig("/home/wangnxr/results_tmp/vid_orig_%i_%i.png" % (b,i))
                #plt.imshow(cv2.resize(np.ndarray.transpose(test[0][0][i], (1,2,0)), (56,56)))
                #plt.savefig("/home/wangnxr/results_tmp/vid_input_%i_%i.png" % (b,i))
                dist= np.array(extract_max(test[1][i])) - np.array(extract_max(prediction[i]))
                print dist
                total_dist += np.sum(np.abs(dist))
        print total_dist
        pdb.set_trace()

