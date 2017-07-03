import keras
from keras.preprocessing.image_reg import ImageDataGenerator
from keras.layers import Flatten, Dense, Input, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model, load_model
import numpy as np
import pdb
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import cv2

"""Accuracy of test set using ECoG LSTM model.

"""

sbj_ids = ['a0f']
days = [11]
start_times = [2800,3400,4000]
channels_list = [np.hstack([np.arange(36), np.arange(37, 65), np.arange(66, 92)])]
for s, sbj in enumerate(sbj_ids):
    main_vid_dir = '/home/wangnxr/dataset_xy_reg/ecog_vid_combined_%s_day%i/' % (sbj, days[s])
    for t, time in enumerate(start_times):
        try:
            model_file =  "/home/wangnxr/models/vid_model2_reg_chkpt.h5"
            model = load_model(model_file)
        except:
            continue
        # pre_shuffle_index = np.random.permutation(len(glob.glob('%s/train/*/*.npy' % main_ecog_dir)))
        ## Data generation ECoG
        channels = channels_list[s]
        times = [3900,3850,3800,3750,3700,3650,3600,3550,3500,3450,3400]

        test_datagen_vid = ImageDataGenerator(
            rescale=1. / 255,
            start_time=times[0],
            center_crop=(224, 224))

        dgdx_val_vid = test_datagen_vid.flow_from_directory(
            '/%s/val/' % main_vid_dir,
            read_formats={'png'},
            target_size=(int(224), int(224)),
            num_frames=12,
            batch_size=10,
            shuffle=False,
            class_mode='binary')

        validation_generator =  dgdx_val_vid

        #for layer in base_model.layers[:10]:
        #    layer.trainable = False
        #pdb.set_trace()
        files = dgdx_val_vid.filenames
        #results = model.predict_generator(validation_generator, len(files))
	for b in xrange(4):
	    test = validation_generator.next()
            for i in xrange(10):
                plt.imshow(model.predict(test[0])[i][0])
                plt.savefig("/home/wangnxr/test_%i_%i.png" % (b,i))
                plt.imshow(test[1][i][0])
                plt.savefig("/home/wangnxr/orig_%i_%i.png" % (b,i))
		plt.imshow(cv2.resize(np.ndarray.transpose(test[0][i], (1,2,0)), (56,56)))
		plt.savefig("/home/wangnxr/input_%i_%i.png" % (b,i))
        pdb.set_trace()

