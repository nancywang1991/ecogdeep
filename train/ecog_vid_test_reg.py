import keras
import keras.backend as K
import theano
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
import h5py

"""Accuracy of test set using ECoG LSTM model.

"""

sbj_to_do = ["cb4"]
start_times = [2800,3400,4000]
for s, sbj in enumerate(sbj_ids):
    if sbj in sbj_to_do:
        main_ecog_dir = '/home/wangnxr/dataset_xy_reg/ecog_vid_combined_%s_day%i/' % (sbj, days[s])
        main_vid_dir = '/home/wangnxr/dataset_xy_reg/ecog_vid_combined_%s_day%i/' % (sbj, days[s])
    else:
        continue
    for t, time in enumerate(start_times):
        try:
            model_file =  "/home/wangnxr/models/ecog_vid_model_cb4_itr_0_reg_v4.h5"
	    try:
	    	f = h5py.File(model_file, 'r+')
	    	del f['optimizer_weights']
	    	f.close()
	    except:
		pass
            model = load_model(model_file)
	    #pdb.set_trace()
	    #model = Model(model.input, model.layers[9].output)
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
            '%s/train/' % main_ecog_dir,
            batch_size=10,
            shuffle=False,
            target_size=(1, len(channels), 1000),
            final_size=(1,len(channels),1000),
            channels = channels,
            class_mode='binary')

        test_datagen_vid = ImageDataGenerator(
            rescale=1. / 255,
            start_time=times[0],
            center_crop=(224, 224))

        dgdx_val_vid = test_datagen_vid.flow_from_directory(
            '/%s/train/' % main_vid_dir,
            read_formats={'png'},
            target_size=(int(224), int(224)),
            num_frames=12,
            batch_size=10,
            shuffle=False,
            class_mode='binary')


        def izip_input(gen1, gen2):
            while 1:
                # pdb.set_trace()
                x1, y1 = gen1.next()
                x2, y2 = gen2.next()
                if not x1.shape[0] == x2.shape[0]:
                    pdb.set_trace()
                
                #yield [x1, x2], [y1, y2]
		yield [x1, x2], [y1, y2]
        def extract_max(heatmap):
            max_point = np.argmax(heatmap)
            return (max_point%56, max_point/56)
        validation_generator = izip_input(dgdx_val_vid, dgdx_val_edf)
        #validation_generator = dgdx_val_vid
        #for layer in base_model.layers[:10]:
        #    layer.trainable = False
        #pdb.set_trace()
        files = dgdx_val_edf.filenames
        total_dist = 0
        #results = model.predict_generator(validation_generator, len(files))
        for b in xrange(4):
            test = validation_generator.next()
	    #get_activations = K.function([model.layers[3].layers[0].input, K.learning_phase()], [model.layers[3].layers[-1].output])
	    #prediction = get_activations([test[0][1],1])[0]
            prediction = model.predict(test[0])
            for i in xrange(10):
		#for j in xrange(256):
		#	plt.imshow(np.reshape(prediction[i][j], (56,56)))
                #	plt.savefig("/home/wangnxr/results_tmp/ecogvid2_test_%i_%i.png" % (j,i))
		#pdb.set_trace()
                plt.imshow(np.reshape(prediction[0][i], (56,56)))
                plt.savefig("/home/wangnxr/results_tmp/ecogvid2_test_%i_%i.png" % (b,i))
                plt.imshow(np.reshape(test[1][0][i], (56,56)))
                plt.savefig("/home/wangnxr/results_tmp/ecogvid2_orig_%i_%i.png" % (b,i))
                plt.imshow(cv2.resize(np.ndarray.transpose(test[0][0][i], (1,2,0)), (56,56)))
                plt.savefig("/home/wangnxr/results_tmp/ecogvid2_input_%i_%i.png" % (b,i))
                dist= np.array(extract_max(test[1][0][i])) - np.array(extract_max(prediction[0][i]))
                print dist
                total_dist += np.sum(np.abs(dist))
		print prediction[1][i], test[1][1][0][i]
        print total_dist
        pdb.set_trace()

