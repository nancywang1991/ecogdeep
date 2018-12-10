import keras
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense, Input, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Model, load_model
from keras.utils.generic_utils import get_custom_objects
from ecogdeep.imputation.conv_mlp import selected_loss
from copy import copy
#from keras.imagenet_utils import decode_predictions, preprocess_input, _obtain_input_shape
import numpy as np
import pdb
import glob
import os
import matplotlib.pyplot as plt

for sbj in ["a0f", "cb4", "c95", "d65"]:
    loss = selected_loss(input=np.zeros(shape=(1,1,1,1), dtype='float32'))
    #main_ecog_dir = '/data2/users/wangnxr/dataset/ecog_mni_ellip_%s' % (sbj)
    main_ecog_dir = '/home/wangnxr/standardized_clips_ellip/%s*/' % sbj
    model_file =  "/home/wangnxr/models/ecog_model_ellip_impute_sequence_bothloss_skip_10output_allplus_itr_0_3d_best.h5"
    ## Data generation ECoG
    channels = np.arange(100)
    model = load_model(model_file, custom_objects={"loss":loss})
    files = glob.glob("%s/val/*.npy" % main_ecog_dir)
    for file in sorted(files):
	print file
	orig = np.load(file)
	orig2 = copy(orig)
	orig[np.where(orig[:,0]!=0)[0][5:10]] = 0
	new = copy(orig)
	
	fill_inds = np.where(orig[:,0]==0)[0]
	orig_batch = np.zeros(shape=(orig.shape[-1],1,10,10, 80))
	for t in xrange(80, orig.shape[-1]):
	    orig_batch[t] = np.reshape(orig[:,t-80:t], (10,10,80))
	new[fill_inds] = model.predict(orig_batch)[:,9::10][:,fill_inds].T
        #new = model.predict(orig_batch).T
	try:
  	    np.save("%s/ecog_mni_ellip_deep_impute_analyallsbjskipbothloss10output_%s/%s" % ("/".join(main_ecog_dir.split("/")[:-1]), sbj, "/".join(file.split("/")[-3:])), new)
	except IOError:
	    os.makedirs("%s/ecog_mni_ellip_deep_impute_analyallsbjskipbothloss10output_%s/%s" % ("/".join(main_ecog_dir.split("/")[:-1]), sbj, "/".join(file.split("/")[-3:-1])))
	    np.save("%s/ecog_mni_ellip_deep_impute_analyallsbjskipbothloss10output_%s/%s" % ("/".join(main_ecog_dir.split("/")[:-1]), sbj, "/".join(file.split("/")[-3:])), new)


