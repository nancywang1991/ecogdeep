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
    main_ecog_dir = '/data2/users/wangnxr/dataset/ecog_mni_%s' % (sbj)
    model_file = "/home/wangnxr/models/ecog_model_impute_ablate_more_a0f_d65_c95_cb4_itr_0_best.h5"
    ## Data generation ECoG
    channels = np.arange(100)
    model = load_model(model_file, custom_objects={"loss":loss})
    files = glob.glob("%s/*/*/*.npy" % main_ecog_dir)
    for file in files:
	print file
	orig = np.load(file)
	new = copy(orig)
	fill_inds = np.where(orig[:,0]==0)[0]
	orig_batch = np.zeros(shape=(orig.shape[-1],1,orig.shape[0], 20))
	for t in xrange(20, orig.shape[-1]):
	    orig_batch[t] = orig[:,t-20:t]
	new[fill_inds] = model.predict(orig_batch)[:,fill_inds].T
        try:
  	    np.save("%s/ecog_mni_deep_impute_%s/%s" % ("/".join(main_ecog_dir.split("/")[:-1]), sbj, "/".join(file.split("/")[-3:])), new)
	except IOError:
	    os.makedirs("%s/ecog_mni_deep_impute_%s/%s" % ("/".join(main_ecog_dir.split("/")[:-1]), sbj, "/".join(file.split("/")[-3:-1])))
	    np.save("%s/ecog_mni_deep_impute_%s/%s" % ("/".join(main_ecog_dir.split("/")[:-1]), sbj, "/".join(file.split("/")[-3:])), new)


