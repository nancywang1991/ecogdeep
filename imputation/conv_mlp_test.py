import keras
from ecogdeep.data.preprocessing.ecog_for_imputation import EcogDataGenerator
from keras.layers import Flatten, Dense, Input, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model, load_model
import numpy as np
import pdb
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv
from keras.utils.generic_utils import get_custom_objects
from ecogdeep.imputation.conv_mlp import selected_loss
import tensorflow as tf
import scipy
import glob
import copy
"""Accuracy of imputation test set using ECoG model.

"""

sbj_to_do = ["a0f", "d65", "a0f", "cb4", "c95"]

def interpolate_data(orig_old):
    orig = copy.copy(orig_old)
    valid = np.where(orig[0,:]!=0)[0]
    gridinds = np.array([valid/10, valid%10])
    gridx, gridy = np.mgrid[0:10,0:10]
    for b in xrange(orig.shape[0]):
        orig[b,:] = np.ndarray.flatten(scipy.interpolate.griddata(gridinds.T, orig[b,valid], (gridx, gridy), fill_value=0, method='cubic'))

    valid = np.where(orig[0,:]!=0)[0]
    gridinds = np.array([valid/10, valid%10])
    for b in xrange(orig.shape[0]):
        orig[b,:] = np.ndarray.flatten(scipy.interpolate.griddata(gridinds.T, orig[b,valid], (gridx, gridy), method='nearest'))
    return orig


for s, sbj in enumerate(sbj_to_do):
    print sbj
    main_ecog_dir = '/data2/users/wangnxr/dataset/ecog_mni_ellipv2_%s/' % (sbj)
    #main_ecog_dir = '/data2/users/wangnxr/dataset/standardized_clips/'
    loss = selected_loss(input=np.zeros(shape=(1,1,1,1), dtype='float32'))

    model_file =  "/home/wangnxr/models/ecog_model_ellipv2_impute_all_plus2_itr_0_3d_best.h5"
    model_file2 = "/home/wangnxr/models/ecog_model_ellipv2_impute_jitter_True_all_plus_itr_0_best.h5" 
    model_file3 = "/home/wangnxr/models/ecog_model_ellipv2_impute_all_itr_0_3d_best.h5"
    model = load_model(model_file, custom_objects={'loss':loss})
    model2 = load_model(model_file2, custom_objects={'loss':loss})
    model3 = load_model(model_file3, custom_objects={'loss':loss})
    ## Data generation ECoG
    channels = np.arange(100)

    test_datagen_edf = EcogDataGenerator(
            seq_len=20,
	    test = True
        )


    dgdx_val_edf = test_datagen_edf.flow_from_directory(
            '%s/test/' % main_ecog_dir,
            batch_size=len(glob.glob( '%s/test/*/*' % main_ecog_dir)),
            ablate_range = (1,2),
            channels=channels)

    validation_generator = dgdx_val_edf
    files = dgdx_val_edf.filenames
    total_dist = 0
    predictions = []
    predictions2 = []
    predictions3 = []
    averages = []
    test = validation_generator.next()
    test2 = (np.array([x.reshape(1,10,10,20) for x in test[0]]), test[1])
    prediction = model.predict(test2[0])
    prediction2 = model2.predict(test[0][:,:,:,-1:])
    prediction3 = model3.predict(test2[0])
    interp = interpolate_data(test[0][:,0,:,-1])
    #prediction3 = model3.predict(test2[0][:,:,:,:,-8:])

    for b in xrange(len(files)):
	inds = np.where(test[0][b,0,:,-1] != test[1][b])[0]
	#inds = np.where(test[1][b]!=0)[0][0]
	#print "sample" + str(b)
	#print prediction[b][inds]
	#print prediction2[b][inds]
	#print test[1][b][inds]
	#print np.mean(test[1][b][np.where(test[1][b]!=0)])
	pdb.set_trace()
   	predictions = predictions + list(np.abs( test2[1][b][inds]- prediction[b][inds]))
	predictions2 = predictions2 + list(np.abs( test2[1][b][inds]- prediction2[b][inds]))
	predictions3 = predictions3 + list(np.abs( test2[1][b][inds]- prediction3[b][inds]))
	averages = averages + list(np.abs( test2[1][b][inds]- interp[b][inds]))

    print "model1: %f" % np.mean(predictions)
    print "model2: %f" % np.mean(predictions2)
    print "model3: %f" % np.mean(predictions3)
    print "interpolation: %f" % np.mean(averages)
