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

"""Accuracy of imputation test set using ECoG model.

"""

sbj_to_do = ["a0f", "d65", "cb4", "c95"]
for s, sbj in enumerate(sbj_to_do):
    print sbj
    main_ecog_dir = '/data2/users/wangnxr/dataset/ecog_mni_ellipv2_%s/' % (sbj)
    #main_ecog_dir = '/data2/users/wangnxr/dataset/standardized_clips/'
    loss = selected_loss(input=np.zeros(shape=(1,1,1,1), dtype='float32'))

    model_file3 =  "/home/wangnxr/models/ecog_model_ellipv2_impute_jitter_True_all_plus_itr_0_3d_best.h5"
    model_file2 = "/home/wangnxr/models/ecog_model_ellipv2_impute_jitter_True_all_itr_0_3d_best.h5" 
    model_file = "/home/wangnxr/models/ecog_model_ellipv2_impute_all_itr_0_3d_best.h5"
    model = load_model(model_file, custom_objects={'loss':loss})
    model2 = load_model(model_file2, custom_objects={'loss':loss})
    model3 = load_model(model_file3, custom_objects={'loss':loss})
    ## Data generation ECoG
    channels = np.arange(100)

    test_datagen_edf = EcogDataGenerator(
            seq_len=20
        )


    dgdx_val_edf = test_datagen_edf.flow_from_directory(
            '%s/test/' % main_ecog_dir,
            batch_size=500,
            ablate_range = (1,2),
            channels=channels)

    test_datagen_edf2 = EcogDataGenerator(
            seq_len=20,
	    three_d = True
        )

    dgdx_val_edf2 = test_datagen_edf2.flow_from_directory(
            '%s/test/' % main_ecog_dir,
            batch_size=500,
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
    prediction2 = model2.predict(test2[0])
    prediction3 = model3.predict(test2[0])
    #prediction3 = model3.predict(test2[0][:,:,:,:,-8:])

    for b in xrange(50):
	inds = np.where(test[0][b,0,:,-1] != test[1][b])[0][0]
	#print "sample" + str(b)
	#print prediction[b][inds]
	#print prediction2[b][inds]
	#print test[1][b][inds]
	#print np.mean(test[1][b][np.where(test[1][b]!=0)])
   	predictions.append(np.abs( test2[1][b][inds]- prediction[b][inds]))
	predictions2.append(np.abs( test2[1][b][inds]- prediction2[b][inds]))
	predictions3.append(np.abs( test2[1][b][inds]- prediction3[b][inds]))
        grid_loc = (inds/10, inds%10)
        naive_avg = []
	#pdb.set_trace()
        for loc_x in range(grid_loc[0]-1, grid_loc[0]+2):
	    for loc_y in range(grid_loc[1]-1, grid_loc[1]+2):
		new_ind = loc_x*10+loc_y
		if (not new_ind==inds) and new_ind >=0 and new_ind < 100 and test[1][b][new_ind]!=0:
			naive_avg.append(test[1][b][new_ind])

	if len(naive_avg) == 0:
           print "none nearby"
	   averages.append(np.abs(np.mean(test[1][b][np.where(test[1][b]!=0)]) - test[1][b][inds]))
        else:
	   averages.append(np.abs(np.mean(naive_avg)-test[1][b][inds]))

    print "model1: %f" % np.mean(predictions)
    print "model2: %f" % np.mean(predictions2)
    print "model3: %f" % np.mean(predictions3)
    print "interpolation: %f" % np.mean(averages)
