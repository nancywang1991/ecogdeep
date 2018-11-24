import keras
from ecogdeep.data.preprocessing.ecog import EcogDataGenerator
from ecogdeep.data.preprocessing.image2 import ImageDataGenerator, center_crop
from keras.layers import Flatten, Dense, Input, Dropout, Activation, merge, TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.models import Model
from hyperopt import Trials, fmin, tpe, hp, STATUS_OK
from keras.regularizers import l2
from itertools import izip
from keras.callbacks import ModelCheckpoint, EarlyStopping
from ecogdeep.train.ecog_3d_model import ecog_3d_model
from sbj_parameters import *

#from keras.imagenet_utils import decode_predictions, preprocess_input, _obtain_input_shape
import numpy as np
import pdb
import pickle
import glob

#PARAMS
sbj_to_do = ["a0f", "c95", "cb4", "d65", "a0f_d65"]
sbj_to_do = ["d65", "a0f_d65"]
jitter = True
imputation_type = "interp"
data_dir = "/data2/users/wangnxr/dataset/"
model_dir = "/home/wangnxr/models/"
history_dir = "/home/wangnxr/history/"

for itr in range(1):
    for s, sbj in enumerate(sbj_to_do):
	if imputation_type == "zero":
		main_ecog_dir = '/%s/ecog_mni_ellipv2_%s/' % (data_dir, sbj)
	if imputation_type == "interp":
		main_ecog_dir = '/%s/ecog_mni_ellipv2_interp_%s/' % (data_dir, sbj)
	if imputation_type == "deep":
        	main_ecog_dir = '/%s/ecog_mni_ellipv2_deep_impute_%s/' % (data_dir, sbj)

        for t, time in enumerate(start_times[2:]):
	    print sbj
	    print time
            ## Data generation ECoG
            channels = np.arange(100)
            train_datagen_edf = EcogDataGenerator(
                time_shift_range=200,
                center=False,
                seq_len=200,
                start_time=time,
                seq_num = 5,
                seq_st = 200,
		spatial_shift = jitter,
                three_d = True
            )

            test_datagen_edf = EcogDataGenerator(
                time_shift_range=200,
                center=True,
                seq_len=200,
                start_time=time,
                seq_num=5,
                seq_st=200,
                three_d = True
            )
            
            dgdx_edf = train_datagen_edf.flow_from_directory(
                '%s/train/' % main_ecog_dir,
                batch_size=24,
                class_mode='binary',
                shuffle=False,
                channels = channels,
                pre_shuffle_ind=1,
                target_size=(1, len(channels), 1000),
                final_size=(1,len(channels),200),
                )

            dgdx_val_edf = test_datagen_edf.flow_from_directory(
                '%s/val/' % main_ecog_dir,
                batch_size=10,
                shuffle=False,
                target_size=(1, len(channels), 1000),
                final_size=(1,len(channels),200),
                channels = channels,
                class_mode='binary')
            
	    ecog_model = ecog_3d_model(channels=len(channels))
            train_generator =  dgdx_edf
            validation_generator =  dgdx_val_edf
            base_model_ecog = Model(ecog_model.input, ecog_model.get_layer("fc1").output)

            ecog_series = Input(shape=(5,1,10,10,200))

            x = base_model_ecog(ecog_series)

            x = Dropout(0.5)(x)
            x = TimeDistributed(Dense(32, kernel_regularizer=l2(0.01), name='merge2'))(x)
            #x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Dropout(0.5)(x)
            x = LSTM(20, dropout=0.2, recurrent_dropout=0.2, name='lstm')(x)
            x = Dense(1, name='predictions')(x)
            #x = BatchNormalization()(x)
            predictions = Activation('sigmoid')(x)

            for layer in base_model_ecog.layers:
                layer.trainable = True


            model = Model(inputs=[ecog_series], outputs=predictions)

            sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9)

            model_savepath = "%s/ecog_model_mni_ellipv2_%s_jitter_%s_%s_itr_%i_t_%i_3d" % (model_dir,imputation_type, jitter, sbj,itr,time)
            
            model.compile(optimizer=sgd,
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
            early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=10, verbose=0, mode='auto')
            checkpoint = ModelCheckpoint("%s_best.h5" % model_savepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
            history_callback = model.fit_generator(
                train_generator,
                steps_per_epoch=len(dgdx_edf.filenames)/24,
                epochs=200,
                validation_data=validation_generator,
                validation_steps=len(dgdx_val_edf.filenames)/10, 
		callbacks=[checkpoint, early_stop]
	    )

            model.save("%s.h5" % model_savepath)
            pickle.dump(history_callback.history, open("%s/%s.p" % (history_dir, model_savepath.split("/")[-1].split(".")[0]), "wb"))
