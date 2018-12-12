import keras
from ecogdeep.data.preprocessing.ecog_sequence import EcogDataGenerator
from keras.layers import Flatten, Dense, Input, Dropout, Activation, merge, TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.regularizers import l2
from itertools import izip
from keras.callbacks import ModelCheckpoint, EarlyStopping
from ecogdeep.train.ecog_1d_model_seq import ecog_1d_model
from sbj_parameters import *

#from keras.imagenet_utils import decode_predictions, preprocess_input, _obtain_input_shape
import numpy as np
import pdb
import pickle
import glob

#PARAMS
#sbj_to_do = ["a0f", "d65", "cb4"]
sbj_to_do = ["a0f_d65"]
jitter = True
imputation_type = "deep"
data_dir = "/data2/users/wangnxr/dataset/"
model_dir = "/home/wangnxr/models/"
history_dir = "/home/wangnxr/history/"

for itr in range(1):
    for s, sbj in enumerate(sbj_to_do):
        if imputation_type == "zero":
            main_ecog_dir = '/%s/ecog_mni_ellip_%s/' % (data_dir, sbj)
        if imputation_type == "interp":
            main_ecog_dir = '/%s/ecog_mni_ellip_interp_%s/' % (data_dir, sbj)
        if imputation_type == "deep":
            main_ecog_dir = '/%s/ecog_mni_ellip_deep_impute_%s/' % (data_dir, sbj)

        # Data Param Prep
        classes_dict = {cl.split("/")[-1]:c for c, cl in enumerate(sorted(glob.glob(main_ecog_dir + "/train/*")))}
        train_IDs = glob.glob(main_ecog_dir + "/train/*/*.npy")
        val_IDs = glob.glob(main_ecog_dir + "/val/*/*.npy")
        np.random.shuffle(train_IDs)
        np.random.shuffle(val_IDs)
        train_label = {ID:classes_dict[ID.split("/")[-2]] for ID in train_IDs}
        val_label = {ID:classes_dict[ID.split("/")[-2]] for ID in val_IDs}
        for t, time in enumerate(start_times):
            print sbj
            print time
            ## Data generation ECoG
            train_datagen_edf = EcogDataGenerator(
                train_IDs,
                train_label,
                dim = (100, 1000),
                time_shift_range=200,
                center=False,
                seq_len=200,
                start_time=time,
                seq_num = 5,
                seq_st = 200,
                batch_size = 24,
                spatial_shift = jitter,
                shuffle=True,
                n_classes = len(classes_dict),
                n_channels = 1
            )

            test_datagen_edf = EcogDataGenerator(
                val_IDs,
                val_label,
                dim = (100,1000),
                time_shift_range=200,
                center=True,
                seq_len=200,
                start_time=time,
                seq_num=5,
                batch_size = 10,
                seq_st=200,
                spatial_shift = False,
                shuffle=False,
                n_classes = len(classes_dict),
                n_channels = 1

            )
            ecog_model = ecog_1d_model(channels=100)
            base_model_ecog = Model(ecog_model.input, ecog_model.get_layer("fc1").output)
            ecog_series = Input(shape=(5, 1, 100, 200))

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
            model_savepath = "%s/ecog_model_mni_ellip_%s_sequence_%s_itr_%i_t_%i" % (model_dir,imputation_type, sbj,itr,time)

            model.compile(optimizer=sgd,
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
            early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=10, verbose=0, mode='auto')
            checkpoint = ModelCheckpoint("%s_best.h5" % model_savepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
            history_callback = model.fit_generator(
                train_datagen_edf,
                epochs=200,
                validation_data=test_datagen_edf,
                callbacks=[checkpoint, early_stop],
                workers=8,
                use_multiprocessing=True
            )

            model.save("%s.h5" % model_savepath)
            pickle.dump(history_callback.history, open("%s/%s.p" % (history_dir, model_savepath.split("/")[-1].split(".")[0]), "wb"))
