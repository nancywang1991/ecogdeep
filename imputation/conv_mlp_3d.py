import keras
from keras import backend as K
from ecogdeep.data.preprocessing.ecog_for_imputation import EcogDataGenerator
from keras.layers import Flatten, Dense, Input, Dropout, Activation, merge, TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Convolution3D, MaxPooling3D, AveragePooling3D

# from keras.imagenet_utils import decode_predictions, preprocess_input, _obtain_input_shape
import numpy as np
import pdb
import pickle
import glob
import tensorflow as tf

def selected_loss(input):
    def loss(y_true, y_pred):
	#pdb.set_trace()
        #inds = K.cast(K.not_equal(K.reshape(input[:,0,:,:,-1], (-1,100)), y_true), 'float32')
	inds = K.cast(K.not_equal(y_true, 0), 'float32')
	return K.sum(K.square((y_pred - y_true)*inds), axis=-1)/K.sum(inds)
    return loss

def main():
    sbj_to_do = ["all"]
    for itr in range(1):
        for s, sbj in enumerate(sbj_to_do):
            main_ecog_dir = '/data2/users/wangnxr/dataset/standardized_clips_ellip/' 
	    main_ecog_dir2 = '/data2/users/wangnxr/dataset/ecog_mni_ellipv2_%s/' % ('a0f_d65_cb4_c95')
            ## Data generation ECoG
            channels = np.arange(100)
            train_datagen_edf = EcogDataGenerator(
                seq_len=80,
		three_d = True,
		test = False	
            )

            test_datagen_edf = EcogDataGenerator(
                seq_len=80,
		three_d = True,
		test = True
            )

            dgdx_edf = train_datagen_edf.flow_from_directory(
            '%s/train/' % main_ecog_dir2,
            batch_size=24,
            channels=channels,
            ablate_range = (3,10),
            pre_shuffle_ind=1,
	    spatial_shift=True
            )

            dgdx_val_edf = test_datagen_edf.flow_from_directory(
            '%s/val/' % main_ecog_dir2,
            batch_size=1,
	    ablate_range = (3,10),
            channels=channels)
            
            train_generator = dgdx_edf
            validation_generator = dgdx_val_edf
	    test = train_generator.next()
	    ecog_series = Input(shape=(1, 10,10, 80))
            x = Convolution3D(16, (1, 1, 3), padding='same', name='block1_conv1')(ecog_series)
            # x = BatchNormalization(axis=1)(x)
            x = Activation('tanh')(x)
            x = MaxPooling3D((2, 2, 1), name='block1_pool')(x)
	   
            # Block 2
            x = Convolution3D(32, (1, 1, 5), padding='same', name='block2_conv1')(x)
            # x = BatchNormalization(axis=1)(x)
            x = Activation('tanh')(x)
            #x = MaxPooling2D((1, 3), name='block2_pool')(x)

            # Block 3
            x = Convolution3D(64, (1, 1, 9), padding='same', name='block3_conv1')(x)
            # x = BatchNormalization(axis=1)(x)
            x = Activation('tanh')(x)
	    x = Flatten(name='flatten')(x)
            x = Dense(256, name='fc1')(x)
            x = Activation('tanh')(x)
	    x = Dense(128, name='fc2')(x)
	    x = Activation('tanh')(x)
            x = Dense(len(channels), name='predictions')(x)
            # x = BatchNormalization()(x)
            predictions = Activation('linear')(x)

            model = Model(inputs=[ecog_series], outputs=predictions)
            sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9)
            model_savepath = "/home/wangnxr/models/ecog_model_ellipv2_impute_allloss_long80_%s_itr_%i_3d" % (sbj, itr)
            model.compile(optimizer=sgd,
                      loss=[selected_loss(input=ecog_series)])
            early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=10, verbose=0, mode='auto')
            checkpoint = ModelCheckpoint("%s_best.h5" % model_savepath, monitor='val_loss', verbose=1,
                                     save_best_only=True, mode='min')
            history_callback = model.fit_generator(
            train_generator,
            steps_per_epoch=len(dgdx_edf.filenames)/24,
            epochs=60,
            validation_data=validation_generator,
            validation_steps=len(dgdx_val_edf.filenames), callbacks=[checkpoint]
	    )

            model.save("%s.h5" % model_savepath)
            pickle.dump(history_callback.history,
                    open("/home/wangnxr/history/%s.p" % model_savepath.split("/")[-1].split(".")[0], "wb"))

if __name__ == "__main__":
    main()
