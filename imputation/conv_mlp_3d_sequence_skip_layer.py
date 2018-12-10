import keras
from keras import backend as K
from ecogdeep.data.preprocessing.ecog_sequence_for_imputation import EcogDataGenerator
from keras.layers import Flatten, Dense, Input, Dropout, Activation, merge, TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import concatenate, Lambda, Convolution3D, MaxPooling3D, AveragePooling3D

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
	inds = K.cast(K.not_equal(y_true, 0), 'int32')
	x = K.gather(y_true, inds)
    	y = K.gather(y_pred, inds)
        mx = K.mean(x)
    	my = K.mean(y)
    	xm, ym = x-mx, y-my
    	r_num = K.sum(tf.multiply(xm,ym))
    	r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    	r = r_num / r_den

    	r = K.maximum(K.minimum(r, 1.0), -1.0)
    	return 1-r + K.mean(K.square((y_pred - y_true)))
	#return K.sum(K.square((y_pred - y_true)*inds), axis=-1)/K.sum(inds, axis=-1)
    return loss

def main():
    sbj_to_do = ["allplus"]

    for itr in range(1):
        for s, sbj in enumerate(sbj_to_do):
            main_ecog_dir = '/home/wangnxr/standardized_clips_ellip/' 
            ## Data generation ECoG
	    train_IDs = glob.glob(main_ecog_dir + "/*/train/*.npy")
	    val_IDs = glob.glob(main_ecog_dir + "/*/val/*.npy")
	    np.random.shuffle(train_IDs)
	    np.random.shuffle(val_IDs)
            train_generator = EcogDataGenerator(
		train_IDs,	
		test = False,
		dim = (10,10),
		seq_len = 80,
		output_len = 10,
		ablate_range = (3,10),
		spatial_shift=True,
		batch_size = 48,
            )

            validation_generator = EcogDataGenerator(
		val_IDs,
		test = True,
		dim = (10,10),
                seq_len = 80,
		output_len = 10,
                ablate_range = (3,10),
                spatial_shift=False,
                batch_size = 1,

            )
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
	    x2 = Lambda(lambda x : x[:,:,:,-1:])(ecog_series)
            x2 = Flatten(name='flatten_slice')(x2)
            x = Flatten(name='flatten')(x)
            x3 = concatenate([x,x2], name='concat')
	    x3 = Dropout(0.2)(x3)
            x3 = Dense(256, name='fc1')(x3)
            x3 = Activation('tanh')(x3)
	    x3 = Dropout(0.2)(x3)
            x3 = Dense(128, name='fc2')(x3)
            x3 = Activation('tanh')(x3)
	    x3 = Dropout(0.2)(x3)
            x3 = Dense(1000, name='predictions')(x3)
	    predictions = Activation('linear')(x3)
            model = Model(inputs=[ecog_series], outputs=predictions)
            sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9)
            model_savepath = "/home/wangnxr/models/ecog_model_ellip_impute_sequence_bothloss_skip_10output_%s_itr_%i_3d" % (sbj, itr)
            model.compile(optimizer=sgd,
                      loss=[selected_loss(input=ecog_series)])
            early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=10, verbose=0, mode='auto')
            checkpoint = ModelCheckpoint("%s_best.h5" % model_savepath, monitor='val_loss', verbose=1,
                                     save_best_only=True, mode='min')
            history_callback = model.fit_generator(
            train_generator,
            epochs=60,
            validation_data=validation_generator,
            callbacks=[checkpoint],
	    workers = 8,
	    use_multiprocessing=True
	    )

            model.save("%s.h5" % model_savepath)
            pickle.dump(history_callback.history,
                    open("/home/wangnxr/history/%s.p" % model_savepath.split("/")[-1].split(".")[0], "wb"))

if __name__ == "__main__":
    main()
