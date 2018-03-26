import keras
from keras.layers import Flatten, Dense, Input, Dropout, Activation, merge
from keras.preprocessing.image_reg import ImageDataGenerator, center_crop
from keras.models import Model
from ecogdeep.train.ecog_1d_model_reg import ecog_1d_model
from keras.preprocessing.ecog_reg_xy import EcogDataGenerator
from ecogdeep.train.vid_model_reg import vid_model
from keras.callbacks import ModelCheckpoint
from sbj_parameters import *
from keras.layers import Convolution2D, RepeatVector, UpSampling2D, LocallyConnected2D, Reshape

#from keras.imagenet_utils import decode_predictions, preprocess_input, _obtain_input_shape
import numpy as np
import pdb
import pickle
import glob
import time
import h5py

sbj_to_do = ["cb4"]
for s, sbj in enumerate(sbj_ids):
    if sbj in sbj_to_do:
        main_vid_dir = '/home/wangnxr/dataset_xy_reg/ecog_vid_combined_%s_day%i/' % (sbj, days[s])
        main_ecog_dir = '/home/wangnxr/dataset_xy_reg/ecog_vid_combined_%s_day%i/' % (sbj, days[s])
    else:
        continue
    for itr in range(1):
        times = [3500]

        # Video data generators
        train_datagen_vid = ImageDataGenerator(
            #rotation_range=40,
            rescale=1./255,
            #zoom_range=0.2,
            #horizontal_flip=True,
            start_time=times,
            center_crop=(224,224))

        test_datagen_vid = ImageDataGenerator(
            rescale=1./255,
            start_time=times[0],
            center_crop=(224, 224))


        video_model = vid_model()


        dgdx_vid = train_datagen_vid.flow_from_directory(
            '/%s/train/' % main_vid_dir,
            read_formats={'png'},
            target_size=(int(224), int(224)),
            num_frames=12,
            batch_size=24,
            class_mode='binary',
            shuffle=False,
            pre_shuffle_ind=1)


        dgdx_val_vid = test_datagen_vid.flow_from_directory(
            '/%s/val/' % main_vid_dir,
            read_formats={'png'},
            target_size=(int(224), int(224)),
            num_frames=12,
            batch_size=10,
            shuffle=False,
            class_mode='binary')

        ## Data generation ECoG
        channels = channels_list[sbj_ids.index(sbj)]

        train_datagen_edf = EcogDataGenerator(
            time_shift_range=200,
            gaussian_noise_range=0.001,
            center=False,
            start_time=times,
        )

        test_datagen_edf = EcogDataGenerator(
            time_shift_range=200,
            center=True,
            start_time=times[0],
        )

        dgdx_edf = train_datagen_edf.flow_from_directory(
            #'/mnt/cb46fd46_5_no_offset/train/',
            '%s/train/' % main_ecog_dir,
            batch_size=24,
            target_size=(1,len(channels),1000),
            final_size=(1,len(channels),1000),
            class_mode='binary',
            shuffle=False,
            channels = channels,
            pre_shuffle_ind=1)

        dgdx_val_edf = test_datagen_edf.flow_from_directory(
            #'/mnt/cb46fd46_5_no_offset/test/',
            '%s/val/' % main_ecog_dir,
            batch_size=10,
            shuffle=False,
            target_size=(1,len(channels),1000),
            final_size=(1,len(channels),1000),
            channels = channels,
            class_mode='binary')


        def izip_input(gen1, gen2):
            while 1:
                #pdb.set_trace()
                x1, y1 = gen1.next()
                x2, y2 = gen2.next()
                if not x1.shape[0] == x2.shape[0]:
                    pdb.set_trace()
                yield [x1, x2], [y1, y2[0]]
		#yield x1, y1

        ecog_model = ecog_1d_model(channels=len(channels))
        base_model_ecog1 = Model(ecog_model.input, ecog_model.get_layer('fc1a').output, name="base_direction_model")
	base_model_ecog2 = Model(ecog_model.input, ecog_model.get_layer('fc1b').output, name="base_magnitude_model")
        ecog_series = Input(shape=(1,len(channels),1000), name="ecog_input")
        train_generator = izip_input(dgdx_vid, dgdx_edf)
        validation_generator = izip_input(dgdx_val_vid, dgdx_val_edf)

	
        #base_model_vid = Model(video_model.input, video_model.get_layer('block8_conv1').output)
	base_model_vid = Model(video_model.input, video_model.get_layer('flatten').output, name="base_vid_model")
        frame_a = Input(shape=(3,224,224), name="video_input")
	#pdb.set_trace()
        #for layer in base_model_vid.layers:
        #    layer.trainable = False
	
        tower1 = base_model_vid(frame_a)
        tower2 = base_model_ecog1(ecog_series)
        predictions2 = tower1
        tower1 = Dense(1024, name='tower1_dense')(tower1)
	tower1 = Activation('relu')(tower1)
	tower1 = Dense(128, name='tower1_dense2')(tower1)
        tower1 = Activation('relu')(tower1)
	#tower2 = RepeatVector(1)(tower2)
	#tower2 = Reshape((1,1,1))(tower2)
        #tower2 = UpSampling2D((56, 56))(tower2)
        #tower3 = base_model_ecog2(ecog_series)
        #tower3 = RepeatVector(1)(tower3)
        #tower3 = Reshape((1,1,1))(tower3)
	#tower3 = UpSampling2D((56, 56))(tower3)
        #pdb.set_trace()
	x = merge([tower1, tower2], mode='concat')
        #x = Convolution2D(256, 9, 9, name='block7_lc1', border_mode='same')(x)
        #x = Activation('tanh')(x)
        #x = Convolution2D(256, 3, 3, name='block8_lc1', border_mode='same')(x)
        #x = Activation('tanh')(x)
        #x = Convolution2D(1, 1, 1, border_mode='same', name='block10_conv1')(x)
        # x = BatchNormalization(axis=1)(x)
        #x = Activation('relu')(x)
        x = Activation('relu')(x)
	x = Dense(64, name='Dense_merged1')(x)
        predictions = Dense(1, name='predictions')(x)
        #x = tower1
        #predictions = Dense(3136, name='predictions', init='normal')(x)

        sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9)
	#pdb.set_trace()
        #model = Model(input=[frame_a, ecog_series], output=predictions)
	model = Model(input=[frame_a, ecog_series], output=[predictions2, predictions], name="main")
	model.load_weights("/home/wangnxr/models/ecog_vid_model_%s_itr_0_ecog_freeze_reg_v5.h5" % sbj)
	model_savepath = "/home/wangnxr/models/ecog_vid_model_%s_itr_%i_no_freeze_reg_v5" % (sbj, itr)
	model.compile(optimizer=sgd,
                      loss='mean_squared_error', loss_weights=[1,0.01])
        checkpoint = ModelCheckpoint(model_savepath + "_chkpt.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        history_callback = model.fit_generator(
            train_generator,
            samples_per_epoch=len(dgdx_vid.filenames),
            nb_epoch=500,
            validation_data=validation_generator,
            nb_val_samples=len(dgdx_val_vid.filenames), callbacks=[checkpoint])

        model.save("%s.h5" % model_savepath)
        pickle.dump(history_callback.history, open("/home/wangnxr/history/ecog_vid_history_%s_itr_%i_no_freeze_reg_v5.txt" % (sbj, itr), "wb"))
        #time.sleep(50)
