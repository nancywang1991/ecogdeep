import keras
from keras.layers import Flatten, Dense, Input, Dropout, Activation, merge
from keras.preprocessing.image_reg import ImageDataGenerator, center_crop
from keras.models import Model
from ecogdeep.train.ecog_1d_model_reg import ecog_1d_model
from keras.preprocessing.ecog_reg_xy import EcogDataGenerator
from ecogdeep.train.vid_model_reg import vid_model
from keras.callbacks import ModelCheckpoint
from sbj_parameters import *

#from keras.imagenet_utils import decode_predictions, preprocess_input, _obtain_input_shape
import numpy as np
import pdb
import pickle
import glob
import time

sbj_to_do = ["a0f", "cb4"]
for s, sbj in enumerate(sbj_ids):
    if sbj in sbj_to_do:
        main_vid_dir = '/home/wangnxr/dataset_xy_reg/ecog_vid_combined_%s_day%i/' % (sbj, days[s])
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

        train_generator = dgdx_vid
        validation_generator = dgdx_val_vid

        base_model_vid = Model(video_model.input, video_model.get_layer("flatten").output)

        frame_a = Input(shape=(3,224,224))

        for layer in base_model_vid.layers:
            layer.trainable = True

        sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9)

        predictions = base_model_vid(frame_a)

        model = Model(input=frame_a, output=predictions)
        model_savepath = "/home/wangnxr/models/vid_model_%s_itr_%i_reg" % (sbj, itr)
        model.compile(optimizer=sgd,
                      loss='mean_squared_error')
        checkpoint = ModelCheckpoint(model_savepath + "_" + "{epoch:02d}" + "_chkpt.h5", monitor='val_loss', verbose=1, save_best_only=False, mode='min', period=30)
        history_callback = model.fit_generator(
            train_generator,
            samples_per_epoch=len(dgdx_vid.filenames),
            nb_epoch=100,
            validation_data=validation_generator,
            nb_val_samples=len(dgdx_val_vid.filenames), callbacks=[checkpoint])

        model.save("%s.h5" % model_savepath)
	model.save_weights("%s_weights.h5" % model_savepath)
        pickle.dump(history_callback.history, open("/home/wangnxr/history/vid_model_%s_itr_%i_reg.txt" % (sbj, itr), "wb"))
    time.sleep(50)

