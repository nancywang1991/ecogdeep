import keras
from keras.preprocessing.ecog import EcogDataGenerator
from keras.preprocessing.image2 import ImageDataGenerator, center_crop
from keras.layers import Flatten, Dense, Input, Dropout, Activation, merge, TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.models import Model
from hyperopt import Trials, fmin, tpe, hp, STATUS_OK
from keras.regularizers import l2
from itertools import izip
from keras.callbacks import ModelCheckpoint, EarlyStopping
from ecogdeep.train.ecog_1d_model_seq import ecog_1d_model
import ecogdeep.train.vid_model_seq as vid_model_seq
from sbj_parameters import *
import pickle

sbj_to_do = ["c95"]

for itr in xrange(3,4):
    for s, sbj in enumerate(sbj_ids):
        if sbj in sbj_to_do:
            main_vid_dir = '/home/wangnxr/dataset/ecog_vid_combined_%s_day%i/' % (sbj, days[s])
        else:
            continue

        for t, time in enumerate(start_times):
	    if time == 2700:
		continue
            # Video data generators
            train_datagen_vid = ImageDataGenerator(
                rescale=1./255,
                random_black=False,
                random_crop=(224,224),
                keep_frames=frames[t])

            test_datagen_vid = ImageDataGenerator(
                random_black=False,
                rescale=1./255,
                center_crop=(224, 224),
                keep_frames=frames[t])

            vid_model = vid_model_seq.vid_model()

            dgdx_vid = train_datagen_vid.flow_from_directory(
                '/%s/train/' % main_vid_dir,
                img_mode="seq",
                read_formats={'png'},
                target_size=(int(224), int(224)),
                num_frames=12,
                batch_size=24,
                class_mode='binary',
                shuffle=False,
                pre_shuffle_ind=1)

            dgdx_val_vid = test_datagen_vid.flow_from_directory(
                '/%s/val/' % main_vid_dir,
                img_mode="seq",
                read_formats={'png'},
                target_size=(int(224), int(224)),
                num_frames=12,
                batch_size=10,
                shuffle=False,
                class_mode='binary')

            train_generator = dgdx_vid
            validation_generator = dgdx_val_vid
            base_model_vid = Model(vid_model.input, vid_model.get_layer("fc1").output)

            frame_a = Input(shape=(5,3,224,224))


            x = base_model_vid(frame_a)
            x = Dropout(0.5)(x)
            x = TimeDistributed(Dense(256, W_regularizer=l2(0.01), name='merge2'))(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Dropout(0.5)(x)
            x = LSTM(20, dropout_W=0.2, dropout_U=0.2, name='lstm')(x)
            x = Dense(1, name='predictions')(x)
            #x = BatchNormalization()(x)
            #predictions = Dense(2, activation='softmax', name='predictions')(x)
            predictions = Activation('sigmoid')(x)

            for layer in base_model_vid.layers:
                layer.trainable = True

            model = Model(input=frame_a, output=predictions)

            sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9)

            model_savepath = "/home/wangnxr/models/vid_model_lstm_%s_itr_%i_t_%i_" % (sbj, itr, time)
            model.compile(optimizer=sgd,
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
            early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=10, verbose=0, mode='auto')
            checkpoint = ModelCheckpoint("%s_chkpt.h5" % model_savepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
            history_callback = model.fit_generator(
                train_generator,
                samples_per_epoch=len(dgdx_vid.filenames),
                nb_epoch=200,
                validation_data=validation_generator,
                nb_val_samples=len(dgdx_val_vid.filenames), callbacks=[checkpoint,early_stop])

            model.save("%s.h5" % model_savepath)
            pickle.dump(history_callback.history,
                        open("/home/wangnxr/history/%s.p" % model_savepath.split("/")[-1].split(".")[0], "wb"))
