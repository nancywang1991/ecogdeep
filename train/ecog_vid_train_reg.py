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

main_vid_dir = '/home/wangnxr/dataset_xy_reg/ecog_vid_combined_a0f_day11/'
main_ecog_dir = '/home/wangnxr/dataset_xy_reg/ecog_vid_combined_a0f_day11/'
times = [3900,3700,3500]

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


vid_model = vid_model()

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
    '/%s/test/' % main_vid_dir,
    read_formats={'png'},
    target_size=(int(224), int(224)),
    num_frames=12,
    batch_size=10,
    shuffle=False,
    class_mode='binary')

## Data generation ECoG
channels = np.hstack([np.arange(36), np.arange(37, 68), np.arange(68, 92)])

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
    '%s/test/' % main_ecog_dir,
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
        x2 = gen2.next()[0]
        if not x1.shape[0] == x2.shape[0]:
            pdb.set_trace()
        x1 = [x1,x2]
        yield x1, y1


ecog_model = ecog_1d_model(channels=len(channels))
base_model_ecog = Model(ecog_model.input, ecog_model.get_layer("fc2").output)
ecog_series = Input(shape=(1,len(channels),1000))

train_generator = izip_input(dgdx_vid, dgdx_edf)
validation_generator = izip_input(dgdx_val_vid, dgdx_val_edf)

base_model_vid = Model(vid_model.input, vid_model.get_layer("flatten").output)

frame_a = Input(shape=(3,224,224))


predictions = base_model_vid(frame_a)

for layer in base_model_vid.layers:
    layer.trainable = True

tower1 = base_model_vid(frame_a)
tower2 = base_model_ecog(ecog_series)
#tower2 = Dense(3136, init='normal')(tower2)
predictions = merge([tower1, tower2], mode='sum', concat_axis=-1)
#x = tower1
#predictions = Dense(3136, name='predictions', init='normal')(x)

sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9)

model = Model(input=[frame_a, ecog_series], output=predictions)

model_savepath = "/home/wangnxr/models/ecog_vid_model_reg"
model.compile(optimizer=sgd,
              loss='mean_squared_error')
checkpoint = ModelCheckpoint("%s_chkpt.h5" % model_savepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
history_callback = model.fit_generator(
    train_generator,
    samples_per_epoch=len(dgdx_vid.filenames),
    nb_epoch=1000,
    validation_data=validation_generator,
    nb_val_samples=len(dgdx_val_vid.filenames), callbacks=[checkpoint])

model.save("%s.h5" % model_savepath)
pickle.dump(history_callback.history, open("/home/wangnxr/models/ecog_vid_history_reg.txt", "wb"))

