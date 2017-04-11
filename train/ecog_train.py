import keras
from keras.preprocessing.ecog import EcogDataGenerator
from keras.preprocessing.image2 import ImageDataGenerator, center_crop
from keras.layers import Flatten, Dense, Input, Dropout, Activation, merge
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Convolution2D, MaxPooling2D
from hyperopt import Trials, fmin, tpe, hp, STATUS_OK
from keras.regularizers import l2
from itertools import izip
from keras.callbacks import ModelCheckpoint
from ecogdeep.train.ecog_1d_model import ecog_1d_model
from ecogdeep.train.vid_alexnet_2towers_model import video_2tower_model

#from keras.imagenet_utils import decode_predictions, preprocess_input, _obtain_input_shape
import numpy as np
import pdb
import pickle
import glob

main_ecog_dir = '/home/wangnxr/dataset/ecog_vid_combined_a0f_day6/'
main_vid_dir = '/home/wangnxr/dataset/ecog_vid_combined_a0f_day6/'
#pre_shuffle_index = np.random.permutation(len(glob.glob('%s/train/*/*.npy' % main_ecog_dir)))
## Data generation ECoG
train_datagen_edf = EcogDataGenerator(
    start_time=3300,
    time_shift_range=200,
    gaussian_noise_range=0.001,
    center=False
)

test_datagen_edf = EcogDataGenerator(
    start_time=3300,
    center=True
)
channels = np.hstack([np.arange(36), np.arange(37, 68), np.arange(68, 92)])
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

ecog_model = ecog_1d_model()

train_generator = dgdx_edf
validation_generator = dgdx_val_edf

base_model_ecog = Model(ecog_model.input, ecog_model.get_layer("fc1").output)

ecog_series = Input(shape=(1,len(channels),1000))

x = base_model_ecog(ecog_series)
x = Dropout(0.5)(x)
x = Dense(1024, W_regularizer=l2(0.01), name='merge1')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, W_regularizer=l2(0.01), name='merge2')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)
x = Dense(1, name='predictions')(x)
#x = BatchNormalization()(x)
predictions = Activation('sigmoid')(x)

for layer in base_model_ecog.layers:
    layer.trainable = True


model = Model(input=[ecog_series], output=predictions)

sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9)

model_savepath = "/home/wangnxr/models/ecog_history_alexnet_3towers_dense1_a0f_pred"
model.compile(optimizer=sgd,
              loss='binary_crossentropy',
              metrics=['accuracy'])
checkpoint = ModelCheckpoint("%s_chkpt.h5" % model_savepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
history_callback = model.fit_generator(
    train_generator,
    samples_per_epoch=len(dgdx_edf.filenames),
    nb_epoch=40,
    validation_data=validation_generator,
    nb_val_samples=len(dgdx_val_edf.filenames), callbacks=[checkpoint])

model.save("%s.h5" % model_savepath)
pickle.dump(history_callback.history, open("/home/wangnxr/models/ecog_history_alexnet_3towers_dense1_a0f_pred", "wb"))
