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
from keras.callbacks import ModelCheckpoint
from ecogdeep.train.ecog_1d_model_seq import ecog_1d_model
from ecogdeep.train.vid_model_seq import vid_model

#from keras.imagenet_utils import decode_predictions, preprocess_input, _obtain_input_shape
import numpy as np
import pdb
import pickle
import glob

main_ecog_dir = '/home/wangnxr/dataset/ecog_vid_combined_e5b_day9/'
main_vid_dir = '/home/wangnxr/dataset/ecog_vid_combined_e5b_day9/'
#pre_shuffle_index = np.random.permutation(len(glob.glob('%s/train/*/*.npy' % main_ecog_dir)))
## Data generation ECoG
channels = np.arange(111)
train_datagen_edf = EcogDataGenerator(
    gaussian_noise_range=0.001,
    center=False,
    seq_len=200,
    seq_start=4000,
    seq_num = 5,
    seq_st = 200
)

test_datagen_edf = EcogDataGenerator(
    center=False,
    seq_len=200,
    seq_start=4000,
    seq_num=5,
    seq_st=200
)

dgdx_edf = train_datagen_edf.flow_from_directory(
    #'/mnt/cb46fd46_5_no_offset/train/',
    '%s/train/' % main_ecog_dir,
    batch_size=24,
    class_mode='categorical',
    shuffle=False,
    channels = channels,
    pre_shuffle_ind=1,
    final_size=(1,len(channels),200),
    )

dgdx_val_edf = test_datagen_edf.flow_from_directory(
    #'/mnt/cb46fd46_5_no_offset/test/',
    '%s/val/' % main_ecog_dir,
    batch_size=10,
    shuffle=False,
    final_size=(1,len(channels),200),
    channels = channels,
    class_mode='categorical')


ecog_model = ecog_1d_model(channels=len(channels))


train_generator =  dgdx_edf
validation_generator =  dgdx_val_edf
base_model_ecog = Model(ecog_model.input, ecog_model.get_layer("fc1").output)

ecog_series = Input(shape=(5,1,len(channels),200))

x = base_model_ecog(ecog_series)

x = Dropout(0.5)(x)
x = TimeDistributed(Dense(256, W_regularizer=l2(0.01), name='merge2'))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)
x = LSTM(20, dropout_W=0.2, dropout_U=0.2, name='lstm')(x)
#x = Dense(1, name='predictions')(x)
#x = BatchNormalization()(x)
#predictions = Activation('sigmoid')(x)
predictions = Dense(2, activation='softmax', name='predictions')(x)

for layer in base_model_ecog.layers:
    layer.trainable = True


model = Model(input=[ecog_series], output=predictions)
#model = Model(input=[base_model_vid.input, base_model_ecog.input], output=predictions)

sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9)

model_savepath = "/home/wangnxr/models/ecog_model_lstm_e5b_5st_dec"
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
checkpoint = ModelCheckpoint("%s_chkpt.h5" % model_savepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
history_callback = model.fit_generator(
    train_generator,
    samples_per_epoch=len(dgdx_edf.filenames),
    nb_epoch=80,
    validation_data=validation_generator,
    nb_val_samples=len(dgdx_val_edf.filenames), callbacks=[checkpoint])

model.save("%s.h5" % model_savepath)
pickle.dump(history_callback.history, open("/home/wangnxr/history/ecog_history_lstm_e5b_5st_dec", "wb"))
