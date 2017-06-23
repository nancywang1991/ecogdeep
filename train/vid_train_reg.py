import keras
from keras.preprocessing.ecog import EcogDataGenerator
from keras.layers import Flatten, Dense, Input, Dropout, Activation, merge
from keras.preprocessing.image_reg import ImageDataGenerator, center_crop
from keras.models import Model
from ecogdeep.train.ecog_1d_model import ecog_1d_model
from ecogdeep.train.vid_model_reg import vid_model
from keras.callbacks import ModelCheckpoint

#from keras.imagenet_utils import decode_predictions, preprocess_input, _obtain_input_shape
import numpy as np
import pdb
import pickle
import glob

main_ecog_dir = '/home/wangnxr/dataset/ecog_vid_combined_a0f_day6/'
main_vid_dir = '/home/wangnxr/dataset/ecog_vid_combined_a0f_day6/'

times = [3900,3700,3500]

# Video data generators
train_datagen_vid = ImageDataGenerator(
    #rotation_range=40,
    rescale=1./255,
    #zoom_range=0.2,
    #horizontal_flip=True,
    start_time=times,
    random_crop=(224,224))

test_datagen_vid = ImageDataGenerator(
    rescale=1./255,
    start_time=times[0],
    center_crop=(224, 224)),


vid_model = vid_model()

dgdx_vid = train_datagen_vid.flow_from_directory(
    '/%s/train/' % main_vid_dir,
    read_formats={'png'},
    target_size=(int(224), int(224)),
    num_frames=11,
    batch_size=24,
    class_mode='binary',
    shuffle=False,
    pre_shuffle_ind=1)

dgdx_val_vid = test_datagen_vid.flow_from_directory(
    '/%s/test/' % main_vid_dir,
    read_formats={'png'},
    target_size=(int(224), int(224)),
    num_frames=11,
    batch_size=10,
    shuffle=False,
    class_mode='binary')
train_generator = dgdx_vid
validation_generator = dgdx_val_vid

base_model_vid = Model(vid_model.input, vid_model.get_layer("block8_conv1").output)

frame_a = Input(shape=(3,224,224))


predictions = base_model_vid(frame_a)

for layer in base_model_vid.layers:
    layer.trainable = True

model = Model(input=[frame_a], output=predictions)

sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9)

model_savepath = "/home/wangnxr/models/vid_model_reg"
model.compile(optimizer=sgd,
              loss='mean_squared_error')
checkpoint = ModelCheckpoint("%s_chkpt.h5" % model_savepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
history_callback = model.fit_generator(
    train_generator,
    samples_per_epoch=len(dgdx_vid.filenames),
    nb_epoch=400,
    validation_data=validation_generator,
    nb_val_samples=len(dgdx_val_vid.filenames), callbacks=[checkpoint])

model.save("%s.h5" % model_savepath)
pickle.dump(history_callback.history, open("/home/wangnxr/models/vid_history_reg.txt", "wb"))

