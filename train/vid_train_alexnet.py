import keras
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator, center_crop
from keras.layers import Flatten, Dense, Input, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from convnetskeras.convnets import convnet
import pickle

import numpy as np
import pdb
train_datagen = ImageDataGenerator(
        rotation_range=40,
        rescale=1./255,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_datagen.config['center_crop_size'] = (227,227)
train_datagen.set_pipeline([center_crop])

test_datagen.config['center_crop_size'] = (227,227)
test_datagen.set_pipeline([center_crop])



dgdx = train_datagen.flow_from_directory(
        '/home/wangnxr/dataset/vid_offset_0/train/',
        read_formats={'png'},
        target_size=(int(340), int(256)),
        batch_size=32,
        class_mode='binary')

dgdx_val = test_datagen.flow_from_directory(
        '/home/wangnxr/dataset/vid_offset_0/test/',
        read_formats={'png'},
        target_size=(int(340), int(256)),
        batch_size=32,
        class_mode='binary')
#pdb.set_trace()
train_datagen.fit_generator(dgdx, nb_iter=96)
test_datagen.fit_generator(dgdx_val, nb_iter=96)

train_generator=dgdx
validation_generator=dgdx_val
#pdb.set_trace()
base_model = convnet('alexnet', weights_path="/home/wangnxr/Documents/ecogdeep/convnets-keras/examples/alexnet_weights.h5")
#base_model = VGG16(input_tensor=(Input(shape=(3,224, 224))), include_top=False, weights='imagenet')

x = base_model.get_layer("dense_1").output

#x = base_model.output
#x = Flatten(name='flatten')(x)
x = Dropout(0.5)(x)
x = Dense(512, W_regularizer=l2(0.01), name='fc1')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, W_regularizer=l2(0.01), name='fc2')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)
x = Dense(1, name='predictions')(x)
x = BatchNormalization()(x)
predictions = Activation('sigmoid')(x)

#for layer in base_model.layers[:10]:
#    layer.trainable = False

model = Model(input=base_model.input, output=predictions)

sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9)

model.compile(optimizer=sgd,
              loss='binary_crossentropy',
              metrics=['accuracy'])

history_callback = model.fit_generator(
        train_generator,
        samples_per_epoch=43904,
        nb_epoch=200,
        validation_data=validation_generator,
        nb_val_samples=11232)
#pdb.set_trace()
#loss_history = history_callback.history["loss"]
#numpy_loss_history = np.array(loss_history)
#writefile = open("loss_history.txt", "wb")
model.save("vid_model_alexnet.h5")
pickle.dump(history_callback.history, open("vid_history_alexnet.p", "wb"))


