import keras
from keras.applications.vgg16 import VGG16
from keras.preprocessing.ecog import EcogDataGenerator
from keras.layers import Flatten, Dense, Input, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from ..layers import Convolution2D, MaxPooling2D
from .imagenet_utils import decode_predictions, preprocess_input, _obtain_input_shape
import numpy as np
import pdb
train_datagen = EcogDataGenerator()

test_datagen = EcogDataGenerator()




dgdx = train_datagen.flow_from_directory(
        '/home/nancy/mvmt_vid_dataset/train/',
        read_formats={'p'},
        batch_size=32,
        class_mode='binary')

dgdx_val = test_datagen.flow_from_directory(
        '/home/nancy/mvmt_vid_dataset/test/',
        read_formats={'p'},
        batch_size=32,
        class_mode='binary')

train_datagen.fit_generator(dgdx, nb_iter=100)
test_datagen.fit_generator(dgdx_val, nb_iter=100)

train_generator=dgdx
validation_generator=dgdx_val

#base_model = VGG16(input_tensor=(Input(shape=(224, 224, 3))), include_top=False)

# Determine proper input shape
input_tensor=Input(shape=(1000,96))

# Block 1
x = Convolution2D(64, 1, 3, activation='relu', border_mode='same', name='block1_conv1')(input_tensor)
x = Convolution2D(64, 1, 3, activation='relu', border_mode='same', name='block1_conv2')(x)
x = MaxPooling2D((1, 2), strides=(1, 2), name='block1_pool')(x)

# Block 2
x = Convolution2D(128, 1, 3, activation='relu', border_mode='same', name='block2_conv1')(x)
x = Convolution2D(128, 1, 3, activation='relu', border_mode='same', name='block2_conv2')(x)
x = MaxPooling2D((1, 2), strides=(1, 2), name='block2_pool')(x)

# Block 3
x = Convolution2D(256, 1, 3, activation='relu', border_mode='same', name='block3_conv1')(x)
x = Convolution2D(256, 1, 3, activation='relu', border_mode='same', name='block3_conv2')(x)
x = Convolution2D(256, 1, 3, activation='relu', border_mode='same', name='block3_conv3')(x)
x = MaxPooling2D((1, 2), strides=(1, 2), name='block3_pool')(x)

# Block 4
x = Convolution2D(512, 1, 3, activation='relu', border_mode='same', name='block4_conv1')(x)
x = Convolution2D(512, 1, 3, activation='relu', border_mode='same', name='block4_conv2')(x)
x = Convolution2D(512, 1, 3, activation='relu', border_mode='same', name='block4_conv3')(x)
x = MaxPooling2D((1, 2), strides=(1, 2), name='block4_pool')(x)

# Block 5
x = Convolution2D(512, 1, 3, activation='relu', border_mode='same', name='block5_conv1')(x)
x = Convolution2D(512, 1, 3, activation='relu', border_mode='same', name='block5_conv2')(x)
x = Convolution2D(512, 1, 3, activation='relu', border_mode='same', name='block5_conv3')(x)
x = MaxPooling2D((1, 2), strides=(1, 2), name='block5_pool')(x)

x = Flatten(name='flatten')(x)
x = Dense(1024,  name='fc1')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, name='fc2')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
#x = Dropout(0.5)(x)
x = Dense(1, name='predictions')(x)
x = BatchNormalization()(x)
predictions = Activation('sigmoid')(x)

#for layer in base_model.layers[:10]:
#    layer.trainable = False

model = Model(input=input_tensor, output=predictions)

sgd = keras.optimizers.SGD(lr=0.001, clipnorm=0.5)

model.compile(optimizer=sgd,
              loss='binary_crossentropy',
              metrics=['accuracy'])

history_callback = model.fit_generator(
        train_generator,
        samples_per_epoch=20000,
        nb_epoch=50,
        validation_data=validation_generator,
        nb_val_samples=800)
#pdb.set_trace()
#loss_history = history_callback.history["loss"]
#numpy_loss_history = np.array(loss_history)
#writefile = open("loss_history.txt", "wb")
with open("loss_history.txt", 'w') as f:
	for key, value in history_callback.history.items():
		f.write('%s:%s\n' % (key, value))

model.save("my_model.h5")


