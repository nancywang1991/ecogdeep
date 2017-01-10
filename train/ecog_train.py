import keras
from keras.applications.vgg16 import VGG16
from keras.preprocessing.ecog import EcogDataGenerator
from keras.layers import Flatten, Dense, Input, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Convolution2D, MaxPooling2D
#from keras.imagenet_utils import decode_predictions, preprocess_input, _obtain_input_shape
import numpy as np
import pdb
train_datagen = EcogDataGenerator()

test_datagen = EcogDataGenerator()

dgdx = train_datagen.flow_from_directory(
        #'/mnt/cb46fd46_5_no_offset/train/',
        '/home/wangnxr/Documents/dataset/ecog/cb46fd46_5/train_small/',
        batch_size=24,
        target_size=(96,1000),
        class_mode='binary')

dgdx_val = test_datagen.flow_from_directory(
        #'/mnt/cb46fd46_5_no_offset/test/',
        '/home/wangnxr/Documents/dataset/ecog/cb46fd46_5/train_small/',
        batch_size=24,
        shuffle=False,
        target_size=(96,1000),
        class_mode='binary')

#train_datagen.fit_generator(dgdx, nb_iter=100)
#test_datagen.fit_generator(dgdx_val, nb_iter=100)

train_generator=dgdx
validation_generator=dgdx_val

#base_model = VGG16(input_tensor=(Input(shape=(224, 224, 3))), include_top=False)

# Determine proper input shape
input_tensor=Input(shape=(96,1000,1))

# Block 1
x = MaxPooling2D((1,5),  name='pre_pool')(input_tensor)
x = Convolution2D(32, 1, 10, activation='relu', border_mode='same', name='block1_conv1')(x)
x = MaxPooling2D((1,2),  name='block1_pool')(x)

# Block 2
x = Convolution2D(64, 1, 10, activation='relu', border_mode='same', name='block2_conv1')(x)
x = MaxPooling2D((1,2),  name='block2_pool')(x)

# Block 3
x = Convolution2D(128, 1, 10, activation='relu', border_mode='same', name='block3_conv1')(x)
#x = Convolution2D(128, 1, 10, activation='relu', border_mode='same', name='block3_conv2')(x)
#x = Convolution2D(128, 1, 10, activation='relu', border_mode='same', name='block3_conv3')(x)
x = MaxPooling2D((1,2),  name='block3_pool')(x)

# Block 4
x = Convolution2D(256, 1, 10, activation='relu', border_mode='same', name='block4_conv1')(x)
#x = Convolution2D(256, 1, 10, activation='relu', border_mode='same', name='block4_conv2')(x)
#x = Convolution2D(256, 1, 10, activation='relu', border_mode='same', name='block4_conv3')(x)
x = MaxPooling2D((1,2), name='block4_pool')(x)


x = Flatten(name='flatten')(x)
x = Dense(1024,  name='fc1')(x)
#x = BatchNormalization()(x)
x = Activation('relu')(x)
#x = Dropout(0.5)(x)
x = Dense(256, name='fc2')(x)
#x = BatchNormalization()(x)
x = Activation('relu')(x)
#x = Dropout(0.5)(x)
#x = Dense(1, name='predictions')(x)
#x = BatchNormalization()(x)
predictions = Activation('sigmoid')(x)

#for layer in base_model.layers[:10]:
#    layer.trainable = False

model = Model(input=input_tensor, output=predictions)
#pdb.set_trace()
sgd = keras.optimizers.SGD(lr=0.01)

model.compile(optimizer=sgd,
              loss='binary_crossentropy',
              metrics=['accuracy'])

history_callback = model.fit_generator(
        train_generator,
        samples_per_epoch=48,
        nb_epoch=100,
        validation_data=validation_generator,
        nb_val_samples=48)
#pdb.set_trace()
#loss_history = history_callback.history["loss"]
#numpy_loss_history = np.array(loss_history)
#writefile = open("loss_history.txt", "wb")
with open("loss_history.txt", 'w') as f:
	for key, value in history_callback.history.items():
		f.write('%s:%s\n' % (key, value))

model.save("my_model.h5")

