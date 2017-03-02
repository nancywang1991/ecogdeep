import keras
from keras.applications.vgg16 import VGG16
from keras.preprocessing.ecog3D import Ecog3DDataGenerator
from keras.layers import Flatten, Dense, Input, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Convolution3D, MaxPooling3D
#from keras.imagenet_utils import decode_predictions, preprocess_input, _obtain_input_shape
import numpy as np
import pdb
import pickle

train_datagen = Ecog3DDataGenerator(
        time_shift_range=200,
        gaussian_noise_range=0.001,
        center=False
)

test_datagen = Ecog3DDataGenerator(
        center=True
)


dgdx = train_datagen.flow_from_directory(
        '/home/wangnxr/dataset/ecog_vid_combined/train/',
        #'/home/nancy/Documents/ecog_dataset/d6532718/train/',
        batch_size=24,
        target_size=(1,8,8,1000),
        class_mode='binary')

dgdx_val = test_datagen.flow_from_directory(
        '/home/wangnxr/dataset/ecog_vid_combined/test/',
        #'/home/nancy/Documents/ecog_dataset/d6532718/test/',
        batch_size=12,
        shuffle=False,
        target_size=(1,8,8,1000),
        class_mode='binary')

#train_datagen.fit_generator(dgdx, nb_iter=100)
#test_datagen.fit_generator(dgdx_val, nb_iter=100)

train_generator=dgdx
validation_generator=dgdx_val

#base_model = VGG16(input_tensor=(Input(shape=(224, 224, 3))), include_top=False)
def f_nn(params):
	# Determine proper input shape
	input_tensor=Input(shape=(1,8,8,1000))

	# Block 1
	x = MaxPooling3D((1,1,5),  name='pre_pool')(input_tensor)
	#x = Convolution2D(32, 1, 10, activation='relu', border_mode='same', name='block1_conv1')(x)

	x = Convolution3D(32, params[0], params[0], params[1], activation='relu', border_mode='same', name='block1_conv2')(x)
	x = MaxPooling3D((1,1,params[3]),  name='block1_pool')(x)

	# Block 2
	x = Convolution3D(64, params[0], params[0], params[1], activation='relu', border_mode='same', name='block2_conv1')(x)
	#x = Convolution2D(64, 1, 10, activation='relu', border_mode='same', name='block2_conv2')(x)
	x = MaxPooling3D((params[2],params[2],params[3]),  name='block2_pool')(x)

	# Block 3
	x = Convolution3D(128, 2,2, 3, activation='relu', border_mode='same', name='block3_conv1')(x)
	#x = Convolution2D(128, 1, 10, activation='relu', border_mode='same', name='block3_conv2')(x)
	#x = Convolution2D(128, 1, 10, activation='relu', border_mode='same', name='block3_conv3')(x)
	x = MaxPooling3D((1,1,2),  name='block3_pool')(x)

	# Block 4
	x = Convolution3D(256, 2, 2, 3, activation='relu', border_mode='same', name='block4_conv1')(x)
	x = MaxPooling3D((1,1,2), name='block4_pool')(x)


	x = Flatten(name='flatten')(x)
	x = Dropout(0.5)(x)
	x = Dense(1024,  name='fc1')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Dropout(0.5)(x)
	x = Dense(256, name='fc2')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Dropout(0.5)(x)
	x = Dense(1, name='predictions')(x)
	predictions = Activation('sigmoid')(x)

#for layer in base_model.layers[:10]:
#    layer.trainable = False

	model = Model(input=input_tensor, output=predictions)
	sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9)

	model.compile(optimizer=sgd,
              loss='binary_crossentropy',
              metrics=['accuracy'])

	#history = keras.callbacks.ModelCheckpoint("/mnt/weights.{epoch:02d}-{val_loss:.2f}.hdf5", save_best_only=True)
	history_callback = model.fit_generator(
        	train_generator,
        	samples_per_epoch=9816,
        	nb_epoch=20,
        	validation_data=validation_generator,
        	nb_val_samples=2124)
#pdb.set_trace()
#loss_history = history_callback.history["loss"]
#numpy_loss_history = np.array(loss_history)
#writefile = open("loss_history.txt", "wb")
#with open("loss_history.txt", 'w') as f:
#	for key, value in history_callback.history.items():
#		f.write('%s:%s\n' % (key, value))

	model.save("/home/wangnxr/model_ecog_3d_%s_small_filt.h5" % "_".join([str(param) for param in params]))
	pickle.dump(history_callback.history, open("/home/wangnxr/history_ecog_3d_%s_small_filt.p" % "_".join([str(param) for param in params]), "wb"))

if __name__=="__main__":
	params_list = [(2, 5, 2,2), (2,10, 2,2),(3,5,2,2),(3,10,2,2), (2, 5, 1,2), (2,10, 1,2),(3,5,1,2),(3,10,1,2)]
	for params in params_list:
		f_nn(params)
