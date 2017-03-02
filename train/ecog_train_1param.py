import keras
from keras.regularizers import l2
from keras.preprocessing.ecog import EcogDataGenerator
from keras.layers import Flatten, Dense, Input, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D

#from keras.imagenet_utils import decode_predictions, preprocess_input, _obtain_input_shape
import numpy as np
import pdb
import pickle

## Data generation
train_datagen = EcogDataGenerator(
        time_shift_range=200,
        gaussian_noise_range=0.001,
        center=False
)

test_datagen = EcogDataGenerator(
        center=True
)

dgdx = train_datagen.flow_from_directory(
        #'/mnt/cb46fd46_5_no_offset/train/',
        '/home/wangnxr/dataset/ecog_offset_0_arm_a0f/train/',
        batch_size=22,
        target_size=(1,64,1000),
        class_mode='binary')
dgdx_val = test_datagen.flow_from_directory(
        #'/mnt/cb46fd46_5_no_offset/test/',
        '/home/wangnxr/dataset/ecog_offset_0_arm_a0f/test/',
        batch_size=14,
        shuffle=False,
        target_size=(1,64,1000),
        class_mode='binary')

train_generator=dgdx
validation_generator=dgdx_val

## Hyperparameter optimization space



def f_nn(params):
    # Determine proper input shape
    
    input_tensor=Input(shape=(1,64,1000))

    # Block 1
    x = AveragePooling2D((1,5),  name='pre_pool')(input_tensor)
    x = Convolution2D(16, params[0], params[1], border_mode='same', name='block1_conv1')(x)
    #x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((params[2],params[3]),  name='block1_pool')(x)

    # Block 2
    x = Convolution2D(32, params[0], params[1],  border_mode='same', name='block2_conv1')(x)
    #x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((params[2],params[3]),  name='block2_pool')(x)

    # Block 3
    x = Convolution2D(64, 1, 3, border_mode='same', name='block3_conv1')(x)
    #x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((1,2),  name='block3_pool')(x)

    # Block 4
    x = Convolution2D(128, 1, 3, border_mode='same', name='block4_conv1')(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((1,2), name='block4_pool')(x)


    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, W_regularizer=l2(0.01), name='fc1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, W_regularizer=l2(0.01), name='fc2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, name='predictions')(x)
    #x = BatchNormalization()(x)
    predictions = Activation('sigmoid')(x)

    #for layer in base_model.layers[:10]:
    #    layer.trainable = False

    model = Model(input=input_tensor, output=predictions)
    #pdb.set_trace()
    sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9)

    model.compile(optimizer=sgd,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    #history = keras.callbacks.ModelCheckpoint("weights.{epoch:02d}-{val_loss:.2f}.hdf5", save_best_only=True)
    history_callback = model.fit_generator(
            train_generator,
            samples_per_epoch=7590,
            nb_epoch=20,
            validation_data=validation_generator,
            nb_val_samples=686)
    #pdb.set_trace()
    #loss_history = history_callback.history["loss"]
    #numpy_loss_history = np.array(loss_history)
    #writefile = open("loss_history.txt", "wb")
    #with open("loss_history.txt", 'w') as f:
    #    for key, value in history_callback.history.items():
    #        f.write('%s:%s\n' % (key, value))

    model.save("/home/wangnxr/model_ecog_1d_%s_small_filt.h5" % "_".join([str(param) for param in params]))
    pickle.dump(history_callback.history,open("/home/wangnxr/history_ecog_1d_%s_small_filt.p" % "_".join([str(param) for param in params]), "wb"))

    loss = history_callback.history["val_loss"][-1]

    return loss

params_list = [(1,3,1,3),(1,3,1,2),(1,10,1,2),(2,10,1,2),(2,10,2,2)]
losses = []
for params in params_list:
    losses.append(f_nn(params))
best_ind = np.array(losses).argmin()
print 'best: %s %f' % ("_".join(params_list[best_ind]), losses[best_ind])

