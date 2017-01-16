import keras
from keras.preprocessing.ecog import EcogDataGenerator
from keras.layers import Flatten, Dense, Input, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Convolution2D, MaxPooling2D
from hyperopt import Trials, fmin, tpe, hp, STATUS_OK

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
        '/home/nancy/Documents/ecog_dataset/d6532718/train/',
        batch_size=25,
        target_size=(1,64,1000),
        class_mode='binary')
dgdx_val = test_datagen.flow_from_directory(
        #'/mnt/cb46fd46_5_no_offset/test/',
        '/home/nancy/Documents/ecog_dataset/d6532718/test/',
        batch_size=24,
        shuffle=False,
        target_size=(1,64,1000),
        class_mode='binary')

train_generator=dgdx
validation_generator=dgdx_val

## Hyperparameter optimization space

space = {"conv_layer1_units": hp.uniform("conv1", 5, 15),
         "conv_layer2_units": hp.uniform("conv2", 5, 15),
         "conv_layer3_units": hp.uniform("conv3", 5, 15),
         "conv_layer4_units": hp.uniform("conv4", 5, 15),
         "pool_layer1": hp.uniform("pool1", 1, 3),
         "pool_layer2": hp.uniform("pool2", 1, 3),
         "pool_layer3": hp.uniform("pool3", 1, 3),
         "pool_layer4": hp.uniform("pool4", 1, 3),
         "dense1_units": hp.choice("fc1",[256, 512, 1024, 2048]),
         "dense2_units": hp.choice("fc2",[256, 512, 1024, 2048]),
         "conv_layer1_filters": hp.choice("filters1",[16, 32, 64, 128]),
         "conv_layer2_filters": hp.choice("filters2",[16, 32, 64, 128]),
         "conv_layer3_filters": hp.choice("filters3",[16, 32, 64, 128]),
         "conv_layer4_filters": hp.choice("filters4",[16, 32, 64, 128]),
         #"batch_size" : hp.uniform('batch_size', 16, 32),
         #'optimizer': hp.choice('optimizer', ['adadelta', 'adam', 'rmsprop'])
         }

def f_nn(params):
    # Determine proper input shape
    global itr
    print "testing itration %i" % itr
    itr+=1
    input_tensor=Input(shape=(1,64,1000))

    # Block 1
    x = MaxPooling2D((1,5),  name='pre_pool')(input_tensor)
    x = Convolution2D(params['conv_layer1_filters'], 1, params['conv_layer1_units'], border_mode='same', name='block1_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((1,params['pool_layer1']),  name='block1_pool')(x)

    # Block 2
    x = Convolution2D(params['conv_layer2_filters'], 1, params['conv_layer2_units'],  border_mode='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((1,params['pool_layer1']),  name='block2_pool')(x)

    # Block 3
    x = Convolution2D(params['conv_layer3_filters'], 1, params['conv_layer3_units'], border_mode='same', name='block3_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((1,params['pool_layer1']),  name='block3_pool')(x)

    # Block 4
    x = Convolution2D(params['conv_layer4_filters'], 1, params['conv_layer4_units'], border_mode='same', name='block4_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((1,params['pool_layer1']), name='block4_pool')(x)


    x = Flatten(name='flatten')(x)
    x = Dense(params['dense1_units'],  name='fc1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = Dropout(0.5)(x)
    x = Dense(params['dense2_units'], name='fc2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = Dropout(0.5)(x)
    x = Dense(1, name='predictions')(x)
    x = BatchNormalization()(x)
    predictions = Activation('sigmoid')(x)

    #for layer in base_model.layers[:10]:
    #    layer.trainable = False

    model = Model(input=input_tensor, output=predictions)
    #pdb.set_trace()
    sgd = keras.optimizers.SGD(lr=0.01)

    model.compile(optimizer=sgd,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    #history = keras.callbacks.ModelCheckpoint("weights.{epoch:02d}-{val_loss:.2f}.hdf5", save_best_only=True)
    history_callback = model.fit_generator(
            train_generator,
            samples_per_epoch=31800,
            nb_epoch=50,
            validation_data=validation_generator,
            nb_val_samples=1400)#, callbacks=[history])
    #pdb.set_trace()
    #loss_history = history_callback.history["loss"]
    #numpy_loss_history = np.array(loss_history)
    #writefile = open("loss_history.txt", "wb")
    with open("loss_history.txt", 'w') as f:
        for key, value in history_callback.history.items():
            f.write('%s:%s\n' % (key, value))

    model.save("ecog_1d_%i.h5" % itr)
    pickle.dump(params, open("ecog_1d_params_%i.p" % itr, "wb"))
    pickle.dump(history_callback.history,open("ecog_1d_history_%i.p" % itr, "wb"))

    loss = history_callback.history["loss_val"][-1]

    return {'loss': loss,'status': STATUS_OK}

itr = 0
trials = Trials()
best = fmin(f_nn, space, algo=tpe.suggest, max_evals=50, trials=trials)
print 'best: '
print best

