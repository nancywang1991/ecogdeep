import keras
from keras.preprocessing.ecog import EcogDataGenerator
from sklearn.linear_model import SGDClassifier
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
        '/home/wangnxr/dataset/vid_ecog_0/train/',
        batch_size=25,
        target_size=(1,64,1000),
        class_mode='binary')
dgdx_val = test_datagen.flow_from_directory(
        #'/mnt/cb46fd46_5_no_offset/test/',
        '/home/wangnxr/dataset/vid_ecog_0/test/',
        batch_size=25,
        shuffle=False,
        target_size=(1,64,1000),
        class_mode='binary')

train_generator=dgdx
validation_generator=dgdx_val
samples_per_epoch=20000
nb_epoch=100

model = SGDClassifier(verbose=1, n_jobs=4)
test_data = np.vstack([val for val in validation_generator])
for e in xrange(nb_epoch):
    print "Epoch: %i" % e
    for b in xrange(samples_per_epoch/train_generator.batch_index):
        data = train_generator.next()
        model.partial_fit(data[0], data[1])
    model.score(test_data[0], test_data[1])

