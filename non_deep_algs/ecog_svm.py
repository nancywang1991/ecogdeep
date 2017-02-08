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
samples_per_epoch_test=validation_generator.nb_sample
nb_epoch=100

model = SGDClassifier(verbose=1, n_jobs=4)
#test_data = np.vstack([val for val in validation_generator])
for e in xrange(nb_epoch):
    print "Epoch: %i" % e
    mean_test_score = []
    for b in xrange(samples_per_epoch/train_generator.batch_size):
        data = train_generator.next()
        if len(data)<train_generator.batch_size:
            train_generator.reset()
            data = train_generator.next()
        model.partial_fit(data[0], data[1])
    for b in xrange(samples_per_epoch_test/validation_generator.batch_size):
        test_data = validation_generator.next()
        if len(test_data)<validation_generator.batch_size:
            validation_generator.reset()
            data = validation_generator.next()
        mean_test_score.append(model.score(test_data[0], test_data[1]))
    print "Val_acc = %f" % np.mean(np.array(mean_test_score))


