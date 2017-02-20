from __future__ import print_function
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
        center=False,
        f_lo=10,
        f_hi=210,
        fft=True
)

test_datagen = EcogDataGenerator(
        f_lo=10,
        f_hi=210,
        fft=True,
        center=True
)

dgdx = train_datagen.flow_from_directory(
        #'/mnt/cb46fd46_5_no_offset/train/',
        '/home/wangnxr/dataset/ecog_offset_0/train/',
        batch_size=24,
        target_size=(1,64,1000),
        final_size=(1,64,200),
        class_mode='binary')
dgdx_val = test_datagen.flow_from_directory(
        #'/mnt/cb46fd46_5_no_offset/test/',
        '/home/wangnxr/dataset/ecog_offset_0/test/',
        batch_size=25,
        shuffle=False,
        target_size=(1,64,1000),
        final_size=(1,64,200),
        class_mode='binary')

train_generator=dgdx
validation_generator=dgdx_val
samples_per_epoch=10000
samples_per_epoch_test=validation_generator.nb_sample
nb_epoch=30

model = SGDClassifier(verbose=0,n_jobs=8)
logfile = open("svm_ecog_result_log.txt", "wb")
#test_data = np.vstack([val for val in validation_generator])
for e in xrange(nb_epoch):
    print("Epoch: %i" % e, file=logfile)
    mean_test_score = []
    for b in xrange(samples_per_epoch/train_generator.batch_size):
        data = train_generator.next()

        if len(data[0])<train_generator.batch_size:
            train_generator.reset()
            data = train_generator.next()
        model.partial_fit(np.reshape(data[0], (train_generator.batch_size, data[0].shape[2]*data[0].shape[3])), data[1], classes=(0,1))
        if b%50==0:
                print(model.score(np.reshape(data[0], (train_generator.batch_size, data[0].shape[2]*data[0].shape[3])),data[1]), file=logfile)
                print("Batch %i of %i" % (b, samples_per_epoch/train_generator.batch_size), file=logfile)
                logfile.flush()
    for b in xrange(samples_per_epoch_test/validation_generator.batch_size):
        test_data = validation_generator.next()
        #pdb.set_trace()
	if len(test_data[0])<validation_generator.batch_size:
            validation_generator.reset()
            test_data = validation_generator.next()
        mean_test_score.append(model.score(np.reshape(test_data[0], (validation_generator.batch_size, test_data[0].shape[2]*data[0].shape[3])), test_data[1]))
    print("Val_acc = %f" % np.mean(np.array(mean_test_score)), file=logfile)





