from __future__ import print_function
import keras
from keras.preprocessing.ecog import EcogDataGenerator
from sklearn.linear_model import SGDClassifier
import numpy as np
import pdb
import pickle

## Data generation
train_datagen = EcogDataGenerator(
        start_time=3500,
        time_shift_range=200,
        gaussian_noise_range=0.001,
        center=False,
        #f_lo=60,
        #f_hi=90,
        fft=True
)

test_datagen = EcogDataGenerator(
        start_time=3500,
        #f_lo=60,
        #f_hi=90,
        fft=True,
        center=True
)


sbj="a0f"
#channels = np.arange(111)
channels = np.hstack([np.arange(36), np.arange(37, 68), np.arange(68, 92)])

dgdx = train_datagen.flow_from_directory(
        #'/mnt/cb46fd46_5_no_offset/train/',
        '/home/wangnxr/dataset/ecog_vid_combined_%s_day8/train/' % sbj,
        batch_size=24,
        target_size=(1,len(channels),1000),
        final_size=(1,len(channels),2),
        channels = channels,
        class_mode='binary')
dgdx_val = test_datagen.flow_from_directory(
        #'/mnt/cb46fd46_5_no_offset/test/',
        '/home/wangnxr/dataset/ecog_vid_combined_%s_day8/val/' % sbj,
        batch_size=10,
        shuffle=False,
        target_size=(1,len(channels),1000),
        final_size=(1,len(channels),2),
        channels=channels,
        class_mode='binary')

train_generator=dgdx
validation_generator=dgdx_val
samples_per_epoch=train_generator.nb_sample
samples_per_epoch_test=validation_generator.nb_sample
nb_epoch=60

model = SGDClassifier(verbose=0,n_jobs=8)
logfile = open("/home/wangnxr/history/svm_ecog_%s_result_log_v2.txt" % sbj, "wb")
#test_data = np.vstack([val for val in validation_generator])
best_score = 0
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
    if np.mean(np.array(mean_test_score)) > best_score:
    	pickle.dump(model, open("/home/wangnxr/models/svm_ecog_%s_v2.p" % sbj, "wb"))
        best_score = np.mean(np.array(mean_test_score))
        print("Model saved", file=logfile)





