import keras
from keras.preprocessing.ecog import EcogDataGenerator
from sklearn.linear_model import SGDClassifier
import numpy as np
import pdb
import pickle

## Data generation

test_datagen = EcogDataGenerator(
        start_time=3500,
        #f_lo=60,
        #f_hi=90,
        fft=True,
        center=True
)
#channels = np.arange(111)
channels = np.hstack([np.arange(36), np.arange(37, 68), np.arange(68, 92)])

dgdx_val = test_datagen.flow_from_directory(
        #'/mnt/cb46fd46_5_no_offset/test/',
        '/home/wangnxr/dataset/ecog_vid_combined_a0f_day8/test/',
        batch_size=10,
        shuffle=False,
        target_size=(1,len(channels),1000),
        final_size=(1,len(channels),2),
        channels=channels,
        class_mode='binary')

validation_generator=dgdx_val
samples_per_epoch_test=validation_generator.nb_sample
mean_test_score=[]
model = pickle.load(open("/home/wangnxr/models/svm_ecog_a0f_v2.p"))
for b in xrange(samples_per_epoch_test/validation_generator.batch_size):
        test_data = validation_generator.next()
        #pdb.set_trace()
	if len(test_data[0])<validation_generator.batch_size:
            validation_generator.reset()
            test_data = validation_generator.next()
        mean_test_score.append(model.score(np.reshape(test_data[0], (validation_generator.batch_size, test_data[0].shape[2]*test_data[0].shape[3])), test_data[1]))
print np.mean(np.array(mean_test_score))




