import keras
from keras.applications.vgg16 import VGG16
from ecogdeep.data.preprocessing.ecog import EcogDataGenerator
from keras.layers import Flatten, Dense, Input, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Model, load_model
#from keras.imagenet_utils import decode_predictions, preprocess_input, _obtain_input_shape
import numpy as np
import pdb
from sbj_parameters import *
import glob
import time as time2


with open("/home/wangnxr/results/ecog_lstm_mni_summary_results2.txt", "wb") as summary_writer:
    for s, sbj in enumerate(["a0f", "d65"]):
        for time in [3900]:
            main_ecog_dir = '/data2/users/wangnxr/dataset/ecog_mni_interp_%s/test/' % (sbj)
            for itr in xrange(1):
                model_file = glob.glob('/home/wangnxr/current_models/ecog_model_mni_interp_jitter_True_%s_itr_%i_t_%i_best.h5' % (sbj, itr, time))[0]
                channels = np.arange(100)
                summary_writer.write(model_file.split("/")[-1].split(".")[0] + "\n")
 		model = load_model(model_file)
		test_datagen_edf = EcogDataGenerator(
                        time_shift_range=200,
                        center=True,
                        seq_len=200,
                        start_time=time,
                        seq_num=5,
                        seq_st=200
                    )

		for ablate_channel in channels:
		    start = time2.time()
                    dgdx_val_edf = test_datagen_edf.flow_from_directory(
                        main_ecog_dir,
                        batch_size=10,
                        shuffle=False,
                        target_size=(1, len(channels), 1000),
                        final_size=(1, len(channels), 200),
                        channels=channels,
                        ablate=[ablate_channel],
                        class_mode='binary')
		    files = dgdx_val_edf.filenames
                    validation_generator = dgdx_val_edf
                    results = model.predict_generator(validation_generator, len(files)/10)
                    true = dgdx_val_edf.classes
                    true_0 = 0
                    true_1 = 0

                    for r, result in enumerate(results):
                        if true[r]== 0 and result<0.5:
                            true_0+=1
                        if true[r]== 1 and result>=0.5:
                            true_1+=1

                    recall = true_1/float(len(np.where(true==1)[0]))
                    precision = true_1/float((true_1 + (len(np.where(true==0)[0])-true_0)))
                    accuracy_1 = true_1/float(sum(true))
                    accuracy_0 = true_0/float((len(np.where(true==0)[0])))

                    summary_writer.write("ablating:%i -> accuracy_1:%f | accuracy_0:%f | average:%f \n" %
                                         (ablate_channel, accuracy_1, accuracy_0, np.mean([accuracy_1, accuracy_0])))
		    print (time2.time()-start)	
