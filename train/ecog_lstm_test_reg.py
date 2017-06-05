import keras
from keras.preprocessing.ecog_reg import EcogDataGenerator
from keras.layers import Flatten, Dense, Input, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model, load_model
import numpy as np
import pdb
"""Accuracy of test set using ECoG LSTM model.

"""

sbj_ids = ['a0f']
days = [11]
days = [7]
start_times = [2800, 3400,4000]
channels_list = [np.hstack([np.arange(36), np.arange(37, 65), np.arange(66, 92)])]
for s, sbj in enumerate(sbj_ids):
    main_ecog_dir = '/home/wangnxr/dataset_reg/ecog_vid_combined_%s_day%i/' % (sbj, days[s])
    for t, time in enumerate(start_times):
        try:
            model_file =  "/home/wangnxr/models/ecog_model_lstm_reg_%s_itr_%i.h5" % (sbj,0)
            model = load_model(model_file)
        except:
            continue
        # pre_shuffle_index = np.random.permutation(len(glob.glob('%s/train/*/*.npy' % main_ecog_dir)))
        ## Data generation ECoG
        channels = channels_list[s]
        times = [3900,3850,3800,3750,3700,3650,3600,3550,3500,3450,3400]

        test_datagen_edf = EcogDataGenerator(
            time_shift_range=200,
            center=True,
            seq_len=200,
            start_time=times[0],
            seq_num=5,
            seq_st=200
        )

        dgdx_val_edf = test_datagen_edf.flow_from_directory(
            '%s/val/' % main_ecog_dir,
            batch_size=10,
            shuffle=False,
            target_size=(1, len(channels), 1000),
            final_size=(1,len(channels),200),
            channels = channels,
            class_mode='binary')

        validation_generator =  dgdx_val_edf

        #for layer in base_model.layers[:10]:
        #    layer.trainable = False
        #pdb.set_trace()
        files = dgdx_val_edf.filenames
        results = model.predict_generator(validation_generator, len(files))
        pdb.set_trace()

