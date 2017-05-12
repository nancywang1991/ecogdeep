import keras
from keras.applications.vgg16 import VGG16
from keras.preprocessing.ecog import EcogDataGenerator
from keras.layers import Flatten, Dense, Input, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Convolution2D, MaxPooling2D
from keras.preprocessing.image2 import ImageDataGenerator
from keras.models import Model, load_model
#from keras.imagenet_utils import decode_predictions, preprocess_input, _obtain_input_shape
import numpy as np
import pdb
from sbj_parameters import *
import glob
import os


def izip_input(gen1, gen2):
    while 1:
        # pdb.set_trace()
        x1, y1 = gen1.next()
        x2 = gen2.next()[0]
        if not x1.shape[0] == x2.shape[0]:
            pdb.set_trace()
        yield [x1, x2], y1

with open("/home/wangnxr/results/ignore.txt", "wb") as summary_writer:
    for s, sbj in enumerate(sbj_ids):
        for t, time in enumerate(start_times):
            main_vid_dir = '/home/wangnxr/dataset/ecog_vid_combined_%s_day%i/train/' % (sbj, days[s])
            main_ecog_dir = '/home/wangnxr/dataset/ecog_vid_combined_%s_day%i/train/' % (sbj, days[s])
            new_dir = "/".join(main_ecog_dir.split("/")[:-2]) + "/ecog_vid_embedding_merge_merged/"

            for itr in xrange(3):
                model_files = glob.glob('/home/wangnxr/models/best/ecog_vid_model_lstm_%s_itr_%i_t_%i_*.h5' % (sbj, itr, time))
                if len(model_files)==0:
                    continue
                last_model_ind = 0
                #pre_shuffle_index = np.random.permutation(len(glob.glob('%s/train/*/*.npy' % main_ecog_dir)))
                ## Data generation ECoG
                channels = channels_list[s]
                model_file = model_files[last_model_ind]

                test_datagen = ImageDataGenerator(
                    rescale=1. / 255,
                    center_crop=(224, 224),
                    keep_frames=frames[t])

                dgdx_val = test_datagen.flow_from_directory(
                    main_vid_dir,
                    img_mode="seq",
                    read_formats={'png'},
                    target_size=(int(224), int(224)),
                    num_frames=12,
                    batch_size=10,
                    shuffle=False,
                    class_mode='binary')

                test_datagen_edf = EcogDataGenerator(
                    time_shift_range=200,
                    center=True,
                    seq_len=200,
                    start_time=time,
                    seq_num=5,
                    seq_st=200
                )

                dgdx_val_edf = test_datagen_edf.flow_from_directory(
                    main_ecog_dir,
                    batch_size=10,
                    shuffle=False,
                    target_size=(1, len(channels), 1000),
                    final_size=(1, len(channels), 200),
                    channels=channels,
                    class_mode='binary')

                validation_generator = izip_input(dgdx_val, dgdx_val_edf)
                model = load_model(model_file)
                new_model = Model(model.input, model.layers[-7].output)
                #pdb.set_trace()
                files = dgdx_val_edf.filenames
                if not os.path.exists(new_dir):
                    os.makedirs(new_dir)
                    os.makedirs(new_dir + files[0].split("/")[0])
                    os.makedirs(new_dir + files[-1].split("/")[0])
                results = new_model.predict_generator(validation_generator, len(files))
                for r, result in enumerate(results):
                    np.save(new_dir + files[r].split(".")[0] + "_" + str(time) + ".npy", result)
