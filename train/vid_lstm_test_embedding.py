import keras
from keras.preprocessing.image2 import ImageDataGenerator, center_crop
from keras.applications.vgg16 import VGG16
from keras.preprocessing.ecog import EcogDataGenerator
from keras.layers import Flatten, Dense, Input, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Model, load_model
import ecogdeep.train.vid_model_seq as vid_model_seq
#from keras.imagenet_utils import decode_predictions, preprocess_input, _obtain_input_shape
import numpy as np
import pdb
from sbj_parameters import *
import glob
import os

with open("/home/wangnxr/results/ignore.txt", "wb") as summary_writer:
    for s, sbj in enumerate(sbj_ids):
        for t, time in enumerate(start_times):
            main_vid_dir = '/home/wangnxr/dataset/ecog_vid_combined_%s_day%i/train/' % (sbj, days[s])
            new_dir = "/".join(main_vid_dir.split("/")[:-2]) + "/vid_embedding/"

            for itr in xrange(1):
                model_files = glob.glob('/home/wangnxr/models/best/vid_model_lstm_%s_itr_*_t_%i_*.h5' % (sbj, time))
                if len(model_files)==0:
                    continue
                last_model_ind = 0
                #pre_shuffle_index = np.random.permutation(len(glob.glob('%s/train/*/*.npy' % main_ecog_dir)))
                ## Data generation ECoG
                channels = channels_list[s]
                model_file = model_files[last_model_ind]

                test_datagen_vid = ImageDataGenerator(
                    random_black=False,
                    rescale=1. / 255,
                    center_crop=(224, 224),
                    keep_frames=frames[t])

                vid_model = vid_model_seq.vid_model()

                dgdx_val_vid = test_datagen_vid.flow_from_directory(
                    main_vid_dir,
                    img_mode="seq",
                    read_formats={'png'},
                    target_size=(int(224), int(224)),
                    num_frames=12,
                    batch_size=10,
                    shuffle=False,
                    class_mode='binary')
    
                validation_generator = dgdx_val_vid
                model = load_model(model_file)
                new_model = Model(model.input, model.layers[-7].output)
                #pdb.set_trace()
                files = dgdx_val_vid.filenames
                if not os.path.exists(new_dir):
                    os.makedirs(new_dir)
                    os.makedirs(new_dir + files[0].split("/")[0])
                    os.makedirs(new_dir + files[-1].split("/")[0])
                results = new_model.predict_generator(validation_generator, len(files))
                for r, result in enumerate(results):
                    np.save(new_dir + files[r].split(".")[0] + "_" + str(time) + ".npy", result)

