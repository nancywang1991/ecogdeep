import keras
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image2 import ImageDataGenerator
from keras.preprocessing.ecog import EcogDataGenerator
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


def izip_input(gen1, gen2):
    while 1:
        # pdb.set_trace()
        x1, y1 = gen1.next()
        x2 = gen2.next()[0]
        if not x1[0].shape[0] == x2.shape[0]:
            pdb.set_trace()
        x1.append(x2)
        yield x1, y1

with open("/home/wangnxr/results/ecog_vid_lstm_summary_results.txt", "wb") as summary_writer:
    for s, sbj in enumerate(sbj_ids):
        for t, time in enumerate(start_times):
            main_vid_dir = '/home/wangnxr/dataset/ecog_vid_combined_%s_day%i/test/' % (sbj, days[s])
            main_ecog_dir = '/home/wangnxr/dataset/ecog_vid_combined_%s_day%i/test/' % (sbj, days[s])
            for itr in xrange(3):
                model_files = glob.glob(
                    '/home/wangnxr/models/ecog_vid_model_lstm_%s_itr_%i_t_%i_*chkpt.h5' % (sbj, itr, time))
                if len(model_files)==0:
                    continue
                ## Data generation ECoG
                channels = channels_list[s]
                model_file = model_files[0]

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

                #pdb.set_trace()
                files = dgdx_val.filenames
                results = model.predict_generator(validation_generator, len(files))
                true = dgdx_val.classes
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

                summary_writer.write(model_file.split("/")[-1].split(".")[0] + "\n")
                summary_writer.write("accuracy_1:%f\n" % accuracy_1)
                summary_writer.write("accuracy_0:%f\n" % accuracy_0)

                with open("/home/wangnxr/results/%s.txt" % model_file.split("/")[-1].split(".")[0], "wb") as writer:
                        writer.write("recall:%f\n" % recall)
                        writer.write("precision:%f\n" % precision)
                        writer.write("accuracy_1:%f\n" % accuracy_1)
                        writer.write("accuracy_0:%f\n" % accuracy_0)

                        for f, file in enumerate(files):
                                writer.write("%s:%f\n" % (file, results[f][0]))
