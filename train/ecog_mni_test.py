import keras
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image2 import ImageDataGenerator
from keras.preprocessing.ecog import EcogDataGenerator
from keras.layers import Flatten, Dense, Input, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model, load_model
import numpy as np
import pdb
"""Accuracy of test set using ECoG LSTM model.

"""

sbj_ids = ['a0f', 'd65']
start_times = [2700, 3300,3900]
frames = [ range(1,6), range(4,9), range(7,12)]
channels_list = [np.arange(100)]
for s, sbj in enumerate(sbj_ids):
    main_ecog_dir = '/data2/users/nancy/dataset/ecog_mni_%s/' % (sbj)
    for t, time in enumerate(start_times):
 	try:
            model_file = "/home/nancy/models/ecog_model_mni_%s_itr_0_t_%i.h5" % (sbj, time)
            model = load_model(model_file)
        except:
            continue
        # pre_shuffle_index = np.random.permutation(len(glob.glob('%s/train/*/*.npy' % main_ecog_dir)))
        ## Data generation ECoG
        channels = channels_list[0]
        mode = "categorical"

        test_datagen_edf = EcogDataGenerator(
            center=False,
            seq_len=200,
            start_time=time,
            seq_num=5,
            seq_st=200
        )

        dgdx_val_edf = test_datagen_edf.flow_from_directory(
            '%s/test/' % main_ecog_dir,
            batch_size=10,
            shuffle=False,
            target_size=(1,len(channels),1000),
            final_size=(1,len(channels),200),
            channels = channels,
            class_mode=mode)

        validation_generator =  dgdx_val_edf

        #for layer in base_model.layers[:10]:
        #    layer.trainable = False
        #pdb.set_trace()
        files = dgdx_val_edf.filenames
        results = model.predict_generator(validation_generator, len(files))

        if mode=="binary":
                true = validation_generator.classes
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
        else:
                true = validation_generator.classes
                true_nums = np.zeros(np.max(true)+1)

                for r, result in enumerate(results):
                    if np.argmax(result) == true[r]:
                        true_nums[true[r]] += 1
                accuracies = [true_nums[c]/float(len(np.where(true==c)[0])) for c in xrange(len(true_nums))]

        with open("/home/nancy/results/%s_mni.txt" % model_file.split("/")[-1].split(".")[0], "wb") as writer:
                if mode=="binary":
                        writer.write("recall:%f\n" % recall)
                        writer.write("precision:%f\n" % precision)
                        writer.write("accuracy_1:%f\n" % accuracy_1)
                        writer.write("accuracy_0:%f\n" % accuracy_0)

                        for f, file in enumerate(files):
                                writer.write("%s:%f\n" % (file, results[f][0]))
                else:
                        for c, acc in enumerate(accuracies):
                                writer.write("accuracy_%i:%f\n" % (c, acc))
                        for f, file in enumerate(files):
                                writer.write("%s:%i->%f, %f\n" % (file, np.argmax(results[f]), np.max(results[f]), np.min(results[f])))


