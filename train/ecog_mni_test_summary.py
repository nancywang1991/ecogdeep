import keras
from keras.applications.vgg16 import VGG16
from ecogdeep.data.preprocessing.ecog import EcogDataGenerator
from keras.layers import Flatten, Dense, Input, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Model, load_model
#from keras.imagenet_utils import decode_predictions, preprocess_input, _obtain_input_shape
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import pdb
import glob
import pickle
import os

def my_train_val_save_fig(data, xlabel, ylabel, ylim, title, savename):
    plt.plot(data[0], label = "train")
    plt.plot(data[1], label = "val")
    plt.ylim(ylim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(savename)
    plt.clf()
    return


for model_root in glob.glob('/home/wangnxr/models/*ellip_*itr_0_t_3900_best.h5'):
    model_root_name =  "_".join(model_root.split("/")[-1].split(".")[0].split("_")[:-2])
    model_type = model_root_name.split("_")[3]
    if model_type == "ellip":
	model_type = model_root_name.split("_")[4]
    #if os.path.exists("/home/wangnxr/results/%s.txt" % model_root_name):
#	continue
    with open("/home/wangnxr/results/%s.txt" % model_root_name, "w") as summary_writer:
        for time in [2700, 3300, 3900]:
	    model_file = "%s_%i_best.h5" % ("_".join(model_root.split("_")[:-2]), time)
	    model_name = "_".join(model_file.split("/")[-1].split(".")[0].split("_")[:-1])
	    print model_file
            # Plot training results
	    #history_file = pickle.load(open('/home/wangnxr/history/%s.p' % model_name, "rb"))
            #my_train_val_save_fig([history_file["acc"], history_file["val_acc"]], "Epochs",  "Accuracy", [0,1], model_name, "/home/wangnxr/results/%s.png" % model_name)  
	    for s, sbj in enumerate(["d65", "a0f", "cb4", "c95"]):
	        if model_type == "deep":
                    main_ecog_dir = '/data2/users/wangnxr/dataset/ecog_mni_ellip_deep_impute_%s/test/' % (sbj)
	        elif model_type == "zero":
	            main_ecog_dir = '/data2/users/wangnxr/dataset/ecog_mni_ellip_%s/test/' % (sbj)
	        elif model_type == "interp":
	            main_ecog_dir = '/data2/users/wangnxr/dataset/ecog_mni_ellip_interp_%s/test/' % (sbj)
		#elif model_root_name.split("_")[3] == "ellipv2":
		#    main_ecog_dir = '/data2/users/wangnxr/dataset/ecog_mni_ellipv2_interp_%s/test/' % (sbj)
		# Data generation ECoG
                channels = np.arange(100)

                test_datagen_edf = EcogDataGenerator(
                    center=True,
                    time_shift_range=200,
                    start_time=time,
                    seq_num=5,
                    seq_st=200,
                    seq_len=200
                )

                dgdx_val_edf = test_datagen_edf.flow_from_directory(
                    #'/mnt/cb46fd46_5_no_offset/test/',
                    main_ecog_dir,
                    batch_size=10,
                    shuffle=False,
                    target_size=(1,len(channels),1000),
                    final_size=(1,len(channels),200),
                    channels = channels,
                    class_mode='binary')

                validation_generator =  dgdx_val_edf
                model = load_model(model_file)

                #pdb.set_trace()
                files = dgdx_val_edf.filenames
                results = model.predict_generator(validation_generator, len(files)/10)
                true = dgdx_val_edf.classes
		if sbj == "c95":
			true = 1-true
		if model_root_name.split("_")[-4] == "c95":
			true = 1-true
                true_0 = 0
                true_1 = 0
                for r, result in enumerate(results):
                    if true[r]== 0 and result<0.5:
                        true_0+=1
                    if true[r]== 1 and result>=0.5:
                        true_1+=1

                #recall = true_1/float(len(np.where(true==1)[0]))
                #precision = true_1/float((true_1 + (len(np.where(true==0)[0])-true_0)))
                accuracy_1 = true_1/float(sum(true))
                accuracy_0 = true_0/float((len(np.where(true==0)[0])))

                summary_writer.write("model: %s\n" % model_name)
                summary_writer.write("subject: %s time: %i\n" % (sbj, time))
                summary_writer.write("accuracy_1:%f\n" % accuracy_1)
                summary_writer.write("accuracy_0:%f\n" % accuracy_0)
                summary_writer.write("average:%f\n" % np.mean([accuracy_1, accuracy_0]))
#                with open("/home/wangnxr/results/%s_mni_diff_sbj.txt" % model_file.split("/")[-1].split(".")[0], "wb") as writer:
                 #       writer.write("recall:%f\n" % recall)
                 #       writer.write("precision:%f\n" % precision)
#                        writer.write("accuracy_1:%f\n" % accuracy_1)
#                        writer.write("accuracy_0:%f\n" % accuracy_0)

#                        for f, file in enumerate(files):
#                                writer.write("%s:%f\n" % (file, results[f][0]))
