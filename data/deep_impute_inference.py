import keras
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense, Input, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Model, load_model
from keras.utils.generic_utils import get_custom_objects
from ecogdeep.imputation.conv_mlp import selected_loss
from copy import copy
#from keras.imagenet_utils import decode_predictions, preprocess_input, _obtain_input_shape
import numpy as np
import pdb
import glob
import os

def main(sbjs, model_file, mask_withheld, savename, input_len=80):
    for sbj in sbjs:
        # Params to set
        main_ecog_dir = '/data2/users/wangnxr/dataset/ecog_mni_ellip_%s' % (sbj)
        files = glob.glob("%s/val/*/*.npy" % main_ecog_dir)

        model = load_model(model_file, custom_objects={"loss":loss})
        loss = selected_loss(input=np.zeros(shape=(1, 1, 1, 1), dtype='float32'))
        for file in sorted(files):
            print file
            orig = np.load(file)
            if mask_withheld:
                orig[np.where(orig[:,0]!=0)[0][5:10]] = 0
            new = copy(orig)
            fill_inds = np.where(orig[:,0]==0)[0]
            orig_batch = np.zeros(shape=(orig.shape[-1],1,10,10, input_len))
            for t in xrange(input_len, orig.shape[-1]):
                orig_batch[t] = np.reshape(orig[:,t-input_len:t], (10,10,input_len))
            #Impute
            new[fill_inds] = model.predict(orig_batch)[:,fill_inds].T
            try:
                np.save("%s/%s_%s/%s" % ("/".join(main_ecog_dir.split("/")[:-1]), savename, sbj, "/".join(file.split("/")[-3:])), new)
            except IOError:
                os.makedirs("%s/%s_%s/%s" % ("/".join(main_ecog_dir.split("/")[:-1]), savename, sbj, "/".join(file.split("/")[-3:-1])))
                np.save("%s/%s_%s/%s" % ("/".join(main_ecog_dir.split("/")[:-1]), savename, sbj, "/".join(file.split("/")[-3:])), new)


if __name__ == "__main__":
    sbjs_to_do = ["a0f", "cb4", "c95", "d65"]
    model_file = "/home/wangnxr/models/ecog_model_ellip_impute_sequence_bothloss_skip_10output_allplus_itr_0_3d_best.h5"
    savename = "ecog_mni_ellip_deep_impute_analyallsbjskipbothloss"
    # Enable if testing withheld electrode reconstruction. Otherwise, when producing imputed files, turn to False
    mask_withheld = True
    main(sbjs_to_do, model_file, mask_withheld, savename)