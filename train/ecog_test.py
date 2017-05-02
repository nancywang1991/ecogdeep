import keras
from keras.applications.vgg16 import VGG16
from keras.preprocessing.ecog import EcogDataGenerator
from keras.layers import Flatten, Dense, Input, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Model, load_model
#from keras.imagenet_utils import decode_predictions, preprocess_input, _obtain_input_shape
import numpy as np
import pdb

main_ecog_dir = '/home/wangnxr/dataset/ecog_vid_combined_c95_day7/'
#pre_shuffle_index = np.random.permutation(len(glob.glob('%s/train/*/*.npy' % main_ecog_dir)))
## Data generation ECoG
channels = np.hstack([np.arange(36), np.arange(37, 65), np.arange(66, 92)])
channels = np.arange(86)


test_datagen_edf = EcogDataGenerator(
    start_time=3400,
    center=True
)


dgdx_val_edf = test_datagen_edf.flow_from_directory(
    #'/mnt/cb46fd46_5_no_offset/test/',
    '%s/test/' % main_ecog_dir,
    batch_size=10,
    shuffle=False,
    target_size=(1,len(channels),1000),
    final_size=(1,len(channels),1000),
    channels = channels,
    class_mode='binary')



validation_generator =  dgdx_val_edf

#for layer in base_model.layers[:10]:
#    layer.trainable = False
model_file = "/home/wangnxr/models/ecog_model_alexnet_3towers_dense1_c95_3400_chkpt.h5"
model = load_model(model_file)

#pdb.set_trace()
files = dgdx_val_edf.filenames
results = model.predict_generator(validation_generator, len(files))
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

#with open("ecog_3d_results.txt", "wb") as writer:
#=======
#pdb.set_trace()

with open("/home/wangnxr/results/%s.txt" % model_file.split("/")[-1].split(".")[0], "wb") as writer:
        writer.write("recall:%f\n" % recall)
        writer.write("precision:%f\n" % precision)
        writer.write("accuracy_1:%f\n" % accuracy_1)
        writer.write("accuracy_0:%f\n" % accuracy_0)

        for f, file in enumerate(files):
                writer.write("%s:%f\n" % (file, results[f][0]))
