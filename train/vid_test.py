import keras
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator, center_crop
from keras.layers import Flatten, Dense, Input, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model, load_model
import numpy as np
import pdb


test_datagen = ImageDataGenerator(rescale=1./255)

test_datagen.config['center_crop_size'] = (227,227)
test_datagen.set_pipeline([center_crop])

dgdx_val = test_datagen.flow_from_directory(
        '/home/wangnxr/dataset/vid_offset_0/val/',
        shuffle=False,
        read_formats={'png'},
        num_frames=10,
        frame_ind=9,
        target_size=(int(340), int(256)),
        batch_size=10,
        class_mode='binary')
test_datagen.fit_generator(dgdx_val, nb_iter=len(dgdx_val.filenames)/10)

validation_generator=dgdx_val

#for layer in base_model.layers[:10]:
#    layer.trainable = False
pdb.set_trace()
model = load_model("/home/wangnxr/vid_model_alexnet_2towers_dense1_5_sec.h5")

#pdb.set_trace()
files = validation_generator.filenames
results = model.predict_generator(validation_generator, len(files))
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

#with open("ecog_3d_results.txt", "wb") as writer:
#=======
#pdb.set_trace()

with open("ecog_1d_results.txt", "wb") as writer:
        writer.write("recall:%f\n" % recall)
        writer.write("precision:%f\n" % precision)
        writer.write("accuracy_1:%f\n" % accuracy_1)
        writer.write("accuracy_0:%f\n" % accuracy_0)

        for f, file in enumerate(files):
                writer.write("%s:%f\n" % (file, results[f][0]))

