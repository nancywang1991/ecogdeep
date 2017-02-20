import keras
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator, center_crop
from keras.layers import Flatten, Dense, Input, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model, load_model
import numpy as np
import pdb

test_datagen = ImageDataGenerator(rescale=1./255)

test_datagen.config['center_crop_size'] = (224,224)
test_datagen.set_pipeline([center_crop])

dgdx_val = test_datagen.flow_from_directory(
        '/home/nancy/mvmt_vid_dataset/test/',
        read_formats={'png'},
        target_size=(300, 224),
        batch_size=32,
        shuffle=False,
        class_mode=None)


test_datagen.fit_generator(dgdx_val, nb_iter=100)

validation_generator=dgdx_val

#for layer in base_model.layers[:10]:
#    layer.trainable = False

model = load_model("my_model.h5")

#pdb.set_trace()
files = validation_generator.filenames
results = model.predict_generator(validation_generator, validation_generator.nb_sample)
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

