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

test_datagen = EcogDataGenerator(
        center=True
)

dgdx_val = test_datagen.flow_from_directory(
        #'/mnt/cb46fd46_5_no_offset/test/',
        '/home/nancy/Documents/ecog_dataset/d6532718/test/',
        batch_size=24,
        shuffle=False,
        target_size=(64,1000,1),
        class_mode='binary')

#train_datagen.fit_generator(dgdx, nb_iter=100)
#test_datagen.fit_generator(dgdx_val, nb_iter=100)

validation_generator=dgdx_val

#base_model = VGG16(input_tensor=(Input(shape=(224, 224, 3))), include_top=False)


model = load_model("ecog_model_1d.h5")
#pdb.set_trace()
files = validation_generator.filenames
results = model.predict_generator(validation_generator, validation_generator.nb_sample)

with open("ecog_1d_results.txt", "wb") as writer:
        for f, file in enumerate(files):
                writer.write("%s:%f\n" % (file, results[f][0]))






