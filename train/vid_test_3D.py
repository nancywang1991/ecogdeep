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
        target_size=(int(256*(224/192.0)), int(192*(224/192.0))),
        batch_size=32,
        shuffle=False,
        class_mode=None)


test_datagen.fit_generator(dgdx_val, nb_iter=100)

validation_generator=dgdx_val

base_model = VGG16(input_tensor=(Input(shape=(224, 224, 3))), weights='imagenet', include_top=False)
#base_model = VGG16(input_tensor=(Input(shape=(224, 224, 3))), include_top=False)


#for layer in base_model.layers[:10]:
#    layer.trainable = False

model = load_model("my_model.h5")

results = model.predict_generator(validation_generator, 32*5)
#pdb.set_trace()

