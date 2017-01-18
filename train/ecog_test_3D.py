import keras
from keras.applications.vgg16 import VGG16
from keras.preprocessing.ecog3D import Ecog3DDataGenerator
from keras.layers import Flatten, Dense, Input, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Model, load_model
#from keras.imagenet_utils import decode_predictions, preprocess_input, _obtain_input_shape
import numpy as np
import pdb

test_datagen = Ecog3DDataGenerator(
        center=True
)

dgdx_val = test_datagen.flow_from_directory(
        #'/mnt/cb46fd46_5_no_offset/test/',
        '/mnt/ecog_dataset/d6532718/test_bal/',
        batch_size=24,
        shuffle=False,
        target_size=(8,8,1000,1),
        class_mode='binary')

#train_datagen.fit_generator(dgdx, nb_iter=100)
#test_datagen.fit_generator(dgdx_val, nb_iter=100)

validation_generator=dgdx_val

#base_model = VGG16(input_tensor=(Input(shape=(224, 224, 3))), include_top=False)



model = load_model("my_model_ecog3D.h5")
#model = load_model("my_model_ecog_1d.h5")
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

with open("ecog_3d_results.txt", "wb") as writer:
        writer.write("recall:%f\n" % recall)
        writer.write("precision:%f\n" % precision)
        writer.write("accuracy_1:%f\n" % accuracy_1)
        writer.write("accuracy_0:%f\n" % accuracy_0)

        for f, file in enumerate(files):
                writer.write("%s:%f\n" % (file, results[f][0]))
	





