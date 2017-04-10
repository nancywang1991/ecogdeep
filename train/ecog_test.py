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
        center=True,
        start_time=3800,
)
type = "binary"
channels = np.hstack([np.arange(36), np.arange(37, 68), np.arange(68, 92)])
dgdx_val = test_datagen.flow_from_directory(
        #'/mnt/cb46fd46_5_no_offset/test/',
        '/home/wangnxr/dataset/ecog_offset_0_a0f_day6/val/',
        batch_size=20,
        shuffle=False,
        channels = channels,
        target_size=(1,len(channels), 1000),
        final_size = (1, len(channels), 1000),
        class_mode='binary')

#train_datagen.fit_generator(dgdx, nb_iter=100)
#test_datagen.fit_generator(dgdx_val, nb_iter=100)

validation_generator=dgdx_val

#base_model = VGG16(input_tensor=(Input(shape=(224, 224, 3))), include_top=False)


#model = load_model("my_model_ecog3D.h5")
model = load_model("/home/wangnxr/models/model_ecog_1d_offset_150_1_3_1_2_a0f.h5")
files = validation_generator.filenames
results = model.predict_generator(validation_generator, len(files))

true = validation_generator.classes
if type == "binary":
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

with open("/home/wangnxr/ecog_1d_results_pred_45_1day.txt", "wb") as writer:
	if type=="binary":
        	writer.write("recall:%f\n" % recall)
        	writer.write("precision:%f\n" % precision)
        	writer.write("accuracy_1:%f\n" % accuracy_1)
        	writer.write("accuracy_0:%f\n" % accuracy_0)

        for f, file in enumerate(files):
		try:
                	#writer.write("%s:%f\n" % (file, np.argmax(results[f])))
                         writer.write("%s:%f\n" % (file, np.round(results[f][0])))
		except:
			pass
	





