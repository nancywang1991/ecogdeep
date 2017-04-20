import keras
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image2 import ImageDataGenerator
from keras.layers import Flatten, Dense, Input, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model, load_model
import numpy as np
import pdb

sbj_ids = ['a0f', 'e5b', 'd65']
days = [8,9,9]
start_times = [3200, 3600, 4000]
frames = [ range(3,8), range(5,10), range(7,12)]
channels_list = [np.hstack([np.arange(36), np.arange(37, 68), np.arange(68, 92)]), np.arange(111), np.arange(82)]

for s, sbj in enumerate(sbj_ids):
    main_vid_dir = '/home/wangnxr/dataset/ecog_vid_combined_%s_day%i/' % (sbj, days[s])
    for t, time in enumerate(start_times):
        mode = 'categorical'
        test_datagen = ImageDataGenerator(
            rescale=1./255,
            center_crop=(224, 224),
            keep_frames=frames[t])

        dgdx_val = test_datagen.flow_from_directory(
            '/%s/test/' % main_vid_dir,
            img_mode="seq",
            read_formats={'png'},
            target_size=(int(224), int(224)),
            #resize_size = (int(340), int(256)),
            num_frames=12,
            batch_size=10,
            shuffle=False,
            class_mode=mode)

        validation_generator=dgdx_val
        #for layer in base_model.layers[:10]:
        #    layer.trainable = False
        model_file = "/home/wangnxr/models/vid_model_lstm_%s_5st_t_%i_chkpt.h5" % (sbj, time)
        model = load_model(model_file)

        #pdb.set_trace()
        files = validation_generator.filenames
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


        with open("/home/wangnxr/results/%s.txt" % model_file.split("/")[-1].split(".")[0], "wb") as writer:
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
                    writer.write("%s:%i->%f\n" % (file, np.argmax(results[f]), np.max(results[f])))


