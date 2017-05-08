from keras import backend as K
import numpy as np
from keras.models import load_model
from scipy.misc import imsave
import pdb
from ecogdeep.train.sbj_parameters import *
import glob


def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

for s, sbj in enumerate(sbj_ids):
    for t, time in enumerate(start_times):
        for itr in xrange(3):
            model_files = glob.glob('/home/wangnxr/models/ecog_vid_model_lstm_%s_itr_%i_t_%i_*chkpt.h5' % (sbj, itr, time))
            if len(model_files) == 0:
                continue
            model = load_model(model_files[0])
            layer_dict = dict([(layer.name, layer) for layer in model.layers])
            layer_name = "merge1"
            input_img = model.input
            step=0.5

            for filter_index in layer_dict[layer_name].output_shape[0]:
                # build a loss function that maximizes the activation
                # of the nth filter of the layer considered
                layer_output = layer_dict[layer_name].output
                loss = K.mean(layer_output[:, filter_index, :, :])

                # compute the gradient of the input picture wrt this loss
                grads = K.gradients(loss, input_img)[0]

                # normalization trick: we normalize the gradient
                grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

                # this function returns the loss and grads given the input picture
                iterate = K.function([input_img], [loss, grads])

                # we start from a gray image with some noise
                input_img_data = np.random.random((1, 1, 64, 200)) * 20 + 128.
                # run gradient ascent for 40 steps
                for i in range(2000):
                        loss_value, grads_value = iterate([input_img_data])
                        input_img_data += grads_value * step


                img = input_img_data[0]
                img = deprocess_image(img)
                imsave('/home/wangnxr/results/weights/%s_filter_%d.png' % (layer_name, filter_index), img[:,:,0])
