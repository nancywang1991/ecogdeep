from keras import backend as K
import numpy as np
from keras.models import load_model
from scipy.misc import imsave
import pdb
from ecogdeep.train.sbj_parameters import *
import glob

K.set_learning_phase(0)
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

sbj_to_do = ["a0f"]
for s, sbj in enumerate(sbj_ids):
    if sbj not in sbj_to_do:
	continue
    for t, time in enumerate(start_times[1:]):
        for itr in xrange(1):
            model_files = glob.glob('/home/wangnxr/models/ecog_vid_model_lstm_%s_itr_1_t_%i_*chkpt.h5' % (sbj, time))
            if len(model_files) == 0:
                continue
            model = load_model(model_files[0])
            layer_dict = dict([(layer.name, layer) for layer in model.layers])
            layer_name = model.layers[-7].name
            input_img =model.input
            step=0.5
            for filter_index in xrange(layer_dict[layer_name].output_shape[-1]-50):
                # build a loss function that maximizes the activation
                # of the nth filter of the layer considered
                layer_output = layer_dict[layer_name].output
		
                loss = K.mean(layer_output[:,:,filter_index])

                # compute the gradient of the input picture wrt this loss
                grads = K.gradients(loss, input_img)
                 
                # normalization trick: we normalize the gradient
                grads[0] /= (K.sqrt(K.mean(K.square(grads[0]))) + 1e-5)
		grads[1] /= (K.sqrt(K.mean(K.square(grads[1]))) + 1e-5)
		
                # this function returns the loss and grads given the input picture
                iterate0 = K.function(input_img, [loss, grads[0]])
		iterate1 = K.function(input_img, [loss, grads[1]])

                # we start from a gray image with some noise
		#pdb.set_trace()
                input_img_data = [np.random.random([1] + input_img[0]._shape_as_list()[1:]) * 20 + 128., np.random.random([1] + input_img[1]._shape_as_list()[1:]) * 20 + 128.]
                # run gradient ascent for 40 steps
                for i in range(1000):
		#	pdb.set_trace()
                        loss_value, grads_value0 = iterate0(input_img_data)
			loss_value, grads_value1 = iterate1(input_img_data)
			#print loss_value
                        input_img_data[0] += grads_value0 * step
			input_img_data[1] += grads_value1 * step


                img0 = np.hstack(input_img_data[0][0])
                img0 = deprocess_image(img0)
		img1 = np.hstack(input_img_data[1][0])
                img1 = deprocess_image(img1)

                imsave('/home/wangnxr/results/weights/ecog_vid_lstm_%s_%i_%s_filter_%d_img0.png' % (sbj, time, layer_name, filter_index), img0[:,:,:])
		imsave('/home/wangnxr/results/weights/ecog_vid_lstm_%s_%i_%s_filter_%d_img1.png' % (sbj, time, layer_name, filter_index), img1[:,:,0])
