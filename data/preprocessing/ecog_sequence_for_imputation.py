import copy
import numpy as np
import keras
import re
from scipy import linalg
import scipy.ndimage as ndi
from six.moves import range
import os
import threading
import warnings
from keras import backend as K
from ecogdeep.util.filter import butter_bandpass_filter
import pdb

def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x, transform_matrix, channel_index=0, fill_mode='nearest', cval=0.):
    x = np.rollaxis(x, channel_index, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
                      final_offset, order=0, mode=fill_mode, cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index+1)
    return x

def array_to_img(x, dim_ordering='default', scale=True):
    from PIL import Image
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError('Expected image array to have rank 2 (single edf image). '
                         'Got array with shape:', x.shape)

    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    if dim_ordering not in {'th', 'tf'}:
        raise ValueError('Invalid dim_ordering:', dim_ordering)

    # Original Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but target PIL image has format (width, height, channel)
    if dim_ordering == 'th':
        x = x.transpose(0,1)
    if scale:
        x += max(-np.min(x), 0)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
    return Image.fromarray(x[:, :].astype('uint8'), 'L')

def list_edfs(directory, ext='npy'):
    return [os.path.join(root, f)
            for root, dirs, files in os.walk(directory) for f in files
            if re.match('([\w]+\.(?:' + ext + '))', f)]

class EcogDataGenerator(keras.utils.Sequence):
    'Generates ECoG data for Keras'
    def __init__(self, list_IDs, seq_len = None, output_len = None, ablate_range=None, spatial_shift=False, test=False,  batch_size=32, dim=(32,32,32), shuffle=True):
        'Initialization'
        self.dim = dim
	self.spatial_shift = spatial_shift
        self.batch_size = batch_size
        self.list_IDs = list_IDs
	self.ablate_range = ablate_range
	self.test = test
	self.seq_len = seq_len
	self.output_len = output_len
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def load_edf(self, ID):
	signal = np.load(ID)
	return signal
    
    def random_transform(self, x):
	channels_to_ablate = np.where(x[:,0] != 0)[0]
	if not self.test:
	    if len(channels_to_ablate) < 15:
               	x_orig = copy.copy(x)
               	np.random.shuffle(channels_to_ablate)
               	x[channels_to_ablate[:1]] = 0
               	rand_start = np.random.randint(self.seq_len, x.shape[-1]-self.seq_len)
            else:
		x[channels_to_ablate[5:10]] = 0
		x_orig = copy.copy(x)
		channels_to_ablate = np.where(x[:,0] != 0)[0]
       		np.random.shuffle(channels_to_ablate)
       		x[channels_to_ablate[:np.random.randint(*self.ablate_range)]] = 0
		rand_start = np.random.randint(self.seq_len, x.shape[-1]-self.seq_len)
	else:
		x_orig = copy.copy(x)
		x[channels_to_ablate[5:10]] = 0
		rand_start = 0

	if self.spatial_shift:
	    sqrt_size = np.sqrt(x.shape[0])
	    if int(sqrt_size) != sqrt_size:
		print("Input grid is not square, currently not supported" )
                raise ValueError

            x_grid = np.reshape(x, (int(sqrt_size), int(sqrt_size), x.shape[-1]))
            x_grid_new = np.zeros(shape=x_grid.shape)
            xshift = np.random.randint(-1,1)
            yshift = np.random.randint(-1,1)
            x_grid_new[max(0,xshift):min(10, 10+xshift), max(0,yshift):min(10, 10+yshift)] = x_grid[max(0,-xshift):min(10, 10-xshift), max(0,-yshift):min(10, 10-yshift)]
            x = np.reshape(x_grid_new, (int(sqrt_size**2), x.shape[-1]))
	return x[:,rand_start:rand_start+self.seq_len], x_orig[:,rand_start:rand_start+self.seq_len]

    def standardize(self, x):
	for c in range(x.shape[0]):
	    if np.any(x[c]) != 0:
                x[c] = butter_bandpass_filter(x[c],10,200, 1000)
                x[c] = (x[c] - np.mean(x[c, :3500]))/np.std(x[c, :3500])
	x = x[:,(self.start_time-self.time_shift_range):]
	if self.center:
	    shift = self.time_shift_range/2
            x = x[:,shift:shift+self.dim[-1]]
	return x
    
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size, 1, self.dim[0], self.dim[1], self.seq_len))
        y = np.zeros((self.batch_size,self.output_len*self.dim[0]*self.dim[1]))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            x = self.load_edf(ID)
	    x, x_orig = self.random_transform(x)
	    X[i,0] = np.reshape(x, (self.dim[0], self.dim[1],self.seq_len))
	    
            # Store class
            y[i] = np.ndarray.flatten(x_orig[:, -self.output_len:])
	return X, y
