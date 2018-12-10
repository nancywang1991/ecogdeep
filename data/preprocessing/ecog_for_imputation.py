'''Fairly basic set of tools for real-time data augmentation on image data.
Can easily be extended to include new transformations,
new preprocessing methods, etc...
'''
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import re
from scipy import linalg
import scipy.ndimage as ndi
from six.moves import range
import os
import threading
import warnings
from keras import backend as K
import copy
from ecogdeep.util.filter import butter_bandpass_filter
import pyedflib
import cPickle as pickle
import pdb


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
        x = x.transpose(0, 1)
    if scale:
        x += max(-np.min(x), 0)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
    return Image.fromarray(x[:, :].astype('uint8'), 'L')


def load_edf(path, channels=None):
    '''Load an edf into numpy format.

    # Arguments
        path: path to edf file
        channels: channels to keep
    '''
    signal = np.expand_dims(np.load(path)[:, :], 0)
    return signal[:, channels]


def list_edfs(directory, ext='npy'):
    return glob.glob(directory + "/*/*.npy") +glob.glob(directory + "/*/*/*.npy") 


class EcogDataGenerator(object):
    '''Generate minibatches with
    real-time data augmentation.

    # Arguments
        ablate_range: range of number of channels to set to 0
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode it is at index 3.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "th".
    '''

    def __init__(self,
                 ablate_range=None,
                 dim_ordering='default',
                 start_time=0,
		 test=False,
		 three_d = False,
                 seq_len=None):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.__dict__.update(locals())
        self.ablate_range = ablate_range
        self.seq_len = seq_len
        if dim_ordering not in {'th'}:
            raise ValueError('dim_ordering should be "th" (channel after row and '
                             'column) ', dim_ordering)
        self.dim_ordering = dim_ordering
        if dim_ordering == 'th':
            self.channel_index = 1
            self.row_index = 2

    def flow_from_directory(self, directory, batch_size=32, shuffle=True, seed=None,
                            pre_shuffle_ind=None, channels=None, ablate_range=None, spatial_shift = False):
        return DirectoryIterator(
            directory, self, dim_ordering=self.dim_ordering, batch_size=batch_size, shuffle=shuffle,
            seed=seed, pre_shuffle_ind=pre_shuffle_ind, channels=channels, ablate_range=ablate_range, seq_len=self.seq_len, spatial_shift=spatial_shift)

    def random_transform(self, x, test, seq_len, ablate_range, spatial_shift):
        channels_to_ablate = list(set(np.where(x != 0)[1]))
	if not test:
		if len(channels_to_ablate) < 15:
                	x_orig = copy.copy(x)
                	np.random.shuffle(channels_to_ablate)
                	x[0,channels_to_ablate[:1]] = 0
                	rand_start = np.random.randint(seq_len, x.shape[-1]-seq_len)
		else:
			x[0,channels_to_ablate[5:10]] = 0
			x_orig = copy.copy(x)
			channels_to_ablate = list(set(np.where(x != 0)[1]))
        		np.random.shuffle(channels_to_ablate)
        		x[0,channels_to_ablate[:np.random.randint(*ablate_range)]] = 0
			rand_start = np.random.randint(seq_len, x.shape[-1]-seq_len)
	else:
		x_orig = copy.copy(x)
		x[0,channels_to_ablate[5:10]] = 0
		rand_start = 0

	if spatial_shift:
	    x_grid = np.reshape(x, (10, 10, x.shape[-1]))
            x_grid_new = np.zeros(shape=x_grid.shape)
            xshift = np.random.randint(-1,1)
            yshift = np.random.randint(-1,1)
            x_grid_new[max(0,xshift):min(10, 10+xshift), max(0,yshift):min(10, 10+yshift)] = x_grid[max(0,-xshift):min(10, 10-xshift), max(0,-yshift):min(10, 10-yshift)]
            x = np.reshape(x_grid_new, (1, 100, x.shape[-1]))
	return x[0, :, rand_start:rand_start+seq_len], x_orig[0, :, rand_start:rand_start+seq_len]


class Iterator(object):
    def __init__(self, N, batch_size, shuffle, seed, pre_shuffle_ind):
        self.N = N
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(N, batch_size, shuffle, seed, pre_shuffle_ind)

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, N, batch_size=32, shuffle=False, seed=None, pre_shuffle_ind=None):
        # ensure self.batch_index is 0
        self.reset()
        while 1:
            if seed is not None:
                np.random.seed(seed + self.total_batches_seen)
            if self.batch_index == 0:
                self.index_array = np.arange(N)
                if shuffle:
                    self.index_array = np.random.permutation(N)
                if pre_shuffle_ind is not None:
                    np.random.seed(self.total_batches_seen)
                    self.index_array = np.random.permutation(N)
            current_index = (self.batch_index * batch_size) % N
            if N >= current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = N - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (self.index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def __iter__(self):
        # needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)


class DirectoryIterator(Iterator):
    def __init__(self, directory, EcogDataGenerator,
                 dim_ordering='default', test=False, three_d = False,
                 batch_size=32, shuffle=True, seed=None,
                 pre_shuffle_ind=None, channels=None, 
		 ablate_range=None, seq_len=None, spatial_shift = False):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.directory = directory
        self.ecog_data_generator = EcogDataGenerator
        self.dim_ordering = dim_ordering
        self.seq_len = seq_len
	self.three_d = three_d
	self.test = test
        self.image_shape = (1,len(channels), seq_len)
        self.channels = channels
        self.ablate_range = ablate_range
	self.spatial_shift = spatial_shift

        white_list_formats = {'npy'}

        # first, count the number of samples and classes
        self.nb_sample = 0

        def _recursive_list(subpath):
            return sorted(os.walk(subpath, followlinks=True), key=lambda tpl: tpl[0])

        for root, dirs, files in _recursive_list(directory):
            for fname in files:
                is_valid = False
                for extension in white_list_formats:
                    if fname.lower().endswith('.' + extension):
                        is_valid = True
                        break
                if is_valid:
                    self.nb_sample += 1
        print('Found %d edfs.' % (self.nb_sample))

        # second, build an index of the edfs in the different class subfolders
        self.filenames = []
        for root, dirs, files in _recursive_list(directory):
            for fname in sorted(files):
                is_valid = False
                for extension in white_list_formats:
                    if fname.lower().endswith('.' + extension):
                        is_valid = True
                        break
                if is_valid:
                    absolute_path = os.path.join(root, fname)
                    self.filenames.append(os.path.relpath(absolute_path, directory))
        super(DirectoryIterator, self).__init__(self.nb_sample, batch_size, shuffle, seed, pre_shuffle_ind)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel

        batch_x = np.zeros((current_batch_size,) + self.image_shape)
        batch_y = np.zeros(shape=(current_batch_size,self.image_shape[1]))
        if self.ecog_data_generator.three_d:
            batch_x = np.zeros((current_batch_size,) + (1, 10,10, self.image_shape[-1]))

	# build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            x = load_edf(os.path.join(self.directory, fname), self.channels)
            x, x_orig = self.ecog_data_generator.random_transform(x, self.ecog_data_generator.test, self.seq_len, self.ablate_range, self.spatial_shift)
	    if self.ecog_data_generator.three_d:
                x = np.reshape(x, (10,10,x.shape[-1]))
            batch_x[i,0] = x
	    batch_y[i] = x_orig[:,-1]
	    
	return batch_x, batch_y
