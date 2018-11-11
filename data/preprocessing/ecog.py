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
from ecogdeep.util.filter import butter_bandpass_filter
import cPickle as pickle
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

def load_edf(path, start_time, channels=None, ablate=None):
    '''Load an edf into numpy format.

    # Arguments
        path: path to edf file
        channels: channels to keep
    '''
    signal = np.expand_dims(np.load(path)[:,:],0)
    for c in channels:#xrange(signal.shape[1]):
        try:
	    if np.any(signal[:,c]) != 0:
            	signal[0,c] = butter_bandpass_filter(signal[:,c],10,200, 1000)
            	signal[0,c] = (signal[0,c] - np.mean(signal[0, c, :3500]))/np.std(signal[0, c, :3500])
        except:
            print(path)
            pass
    if ablate:
        for c in ablate:
            signal[0, c] = np.mean(signal[0,c])
    return signal[:,channels, start_time-100:]


def list_edfs(directory, ext='npy'):
    return [os.path.join(root, f)
            for root, dirs, files in os.walk(directory) for f in files
            if re.match('([\w]+\.(?:' + ext + '))', f)]


class EcogDataGenerator(object):
    '''Generate minibatches with
    real-time data augmentation.

    # Arguments
        featurewise_center: set input mean to 0 over the dataset.
        samplewise_center: set each sample mean to 0.
        featurewise_std_normalization: divide inputs by std of the dataset.
        samplewise_std_normalization: divide each input by its std.
        zca_whitening: apply ZCA whitening.
        time_shift_range: milliseconds to shift.
        gaussian_noise_range: amount of gaussian noise to add to data
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode it is at index 3.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "th".
    '''
    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 time_shift_range=None,
                 spatial_shift=False,
		 gaussian_noise_range=None,
                 fft = False,
                 three_d = False,
                 f_lo = 0,
                 f_hi = 0,
                 samp_rate = 1000,
                 center=True,
                 dim_ordering='default',
                 start_time = 0,
                 seq_len=None, seq_num=None, seq_st=None):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.__dict__.update(locals())
        self.mean = None
        self.std = None
        self.principal_components = None
        self.gaussian_noise_range = gaussian_noise_range
        self.time_shift_range = time_shift_range
        self.spatial_shift = spatial_shift
	self.f_lo=f_lo
        self.f_hi=f_hi
        self.samp_rate=samp_rate
        self.fft=fft
        self.three_d = three_d
        self.start_time=start_time
        self.seq_len = seq_len
        self.seq_num = seq_num
        self.seq_st = seq_st
        if dim_ordering not in {'th'}:
            raise ValueError('dim_ordering should be "th" (channel after row and '
                             'column) ', dim_ordering)
        self.dim_ordering = dim_ordering
        if dim_ordering == 'th':
            self.channel_index = 1
            self.row_index = 2


    def flow_from_directory(self, directory,
                            target_size=None, final_size=None,
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None, save_prefix='', save_format='jpeg',color_mode="rgb",
                            follow_links=False, pre_shuffle_ind = None, channels=None, ablate=None):
        return DirectoryIterator(
            directory, self,
            target_size=target_size, final_size=final_size, color_mode=color_mode,
            classes=classes, class_mode=class_mode,
            dim_ordering=self.dim_ordering,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format,
            follow_links=follow_links, pre_shuffle_ind=pre_shuffle_ind, channels=channels, ablate=ablate)

    def standardize(self, x, target_size):
        if self.center:
            shift = self.time_shift_range/2
            x = x[:,:,shift:shift+target_size[-1]]

        # x is a single image, so it doesn't have image number at index 0
        ecog_channel_index = self.channel_index - 1

        if self.samplewise_center:
            x -= np.mean(x, axis=ecog_channel_index, keepdims=True)
        if self.samplewise_std_normalization:
            x /= (np.std(x, axis=ecog_channel_index, keepdims=True) + 1e-7)

        if self.featurewise_center:
            if self.mean is not None:
                x -= self.mean
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`featurewise_center`, but it hasn\'t'
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        if self.featurewise_std_normalization:
            if self.std is not None:
                x /= (self.std + 1e-7)
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`featurewise_std_normalization`, but it hasn\'t'
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        if self.zca_whitening:
            if self.principal_components is not None:
                flatx = np.reshape(x, (x.size))
                whitex = np.dot(flatx, self.principal_components)
                x = np.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2]))
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`zca_whitening`, but it hasn\'t'
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        if self.seq_len:
            x_temp = np.zeros(shape=(self.seq_num, x.shape[0],x.shape[1], self.seq_len))
            for i in xrange(self.seq_num):
                x_temp[i] = x[:,:,(i*self.seq_st):(i*self.seq_st + self.seq_len)]
            x= x_temp
        return x

    def random_transform(self, x, target_size):
        if self.gaussian_noise_range:
            if np.random.randint(100) < 25:
                noise = np.random.normal(0,self.gaussian_noise_range, x.shape)
                x = x + noise
        if self.time_shift_range and not self.center:
            if target_size[-1]+self.time_shift_range > x.shape[-1]:
                print("time shift must be less than %i" % (x.shape[-1]-target_size[-1]))
                raise ValueError
            if np.random.randint(100) < 10:
                shift = np.random.randint(self.time_shift_range)
            else:
                shift = self.time_shift_range/2
            x = x[:,:,shift:(shift+target_size[-1])]
        if self.spatial_shift:
            x_grid = np.reshape(x, (10, 10, x.shape[-1]))
            x_grid_new = np.zeros(shape=x_grid.shape)
            xshift = np.random.randint(-1,1)
            yshift = np.random.randint(-1,1)
            x_grid_new[max(0,xshift):min(10, 10+xshift), max(0,yshift):min(10, 10+yshift)] = x_grid[max(0,-xshift):min(10, 10-xshift), max(0,-yshift):min(10, 10-yshift)]
            x = np.reshape(x_grid_new, (1, 100, x.shape[-1]))
	return x

    def freq_transform(self, x, f_lo,f_hi, samp_rate):
        f_hi = int(f_hi*(x.shape[-1]/float(samp_rate)))
        f_lo = int(f_lo * (x.shape[-1] / float(samp_rate)))
        freq = np.zeros(shape=(x.shape[0], x.shape[1], 2))
        for c in xrange(freq.shape[1]):
        	freq[0,c,0] = np.mean((np.abs((np.fft.fft(x[0,c])) ** 2))[10:30])
		freq[0,c,1] = np.mean((np.abs((np.fft.fft(x[0,c])) ** 2))[70:100])
        return freq

    def fit(self, X,
            augment=False,
            rounds=1,
            seed=None):
        '''Required for featurewise_center, featurewise_std_normalization
        and zca_whitening.

        # Arguments
            X: Numpy array, the data to fit on. Should have rank 4.
                In case of grayscale data,
                the channels axis should have value 1, and in case
                of RGB data, it should have value 3.
            augment: Whether to fit on randomly augmented samples
            rounds: If `augment`,
                how many augmentation passes to do over the data
            seed: random seed.
        '''
        X = np.asarray(X)
        if X.ndim != 4:
            raise ValueError('Input to `.fit()` should have rank 3. '
                             'Got array with shape: ' + str(X.shape))

        if seed is not None:
            np.random.seed(seed)

        X = np.copy(X)
        if augment:
            aX = np.zeros(tuple([rounds * X.shape[0]] + list(X.shape)[1:]))
            for r in range(rounds):
                for i in range(X.shape[0]):
                    aX[i + r * X.shape[0]] = self.random_transform(X[i])
            X = aX

        if self.featurewise_center:
            self.mean = np.mean(X, axis=(0, self.row_index))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_index - 1] = X.shape[self.channel_index]
            self.mean = np.reshape(self.mean, broadcast_shape)
            X -= self.mean

        if self.featurewise_std_normalization:
            self.std = np.std(X, axis=(0, self.row_index))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_index - 1] = X.shape[self.channel_index]
            self.std = np.reshape(self.std, broadcast_shape)
            X /= (self.std + K.epsilon())

        if self.zca_whitening:
            flatX = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
            sigma = np.dot(flatX.T, flatX) / flatX.shape[0]
            U, S, V = linalg.svd(sigma)
            self.principal_components = np.dot(np.dot(U, np.diag(1. / np.sqrt(S + 10e-7))), U.T)


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
                 target_size=None, final_size=None, color_mode='rgb',
                 dim_ordering='default',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg',
                 follow_links=False, pre_shuffle_ind=None, start_time=0, channels=None, ablate= None):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.directory = directory
        self.ecog_data_generator = EcogDataGenerator
        if target_size:
            self.target_size = tuple(target_size)
        else:
            self.target_size = None
        self.final_size = tuple(final_size)
        #if color_mode not in {'grayscale'}:
        #    raise ValueError('Invalid color mode:', color_mode,
        #                     '; expected "grayscale".')
        self.color_mode = color_mode
        self.dim_ordering = dim_ordering
        self.image_shape = self.final_size
        self.classes = classes
        self.channels = channels
        if class_mode not in {'categorical', 'binary', 'sparse', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.ablate = ablate

        white_list_formats = {'npy'}

        # first, count the number of samples and classes
        self.nb_sample = 0

        if not classes:
            classes = []
            for subdir in sorted(os.listdir(directory)):
                if os.path.isdir(os.path.join(directory, subdir)):
                    classes.append(subdir)
        self.nb_class = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))

        def _recursive_list(subpath):
            return sorted(os.walk(subpath, followlinks=follow_links), key=lambda tpl: tpl[0])

        for subdir in classes:
            subpath = os.path.join(directory, subdir)
            for root, dirs, files in _recursive_list(subpath):
                for fname in files:
                    is_valid = False
                    for extension in white_list_formats:
                        if fname.lower().endswith('.' + extension):
                            is_valid = True
                            break
                    if is_valid:
                        self.nb_sample += 1
        print('Found %d edfs belonging to %d classes.' % (self.nb_sample, self.nb_class))

        # second, build an index of the edfs in the different class subfolders
        self.filenames = []
        self.classes = np.zeros((self.nb_sample,), dtype='int32')
        i = 0
        for subdir in classes:
            subpath = os.path.join(directory, subdir)
            for root, dirs, files in _recursive_list(subpath):
                for fname in sorted(files):
                    is_valid = False
                    for extension in white_list_formats:
                        if fname.lower().endswith('.' + extension):
                            is_valid = True
                            break
                    if is_valid:
                        self.classes[i] = self.class_indices[subdir]
                        i += 1
                        # add filename relative to directory
                        absolute_path = os.path.join(root, fname)
                        self.filenames.append(os.path.relpath(absolute_path, directory))
        super(DirectoryIterator, self).__init__(self.nb_sample, batch_size, shuffle, seed, pre_shuffle_ind)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        if self.ecog_data_generator.seq_num:
            batch_x = np.zeros(shape=((current_batch_size,self.ecog_data_generator.seq_num) + self.image_shape))
        else:
            batch_x = np.zeros((current_batch_size,) + self.image_shape)
	if self.ecog_data_generator.three_d:
            batch_x = np.zeros((current_batch_size,) + (5, 1, 10,10, self.image_shape[-1]))

        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            x = load_edf(os.path.join(self.directory, fname), self.ecog_data_generator.start_time, self.channels, self.ablate)
            x = self.ecog_data_generator.random_transform(x, self.target_size)
            x = self.ecog_data_generator.standardize(x, self.target_size)
            if self.ecog_data_generator.fft:
                x = self.ecog_data_generator.freq_transform(x, self.ecog_data_generator.f_lo, self.ecog_data_generator.f_hi, self.ecog_data_generator.samp_rate)
            if self.ecog_data_generator.three_d:
                new_x = []
                for t in range(x.shape[0]):
                    new_x.append(np.expand_dims(np.reshape(x[t][0], (10, 10, x.shape[-1])), 0))
            	x = np.array(new_x)
	    batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.dim_ordering, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype('float32')
        elif self.class_mode == 'categorical':
            batch_y = np.zeros(shape=((len(batch_x), self.nb_class)), dtype='float32')
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x
        return batch_x, batch_y
