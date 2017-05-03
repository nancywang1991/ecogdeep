import numpy as np


sbj_ids = ['a0f', 'e5b', 'd65', 'cb4', 'c95']
days = [11,9,9,10,7]
start_times = [2700, 3300, 3900]
channels_list = [np.hstack([np.arange(36), np.arange(37, 65), np.arange(66, 92)]),
                 np.arange(82),
                 np.hstack([np.arange(80), np.arange(81, 85), np.arange(86, 104),np.arange(105, 108), np.arange(110, 111)]),
		         np.arange(80),
                 np.arange(86)]