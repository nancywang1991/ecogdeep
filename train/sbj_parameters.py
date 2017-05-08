import numpy as np


sbj_ids = ['a0f', 'd65', 'e5b', 'cb4', 'c95']
days = [11,9,9,10,7]
start_times = [2700, 3300, 3900]
channels_list = [np.hstack([np.arange(36), np.arange(37, 65), np.arange(66, 92)]),
                 np.arange(80),
                 np.arange(82),
		         np.arange(80),
                 np.arange(86)]
frames = [ range(3,8), range(5,10), range(7,12)]