import numpy as np


sbj_ids = ['a0f', 'd65', 'e5b', 'cb4', 'c95']
days = [11,9,9,10,7]
start_times = [2700, 3300, 3900]
channels_list = [np.hstack([np.arange(36), np.arange(37, 65), np.arange(66, 92)]),
                 np.arange(80),
                 np.arange(82),
<<<<<<< HEAD
		 np.arange(80),
=======
		         np.arange(80),
>>>>>>> 3cd0685d61165433807c318e864dfed3749403e8
                 np.arange(86)]
frames = [ range(3,8), range(5,10), range(7,12)]
