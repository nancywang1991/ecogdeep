import pyedflib
import os
import pdb
import glob
import numpy as np
import datetime
from ecogdeep.data.mni_space_ecog_data_transform import *

data_main = '/data1/ecog_project/edf/'
save_main = '/data2/users/wangnxr/dataset/standardized_clips_ellip/'
subjects = ['fcb01f7a', 'cb46fd46', 'd6532718', 'ab2431d9', 'a0f66459', 'e5bad42f', 'ec374ad0', 'c95c1e82', 'ffb52f92', 'c7980193', 'aa97abcd', 'b4ac1726', 'b45e3f7b']
mni_dir = '/home/wangnxr/Documents/mni_coords/'
subject_id_map = {'cdceeb':'fcb01f7a', 'ecb43e':'cb46fd46', '69da36':'d6532718', 'b3719b':'ab2431d9', '294e1c':'a0f66459', '0b5a2e':'e5bad52f', '0a80cf':'ec374ad0', 'c5a5e9':'c95c1e82', 'fca96e':'ffb52f92', '3f2113':'c7980193', 'acabb1':'aa97abcd', 'c19968':'b4ac1726', '13d2d8':'b45e3f7b'}

for gridlabid, sbj in subject_id_map.iteritems():
	if not os.path.exists("%s/%s" % (data_main, sbj)):
		continue
	files = sorted(glob.glob("%s/%s/*.edf" % (data_main, sbj)))
	mni_file = np.loadtxt("%s/%s_Trodes_MNIcoords.txt" % (mni_dir, gridlabid), delimiter=",")
        mapping=VirtualGrid_ellipsoid_mapping(mni_file)
	try:
		os.makedirs("%s/%s/train/" % (save_main, sbj))
		os.makedirs("%s/%s/test/" % (save_main, sbj))
		os.makedirs("%s/%s/val/" % (save_main, sbj))
	except OSError:
		pass

	for file in files:
		data = pyedflib.EdfReader(file)
		start_time = data.getStartdatetime()
		n_channels = 64
		mean = np.zeros(n_channels)
		std = np.zeros(n_channels)
		if data.file_duration < 60000:
                        #Too short
                        continue
		for c in range(1,n_channels+1):
			print "normalising channel %i" % (c-1)
			temp_data = data.readSignal(c, 0, 3*60*60*1000)
			mean[c-1] = np.mean(temp_data)
			std[c-1] = np.std(temp_data)	
		for t in range(120000,data.file_duration*1000-5000, 120000):
			if (start_time + datetime.timedelta(t/1000)).hour > 7:
				chunk = np.zeros(shape=(100, 5000))
				for old_c, new_c in mapping.items():
				    if old_c<64: 
				        chunk[new_c] = (data.readSignal(old_c+1, start=t, n=5000)-mean[old_c])/std[old_c]
				if not np.any(np.abs(chunk) > 2):
				    randnum = np.random.randint(100)
				    print "Saving time=%i from file %s" % (t, file)
				    if randnum<90:
				        np.save("%s/%s/train/%s_t_%i" % (save_main, sbj, file.split("/")[-1].split(".")[0], t/1000), chunk)
				    elif randnum<95:
					np.save("%s/%s/val/%s_t_%i" % (save_main, sbj, file.split("/")[-1].split(".")[0], t/1000), chunk)
				    else:
					np.save("%s/%s/test/%s_t_%i" % (save_main, sbj, file.split("/")[-1].split(".")[0], t/1000), chunk)



		
						
	
		
