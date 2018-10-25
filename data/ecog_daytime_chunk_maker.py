import pyedflib
import os
import pdb
import glob
import numpy as np
import datetime
from ecogdeep.data.mni_space_ecog_data_transform import electrode_mapping

data_main = '/data1/ecog_project/edf/'
save_main = '/data2/users/wangnxr/dataset/standardized_clips/'
subjects = ['fcb01f7a', 'cb46fd46', 'd6532718', 'ab2431d9', 'a0f66459', 'e5bad42f', 'ec374ad0', 'c95c1e82', 'ffb52f92', 'c7980193', 'aa97abcd', 'b4ac1726', 'b45e3f7b']
mni_dir = '/home/wangnxr/Documents/mni_coords/'
subject_id_map = {'cdceeb':'fcb01f7a', 'ecb43e':'cb46fd46', '69da36':'d6532718', 'b3719b':'ab2431d9', '294e1c':'a0f66459', '0b5a2e':'e5bad52f', '0a80cf':'ec374ad0', 'c5a5e9':'c95c1e82', 'fca96e':'ffb52f92', '3f2113':'c7980193', 'acabb1':'aa97abcd', 'c19968':'b4ac1726', '13d2d8':'b45e3f7b'}

for gridlabid, sbj in subject_id_map.iteritems():
	if not os.path.exists("%s/%s" % (data_main, sbj)):
		continue
	files = sorted(glob.glob("%s/%s/*.edf" % (data_main, sbj)))
	mapping = electrode_mapping(open("%s/%s_Trodes_MNIcoords.txt" % (mni_dir, gridlabid)))
	try:
		os.makedirs("%s/%s/train/" % (save_main, sbj))
		os.makedirs("%s/%s/test/" % (save_main, sbj))
	except OSError:
		pass

	for file in files[:-2]:
		data = pyedflib.EdfReader(file)
		start_time = data.getStartdatetime()
		mean = np.zeros(64)
		std = np.zeros(64)
		for c in range(1,65):
			print "normalising channel %i" % (c-1)
			temp_data = data.readSignal(c)
			mean[c-1] = np.mean(temp_data)
			std[c-1] = np.std(temp_data)	
		for t in range(0,data.file_duration*1000-5000, 120000):
			if (start_time + datetime.timedelta(t/1000)).hour > 7:
				chunk = np.zeros(shape=(100, 5000))
				for c in range(1,65):
					if not np.isnan(mapping[c-1]):
						chunk[mapping[c-1]] = (data.readSignal(c, start=t, n=5000)-mean[c-1])/std[c-1]
				print "Saving time=%i from file %s" % (t, file)
				np.save("%s/%s/train/%s_t_%i" % (save_main, sbj, file.split("/")[-1].split(".")[0], t/1000), chunk)
	
	for file in files[-2:]:
		data = pyedflib.EdfReader(file)
		start_time = data.getStartdatetime()
                
		mean = np.zeros(64)
		std = np.zeros(64)
		for c in range(1,65):
			print "normalizing channel %i" % (c-1)
			temp_data = data.readSignal(c)
			mean[c-1] = np.mean(temp_data)
			std[c-1] = np.std(temp_data)	
		for t in range(0,data.file_duration*1000 - 5000, 120000):
			if (start_time + datetime.timedelta(t/1000)).hour > 7:
				chunk = np.zeros(shape=(100, 5000))
				for c in range(1,65):
					if not np.isnan(mapping[c-1]):
						chunk[mapping[c-1]] = (data.readSignal(c, start=t, n=5000)-mean[c-1])/std[c-1]
				print "Saving time=%i from file %s" % (t, file)
				np.save("%s/%s/test/%s_t_%is" % (save_main, sbj, file.split("/")[-1].split(".")[0], t/1000), chunk)


		
						
	
		
