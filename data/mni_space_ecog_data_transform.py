import os
import numpy as np
import glob
import pdb


def find_bin(n, edges):
    if n < edges[0] or n > edges[-1]:
        return np.nan
    else:
        return np.where(n > edges)[0][-1]

def electrode_mapping_func(mni_file, xedges=None, yedges=None):
    if xedges is None:
	xedges = np.array([-30., -22., -14., -6.,  2.,   10.,   18.,  26.,  34.,  42.,  50.])
    if yedges is None:
        yedges = np.array([-40., -32., -24., -16., -8., 0., 8., 16., 24., 32., 40.])
    electrodes = np.vstack([np.array([float(x) for x in channel.split(",")]) for channel in mni_file])
    mapping = {}
    for c in range(64):
        mapping[c] = find_bin(electrodes[c, 2],yedges)*10 + find_bin(electrodes[c,1], xedges)
    for file in glob.glob("%s/ecog_vid_combined_%s_day*/*/*/*.npy" % (main_data_dir, subject_id_map[subject])):
        print file
	orig = np.load(file)
        new = np.zeros(shape=(100, orig.shape[1]))
        for old_c, new_c in electrode_mapping.iteritems():
	    if not np.isnan(new_c):
            	new[new_c] = orig[old_c]
        file_parts = file.split("/")
	try:
            np.save("%s/ecog_mni_%s/%s" % (main_data_dir, subject_id_map[subject], "/".join(file_parts[-3:])), new)
	except IOError:
	    os.makedirs("%s/ecog_mni_%s/%s" % (main_data_dir, subject_id_map[subject], "/".join(file_parts[-3:-1])))
        mapping[c] = find_bin(electrodes[c, 2], xedges) * len(yedges) + find_bin(electrodes[c, 1], yedges)
    return mapping

def main():
    subjects = ['69da36', 'ecb43e', 'c5a5e9']
    subject_id_map = {'69da36': 'd65', '294e1c': 'a0f', 'c5a5e9': 'c95', 'ecb43e': 'cb4'}
    mni_dir = '/data2/users/wangnxr/mni_coords/'
    main_data_dir = "/data2/users/wangnxr/dataset/"
    for subject in subjects:
        print "Working on subject %s" % subject
        mni_file = open("%s/%s_Trodes_MNIcoords.txt" % (mni_dir, subject))
        mapping=electrode_mapping_func(mni_file)
	pdb.set_trace()
        for file in glob.glob("%s/ecog_vid_combined_%s_day*/*/*/*.npy" % (main_data_dir, subject_id_map[subject])):
            print file
            orig = np.load(file)
            new = np.zeros(shape=(100, orig.shape[1]))
            for old_c, new_c in mapping.iteritems():
                if not np.isnan(new_c):
                    new[new_c] = orig[old_c]
            file_parts = file.split("/")
            try:
                np.save("%s/ecog_mni_%s/%s" % (main_data_dir, subject_id_map[subject], "/".join(file_parts[-3:])), new)
            except IOError:
                os.makedirs("%s/ecog_mni_%s/%s" % (main_data_dir, subject_id_map[subject], "/".join(file_parts[-3:-1])))

if __name__=='__main__':
    main()