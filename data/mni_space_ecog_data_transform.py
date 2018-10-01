import numpy as np
import glob
import pdb

xedges = np.array([-60., -50., -40., -30., -20., -10.,   0.,  10.,  20.,  30.,  40.])
yedges = np.array([-40., -32., -24., -16.,  -8.,   0.,   8.,  16.,  24.,  32.,  40.])

subjects = ['69da36', '294e1c']
subject_id_map = {'69da36':'d65', '294e1c':'a0f'}
mni_dir = '/data2/users/wangnxr/mni_coords/'
main_data_dir = "/data2/users/wangnxr/dataset/"

def find_bin(n, edges):
    if n < edges[0] or n > edges[-1]:
        return np.nan
    else:
        return np.where(n>edges)[0][0]

for subject in subjects:
    mni_file = open("%s/%s_Trodes_MNIcoords.txt" % (dir, subject))
    electrodes = np.vstack([np.array([float(x) for x in channel.split(",")]) for channel in file])
    electrode_mapping = {}
    for c in range(64):
        electrode_mapping[c] = find_bin(electrodes[c, 2],xedges)*10 + find_bin(electrodes[c,1], yedges)

    for file in glob.glob("%s/ecog_vid_combined_%s_day*/*/*/*.npy" % (main_data_dir, subject_id_map[subject], type)):
        orig = np.load(file)
        new = np.array(shape=(100, orig.shape[1]))
        for old_c, new_c in electrode_mapping.iteritems():
            new[new_c] = orig[old_c]
        file_parts = file.split("/")
        pdb.set_trace()
        np.save(new, "%s/ecog_mni_%s/%s" % (main_data_dir, subject_id_map[subject], "/".join(file_parts[-3:])))
