import os
import numpy as np
import glob
import pdb
import scipy.interpolate

subjects = ['d65', 'a0f', 'cb4', 'c95']
main_data_dir = "/home/wangnxr/"

for subject in subjects:
    print "Working on subject %s" % subject
    for file in glob.glob("%s/standardized_clips_ellip/%s*/val/*.npy" % (main_data_dir, subject)):
        print file
        orig = np.load(file)
        valid = np.where(orig[:,0]!=0)[0]
	orig[valid[5:10]]=0
	valid = np.where(orig[:,0]!=0)[0]
        gridinds = np.array([valid/10, valid%10])
        gridx, gridy = np.mgrid[0:10,0:10]
	try:
            for t in xrange(orig.shape[-1]):
                 orig[:,t] = np.ndarray.flatten(scipy.interpolate.griddata(gridinds.T, orig[valid,t], (gridx, gridy), fill_value=0, method='cubic'))
	except:
	    pass
	valid = np.where(orig[:,0]!=0)[0]
        gridinds = np.array([valid/10, valid%10])
        for t in xrange(orig.shape[-1]):
            orig[:,t] = np.ndarray.flatten(scipy.interpolate.griddata(gridinds.T, orig[valid,t], (gridx, gridy), method='nearest'))
        try:
            np.save("%s/ecog_mni_ellip_interp_analy_%s/%s" % (main_data_dir, subject, "/".join(file.split("/")[-3:])), orig)
        except IOError:
            os.makedirs("%s/ecog_mni_ellip_interp_analy_%s/%s" % (main_data_dir, subject, "/".join(file.split("/")[-3:-1])))
	    np.save("%s/ecog_mni_ellip_interp_analy_%s/%s" % (main_data_dir, subject, "/".join(file.split("/")[-3:])), orig)

