import os
import numpy as np
import glob
import pdb
import scipy.interpolate

def main(sbjs, main_data_dir, mask_withheld, savename):
    for sbj in sbjs:
        #Params
        files = glob.glob("%s/standardized_clips_ellip/%s*/val/*.npy" % (main_data_dir, sbj))
        print "Working on subject %s" % sbj
        for file in files:
            print file
            orig = np.load(file)
            valid = np.where(orig[:,0]!=0)[0]
            if mask_withheld:
                orig[valid[5:10]]=0
                valid = np.where(orig[:,0]!=0)[0]
            gridinds = np.array([valid/10, valid%10])
            gridx, gridy = np.mgrid[0:10,0:10]
            try:
                # Try cubic interpolation first
                for t in xrange(orig.shape[-1]):
                    orig[:,t] = np.ndarray.flatten(scipy.interpolate.griddata(gridinds.T, orig[valid,t], (gridx, gridy), fill_value=0, method='cubic'))
            except:
                pass
            valid = np.where(orig[:,0]!=0)[0]
            gridinds = np.array([valid/10, valid%10])
            for t in xrange(orig.shape[-1]):
                # Perform nearest neighbor interpolation for places where cubic interpolation cannot be done
                orig[:,t] = np.ndarray.flatten(scipy.interpolate.griddata(gridinds.T, orig[valid,t], (gridx, gridy), method='nearest'))
            try:
                np.save("%s/%s_%s/%s" % (main_data_dir, savename, sbj, "/".join(file.split("/")[-3:])), orig)
            except IOError:
                os.makedirs("%s/%s_%s/%s" % (main_data_dir, savename, sbj, "/".join(file.split("/")[-3:-1])))
                np.save("%s/%s_%s/%s" % (main_data_dir, savename, sbj, "/".join(file.split("/")[-3:])), orig)

if __name__ == "__main__":
    sbjs_to_do = ["a0f", "cb4", "c95", "d65"]
    savename = "ecog_mni_ellip_interp_analy"
    main_data_dir = "/home/wangnxr/"
    # Enable if testing withheld electrode reconstruction. Otherwise, when producing imputed files, turn to False
    mask_withheld = True
    main(sbjs_to_do, main_data_dir, mask_withheld, savename)
