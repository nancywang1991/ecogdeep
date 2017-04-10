import argparse
import glob
import cPickle as pickle
import matplotlib
matplotlib.use("agg")
import numpy as np
import os
import mne.io

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--edf_dir', required=True, help="edf directory")
    parser.add_argument('-s', '--save_dir', required=True, help="Save directory")
    args = parser.parse_args()
    files = glob.glob(args.edf_dir + "/*.edf")
    sbj_id, day = os.path.split(files[0])[-1].split(".")[0].split("_")
    edfs = [mne.io.read_raw_edf(file) for file in files]
    n_channels=edfs[0].ch_names.index('EOGL')-1
    norm_factors = np.zeros(shape=(n_channels, 2))
    for c in xrange(edfs[0].ch_names.index('EOGL')-1):
        print "Working on channel %i" % c
        edf_data = []
        for edf in edfs:
            temp_data, _ = edf[c+1, :]
            edf_data.append(temp_data)
            norm_factors[c, 0] = np.mean(np.hstack(edf_data))
            norm_factors[c, 1] = np.std(np.hstack(edf_data))
    pickle.dump(norm_factors, open("%s/%s_norm_factors.p" % (args.save_dir, sbj_id), "wb"))
