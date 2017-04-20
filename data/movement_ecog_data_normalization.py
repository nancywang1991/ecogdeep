import argparse
import glob
import cPickle as pickle
import matplotlib
matplotlib.use("agg")
import numpy as np
import os
import mne.io
import pdb

def fft_crop(data, freqs):
    fft_data = (np.abs(np.fft.fft(data))**2)
    return np.array([np.sum(fft_data[freq[0]:freq[1]]) for freq in freqs])
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--edf_dir', required=True, help="edf directory")
    parser.add_argument('-s', '--save_dir', required=True, help="Save directory")
    args = parser.parse_args()
    files = glob.glob(args.edf_dir + "/*.edf")
    sbj_id, day = os.path.split(files[0])[-1].split(".")[0].split("_")
    edfs = mne.io.read_raw_edf(files[0])
    n_channels=edfs.ch_names.index('EOGL')-2
    for f, file in enumerate(files):
        norm_factors = np.zeros(shape=(n_channels, 2))
        edf = mne.io.read_raw_edf(file)
        window_mean = []
        window_std = []
        window_fft = []
    	for c in xrange(n_channels):
        	print "Working on channel %i" % c
            	temp_data, _ = edf[c+1, :]
                window_mean.append(np.array([np.mean((temp_data[0,t:t+1000])) for t in range(0,temp_data.shape[1], 1000)])) 
                window_std.append(np.array([np.std((temp_data[0,t:t+1000])) for t in range(0,temp_data.shape[1], 1000)]))
                window_fft.append(np.array([fft_crop(temp_data[0,t:t+1000], [[2,5],[30,40],[59,61],[110,120],[190,200]]) for t in range(0,temp_data.shape[1], 1000)]))
            	norm_factors[c, 0] = np.mean(temp_data)
            	norm_factors[c, 1] = np.std(temp_data)
                 
    	pickle.dump(norm_factors, open("%s/%s_norm_factors_%s.p" % (args.save_dir, sbj_id, file.split("_")[-1].split(".")[0]), "wb"))
        pickle.dump(np.vstack(window_mean), open("%s/%s_means_%s.p" % (args.save_dir, sbj_id, file.split("_")[-1].split(".")[0]), "wb"))
        pickle.dump(np.vstack(window_std), open("%s/%s_stds_%s.p" % (args.save_dir, sbj_id, file.split("_")[-1].split(".")[0]), "wb"))
	pickle.dump(np.array(window_fft), open("%s/%s_ffts_%s.p" % (args.save_dir, sbj_id, file.split("_")[-1].split(".")[0]), "wb"))
