import glob
import scipy.stats
import scipy.fftpack
import numpy as np
import pdb
import matplotlib.pyplot as plt
import scipy

sbj_ids = ["a0f", "cb4", "c95", "d65"]
all_corr_means = []
all_band_corr_means = []
all_recon_means = []

for sbj_id in sbj_ids:
    ground_truth = "C:/Users/Nancy/Downloads/results/ecog_mni_ellip_%s/%s*" % (sbj_id, sbj_id)
    #mComparing 3 models
    model_results = ["C:/Users/Nancy/Downloads/results/ecog_mni_ellip_deep_impute_analyallsbj_%s/%s*" % (sbj_id, sbj_id),
                     "C:/Users/Nancy/Downloads/results/ecog_mni_ellip_deep_impute_analyajilesbj_%s/%s*" % (sbj_id, sbj_id),
                     "C:/Users/Nancy/Downloads/results/ecog_mni_ellip_interp_analy_%s/%s*" % (sbj_id, sbj_id)]

    all_corr={0:[],1:[],2:[],3:[],4:[]}
    all_band_corr={0:[],1:[],2:[],3:[],4:[]}
    all_recon={0:[],1:[],2:[],3:[],4:[]}
    plot=False
    file_len_in_sec = 5
    # Frequencies to look at
    f_lo, f_hi = 5, 150

    for file in glob.glob("%s/*/*.npy" % ground_truth):
        try:
            # Leave out the first 80 timesteps since we can't impute this part
            ground_truth_data = np.load(file)[:,80:]
            ground_truth_band = np.abs(scipy.fftpack.fft(ground_truth_data, axis=1)**2)[:, f_lo*file_len_in_sec:f_hi*file_len_in_sec]
            # Find withheld electrodes
            channels = np.where(ground_truth_data[:,0] != 0)[0][5:10]
            models_data = [np.load(glob.glob(dir + "/val/" + file.split('\\')[-1])[0])[:,80:] for dir in model_results]
            bandpowers = [np.abs(scipy.fftpack.fft(data, axis=1)**2)[:,f_lo*file_len_in_sec:f_hi*file_len_in_sec]
                          for data in models_data]

            for chan in xrange(len(channels)):
                # Calculate Correlations
                corr = [scipy.stats.pearsonr(ground_truth_data[channels[chan]], model_data[channels[chan]]) for m, model_data in enumerate(models_data)]
                band_corr = [scipy.stats.pearsonr(ground_truth_band[channels[chan]], bandpowers[m][channels[chan]]) for m, model_data in enumerate(models_data)]
                recon = [np.mean((model_data[channels[chan]]-ground_truth_data[channels[chan]])**2) for model_data in models_data]
                all_corr[chan].append(np.hstack(corr))
                all_band_corr[chan].append(np.hstack(band_corr))
                all_recon[chan].append(np.hstack(recon))
                if plot:
                    print corr

            if plot:
                # Plot ground truth and reconstructions
                fig, axes = plt.subplots(4,1)
                axes[0].plot(np.array([ground_truth_data[channel] + c for c, channel in enumerate(channels)]).T)
                axes[0].set_yticks([], [])
                for m in range(3):
                    axes[m+1].plot(np.array([models_data[m][channel] + c for c, channel in enumerate(channels)]).T)
                    axes[m+1].set_yticks([], [])
                plt.xlabel("Time(ms)")
                plt.show()

                #Plot ground truth band powers and reconstruction band powers
                fig, axes = plt.subplots(4,1)
                axes[0].plot(np.array([ground_truth_band[channel] + c for c, channel in enumerate(channels)]).T)
                for m in range(3):
                    axes[m+1].plot(np.array([bandpowers[m][channel] + c for c, channel in enumerate(channels)]).T)
                plt.show()

        except IndexError:
            pass

    for chan in xrange(len(channels)):
        all_corr_means.append(np.mean(np.vstack(all_corr[chan]), axis=0))
        all_recon_means.append(np.mean(np.vstack(all_recon[chan]), axis=0))
        all_band_corr_means.append(np.mean(np.vstack(all_band_corr[chan]), axis=0))

# Calculate mean correlations
print np.mean(np.vstack(all_corr_means), axis=0)[::2]
print np.mean(np.vstack(all_band_corr_means), axis=0)[::2]
print np.mean(np.vstack(all_recon_means), axis=0)