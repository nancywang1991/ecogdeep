import scipy.io
import pdb

result_file = "C:/Users/Nancy/OneDrive/Documents/Documents/brunton_lab/NIPS2017/ecog_vid_lstm_summary_results_ablate_main.txt"
#acc_orig = [0.69,0.73,0.77,0.51,0.60,0.65]
acc_orig = [0.64, 0.64,	0.65, 0.87,	0.88, 0.85, 0.69, 0.71,	0.71]

results = open(result_file).readlines()
file_start_inds = [i for i in xrange(len(results)) if results[i][:4]=="ecog"]

for i, ind in enumerate(file_start_inds):
    if i < len(file_start_inds)-1:
        ind2 = file_start_inds[i+1]
    else:
        ind2 = len(results)
    channel_scores = [(float(results[l].split(":")[-1])-acc_orig[i]) for l in range(ind+1, ind2)]
    scipy.io.savemat("/".join(result_file.split("/")[:-1]) + "/" + results[ind][:-1] + "_map.mat", {"score":channel_scores})



