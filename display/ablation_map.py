import scipy.io
import pdb

"""Converting ablation results into format for matlab brain plots.


Example:
        $ python ablation_map.py

"""

result_file = "C:/Users/Nancy/Documents/data/ecog_lstm_mni_summary_results.txt"
# Accuracy without any ablation in the order of the result_file
# multimodal
#acc_orig = [0.67, 0.59, 0.66, 0.67, 0.66, 0.65, 0.66, 0.68,	0.83, 0.57, 0.55, 0.57]
results = open(result_file).readlines()
file_start_inds = [i for i in xrange(len(results)) if results[i][:4]=="ecog"]

for i, ind in enumerate(file_start_inds):
    if i < len(file_start_inds)-1:
        ind2 = file_start_inds[i+1]
    else:
        ind2 = len(results)
    channel_scores = [(float(results[l].split(":")[-1])) for l in range(ind+1, ind2)]
    scipy.io.savemat("/".join(result_file.split("/")[:-1]) + "/" + results[ind][:-1] + "_map.mat", {"score":channel_scores})



