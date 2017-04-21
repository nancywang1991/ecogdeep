import numpy as np

file1 = "/home/wangnxr/results/vid_model_lstm_a0f_5st_pred_chkpt.txt"
file2 = "/home/wangnxr/results/ecog_model_lstm_a0f_5st_pred_chkpt.txt"

new_file = open("/home/wangnxr/results/ecog_vid_avg_model_lstm_a0f_5st_pred_chkpt.txt", "wb")

result1 = open(file1).readlines()
result2 = open(file2).readlines()
combined_results = []
length = len(result1)
for r in xrange(length):
    class1 = result1[r].split(":")[1].split("->")[0]
    class2 = result2[r].split(":")[1].split("->")[0]
    if class1==class2:
        new_file.write("%s:%s" % (result1[r].split(":")[0], class1))
        combined_results.append(int(class1))
    else:
        score1 = result1[r].split(":")[1].split("->")[1]
        score2 = result2[r].split(":")[1].split("->")[1]
        if score1> score2:
            new_file.write("%s:%s" % (result1[r].split(":")[0], class1))
            combined_results.append(int(class1))
        else:
            new_file.write("%s:%s" % (result1[r].split(":")[0], class2))
            combined_results.append(int(class2))
new_file.write("accuracy_0:%f" % len(np.where(combined_results[:length/2]==0)[0])/float(length))
new_file.write("accuracy_1:%f" % len(np.where(combined_results[length/2:]==1)[0])/float(length))