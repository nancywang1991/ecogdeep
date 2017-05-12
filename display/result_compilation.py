import numpy as np
import pdb
from ecogdeep.train.sbj_parameters import *
import glob


with open("/home/wangnxr/results/ecog_vid_avg_summary_results.txt", "wb") as summary_file:
    for s, sbj in enumerate(sbj_ids):
        for time in start_times:
	    
            model1 = glob.glob('/home/wangnxr/models/best/vid_model_lstm_%s_itr_*_t_%i_*' % (sbj, time))
            model2 = glob.glob('/home/wangnxr/models/best/ecog_model_lstm20_%s_itr_*_t_%i_*' % (sbj, time))
            new_file = open("/home/wangnxr/results/ecog_vid_avg_model_lstm_%s_t_%s.txt" % (sbj, time), "wb")
            if len(model1) == 0 or len(model2)==0:
                continue
            file1 = '/home/wangnxr/results/' + model1[0].split("/")[-1].split(".")[0] + ".txt"
            file2 = '/home/wangnxr/results/' + model2[0].split("/")[-1].split(".")[0] + ".txt"
            combined_results = []
            result1 = open(file1).readlines()
            result2 = open(file2).readlines()
	    length = len(result1)
	   
	    classes = np.array([result.split("/")[0] == result1[-1].split("/")[0] for result in result1[4:]])
            for r in xrange(4,length):
                avg_score = (float(result1[r].split(":")[-1]) + float(result2[r].split(":")[-1]))/2
                if avg_score > 0.5:
                    combined_results.append(1)
                else:
                    combined_results.append(0)
            combined_results = np.array(combined_results)
            summary_file.write("/home/wangnxr/results/ecog_vid_avg_model_lstm_%s_t_%s.txt\n" % (sbj, time))
	
	    summary_file.write("accuracy_1:%f\n" % (sum(np.array(classes)*combined_results)/float(sum(classes))))
    
            summary_file.write("accuracy_0:%f\n" % (sum((1-np.array(classes))*(1-combined_results))/float(len(classes)-sum(classes))))
            new_file.write("accuracy_0:%f\n" % (sum((1-np.array(classes))*(1-combined_results))/float(len(classes)-sum(classes))))
            new_file.write("accuracy_1:%f\n" % (sum(np.array(classes)*combined_results)/float(sum(classes))))

            result1 = open(file1).readlines()
            result2 = open(file2).readlines()
            for r in xrange(4,length):
                avg_score = (float(result1[r].split(":")[-1]) + float(result2[r].split(":")[-1]))/2
                new_file.write("%s:%f\n" % (result1[r].split(":")[0], avg_score))

