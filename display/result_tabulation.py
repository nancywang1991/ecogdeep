import csv
import numpy as np
import pickle
from ecogdeep.train.sbj_parameters import *
import pdb

def detect_ind(phrase, lines):
    inds = []
    for l, line in enumerate(lines):
        if phrase in line:
            inds.append(l)
    return inds

def process_result(lines):
    result_dict = {}
    for sbj in sbj_ids:
        result_dict[sbj] = {}
        sbjlines = [lines[ind:ind+3] for ind in detect_ind(sbj, lines)]
        sbjlines = sum(sbjlines, [])
        if len(sbjlines) > 0:
            for time in start_times:
                timelines = [sbjlines[ind:ind+3] for ind in detect_ind("t_" + str(time), sbjlines)]
		if len(timelines) > 1:
                    timelines = timelines[np.argmax([float(timeline[1])+float(timeline[2]) for timeline in timelines])]
		elif len(timelines)==0:
		    timelines = [-1,-1,-1]
		else:
		    timelines = timelines[0]
                result_dict[sbj][time] = timelines
	else:
	    for time in start_times:
		result_dict[sbj][time]=[-1,-1,-1]
    return result_dict

def process_result_valbest(lines):
    result_dict = {}
    for sbj in sbj_ids:
        result_dict[sbj] = {}
        sbjlines = [lines[ind:ind+3] for ind in detect_ind(sbj, lines)]
        sbjlines = sum(sbjlines, [])
        if len(sbjlines) > 0:
            for time in start_times:
                timelines = [sbjlines[ind:ind+3] for ind in detect_ind("t_" + str(time), sbjlines)]
                if len(timelines) > 1:
                    timelines = timelines[np.argmax([max(pickle.load("_".append(timeline[0].split("_")[:8])+ "_")["val_acc"])
                                           for timeline in timelines])]
		else:
                    timelines = timelines[0]
                result_dict[sbj][time] = timelines
    return result_dict



ecog_file = "/home/wangnxr/results/ecog_lstm20_summary_results.txt"
vid_file = "/home/wangnxr/results/vid_lstm_summary_results.txt"
ecog_vid_file = "/home/wangnxr/results/ecog_vid_lstm_summary_results.txt"
svm_file = "/home/wangnxr/results/ecog_svm_summary_results.txt"
ecog_conv_file = "/home/wangnxr/results/ecog_conv_summary_results.txt"
ecog_avg_file = "/home/wangnxr/results/ecog_vid_avg_summary_results.txt"

result_table = "/home/wangnxr/results/summary_table.csv"
result_table_valbest = "/home/wangnxr/results/summary_table_valbest.csv"

with open(result_table, 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(sbj_ids)
    writer.writerow(["start time"] + start_times*5)

    for result_file in [ecog_file, vid_file, ecog_vid_file, svm_file, ecog_avg_file]:
        writer.writerow([result_file])
        result_dict = process_result([line.split(":")[-1][:-1] for line in open(result_file).readlines()])
        accuracy_0 = []
        accuracy_1 = []
        average = []
        for sbj in sbj_ids:
            accuracy_1.append([result_dict[sbj][time][1] for time in start_times])
            accuracy_0.append([result_dict[sbj][time][2] for time in start_times])
	    #pdb.set_trace()
            average.append([np.mean([float(result_dict[sbj][time][1]),float(result_dict[sbj][time][2])]) for time in start_times])
        writer.writerow(["accuracy_0"] + sum(accuracy_0, []))
        writer.writerow(["accuracy_1"] + sum(accuracy_1, []))
        writer.writerow(["average"] + sum(average, []))

with open(result_table_valbest, 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(sbj_ids)
    writer.writerow(["start time"] + start_times*5)

    for result_file in [ecog_file, vid_file, ecog_vid_file, svm_file, ecog_avg_file]:
        writer.writerow([result_file])
        result_dict = process_result_valbest(open(result_file).readlines())
        accuracy_0 = []
        accuracy_1 = []
        average = []
        for sbj in sbj_ids:
            accuracy_1.append([result_dict[sbj][time][1] for time in start_times])
            accuracy_0.append([result_dict[sbj][time][2] for time in start_times])
            average.append([np.mean([float(result_dict[sbj][time][1]),float(result_dict[sbj][time][2])]) for time in start_times])
        writer.writerow(["accuracy_0"] + sum(accuracy_0, []))
        writer.writerow(["accuracy_1"] + sum(accuracy_1, []))
        writer.writerow(["average"] + sum(average, []))

