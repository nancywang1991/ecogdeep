import csv
import numpy as np
import pickle
from ecogdeep.train.sbj_parameters import *
import pdb
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

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
                    timelines = [timelines[np.argsort([float(timeline[1])+float(timeline[2]) for timeline in timelines])[-1]], timelines[np.argsort([float(timeline[1])+float(timeline[2]) for timeline in timelines])[-1]]]
                    timelines = [timelines[0][0], np.mean([float(timeline[1]) for timeline in timelines]), np.mean([float(timeline[2]) for timeline in timelines])]
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
		    try:
			#pdb.set_trace()
                    	timelines = timelines[np.argmax([max(pickle.load(open("/home/wangnxr/history/" + timeline[0].split("__")[0]+ "_.p"))["val_acc"])
                                                     for timeline in timelines])]
		    except:
			timelines = timelines[np.argmax([float(timeline[1])+float(timeline[2]) for timeline in timelines])]
                    print timelines[0] 
		elif len(timelines)==0:
                    timelines = [-1,-1,-1]
                else:
                    timelines = timelines[0]
                result_dict[sbj][time] = timelines
        else:
            for time in start_times:
                result_dict[sbj][time]=[-1,-1,-1]
    return result_dict

ecog_file = "/home/wangnxr/results/ecog_lstm20_summary_results_temp.txt"
vid_file = "/home/wangnxr/results/vid_lstm_summary_results_temp.txt"
ecog_vid_file = "/home/wangnxr/results/ecog_vid_lstm_summary_results_temp.txt"
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

    fig, axes = plt.subplots(3)
    ind = np.arange(4)
    width = 0.1
    colors = 'mgbyc'
    rects_list = []
    for r, result_file in enumerate([vid_file, ecog_file, ecog_avg_file, ecog_vid_file]):
        writer.writerow([result_file])
        result_dict = process_result([line.split(":")[-1][:-1] for line in open(result_file).readlines()])
        accuracy_0 = []
        accuracy_1 = []
        average = []
        rects = []
        [axes[i].grid(zorder=0, which='both') for i in xrange(3)]

        for sbj in sbj_ids:
            accuracy_1.append([result_dict[sbj][time][1] for time in start_times])
            accuracy_0.append([result_dict[sbj][time][2] for time in start_times])
            avg = [np.mean([float(result_dict[sbj][time][1]),float(result_dict[sbj][time][2])]) for time in start_times]
            average.append(avg)
        for t in range(len(start_times)):
            axes[2-t].set_yticks(np.arange(0,101,10), minor=True)
            rects.append(axes[2-t].bar(ind + width*r, [score[t]*100 for score in average], width, color = colors[r], zorder=3))
            axes[2-t].set_ylim(40, 100)
            #axes[2-t].axhline(y=50, linestyle='--', color='k', lw = 0.5)

        rects_list.append(np.array(rects))
        writer.writerow(["Rest"] + sum(accuracy_0, []))
        writer.writerow(["Move"] + sum(accuracy_1, []))
        writer.writerow(["Average"] + sum(average, []))
    rects_list = np.array(rects_list)

    axes[1].set_ylabel("Average Scores")
    #axes[t].set_title("Scores for far back prediction")
    [axes[i].set_xticks(ind + 4*width / 2) for i in xrange(3)]
    plt.setp( axes[1].get_xticklabels(), visible=False)
    plt.setp( axes[0].get_xticklabels(), visible=False)
    axes[-1].set_xticklabels(("S1", "S2", "S3", "S4", "S5"))
    axes[2].text(3,90,"Pred_back", zorder=3)
    axes[1].text(3,90,"Pred", zorder=3)
    axes[0].text(3,90,"Decode", zorder=3)
    lgd = axes[0].legend([rects_list[i, t][0] for i in xrange(4)], ("vid", "ecog", "simp_avg", "ecog+vid"), bbox_to_anchor=(1, 0))
    #    fig.tight_layout()
    fig.savefig("/home/wangnxr/results/tabulated_graph.png", bbox_extra_artists=(lgd,), bbox_inches='tight')

with open(result_table_valbest, 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(sbj_ids)
    writer.writerow(["start time"] + start_times * 5)

    fig, axes = plt.subplots(3)
    ind = np.arange(4)
    width = 0.1
    colors = 'mgbyc'
    rects_list = []
    for r, result_file in enumerate([svm_file, vid_file, ecog_file, ecog_avg_file, ecog_vid_file]):
        writer.writerow([result_file])
	
        result_dict = process_result_valbest([line.split(":")[-1][:-1] for line in open(result_file).readlines()])
        accuracy_0 = []
        accuracy_1 = []
        average = []
        rects = []
        [axes[i].grid(zorder=0, which='both') for i in xrange(3)]

        for sbj in sbj_ids:
            accuracy_1.append([result_dict[sbj][time][1] for time in start_times])
            accuracy_0.append([result_dict[sbj][time][2] for time in start_times])
            avg = [np.mean([float(result_dict[sbj][time][1]), float(result_dict[sbj][time][2])]) for time in
                   start_times]
            average.append(avg)
        for t in range(len(start_times)):
            axes[2 - t].set_yticks(np.arange(0, 101, 10), minor=True)
            rects.append(axes[2 - t].bar(ind + width * r, [score[t] * 100 for score in average], width, color=colors[r],
                                         zorder=3))
            axes[2 - t].set_ylim(40, 100)
            # axes[2-t].axhline(y=50, linestyle='--', color='k', lw = 0.5)

        rects_list.append(np.array(rects))
        writer.writerow(["Rest"] + sum(accuracy_0, []))
        writer.writerow(["Move"] + sum(accuracy_1, []))
        writer.writerow(["Average"] + sum(average, []))
    rects_list = np.array(rects_list)

    axes[1].set_ylabel("Average Scores")
    # axes[t].set_title("Scores for far back prediction")
    [axes[i].set_xticks(ind + 5 * width / 2) for i in xrange(3)]
    plt.setp(axes[1].get_xticklabels(), visible=False)
    plt.setp(axes[0].get_xticklabels(), visible=False)
    axes[-1].set_xticklabels(("S1", "S2", "S3", "S4", "S5"))
    axes[2].text(3.1, 92, "Pred_back", zorder=3)
    axes[1].text(3.1, 92, "Pred", zorder=3)
    axes[0].text(3.1, 92, "Decode", zorder=3)
    lgd = axes[0].legend([rects_list[i, t][0] for i in xrange(5)], ("vid", "ecog", "simp_avg", "ecog+vid"),
                         bbox_to_anchor=(1, 0))
    #    fig.tight_layout()
    fig.savefig("/home/wangnxr/results/tabulated_graph_valbest.png", bbox_extra_artists=(lgd,), bbox_inches='tight')



