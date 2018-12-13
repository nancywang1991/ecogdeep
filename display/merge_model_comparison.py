import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import pdb
from scipy.stats.stats import pearsonr
import seaborn as sns
import pandas as pd
import scipy

def subject_coverage(sbj):
    data = np.load(glob.glob("C:\\Users\\Nancy\\Downloads\\results\\%s*.npy" % sbj)[0])
    return np.where(data[:,0]!=0)[0]


def main():
    coverage_dict = {}
    result_dict = {}
    itr = 0
    overlap_acc = []
    # Prettier names for interpolation types
    better_types = {"deep": "Deep", "interp": "Interpolate", "zero": "Zero"}
    for type in ["interp", "zero", "deep"]:
        for jitter in ["True"]:
            try:
                # Load relevant test result files
                files = np.array(sorted(glob.glob("C:\\Users\\Nancy\\Downloads\\results\\ellip\\ecog_model_mni_ellip_%s_sequence_*itr_%i_t.txt" % (type, itr))))[[0,1,2,3,4]]
                sbjs = ["a0f", "c95", "cb4", "d65"]
                self_acc = []
                other_acc = []
                # Three different time points, number of milliseconds after the start of a chunk, see AJILE paper for more info
                for t in [2700, 3300, 3900]:
                    models = []
                    cur_result = {}
                    for f, file in enumerate(files):
                        # Grab subjects that the model trained on from the filename
                        model_sbjs = file.split("_")[6:-3]
                        if len(model_sbjs) > 1:
                            # Skip the models trained on multiple subjects for now
                            continue
                        for model_sbj in model_sbjs:
                            if model_sbj not in coverage_dict:
                                #Calculate the amount of overlapping coverage for test subjects that are not train subjects
                                coverage_dict[model_sbj] = subject_coverage(model_sbj)
                        model_coverage = set(np.hstack([coverage_dict[model_sbj] for model_sbj in model_sbjs]))
                        with open(file) as results:
                            model = "_".join(os.path.basename(file).split('.')[0].split("_")[4:])
                            models.append(model)
                            #Grab results line by line
                            for line in results:
                                if line[:7] == "subject":
                                    #Grab subjects
                                    sbj = line.split(':')[1][1:4]
                                    if sbj not in coverage_dict:
                                        coverage_dict[sbj] = subject_coverage(sbj)

                                    overlap = len(set(coverage_dict[sbj]) & model_coverage)/float(len(model_coverage))
                                    time = int(line.split('time:')[1][1:5])
                                if line[:7] == "average" and time == t:
                                    #Grab result accuracy
                                    res = float(line.split(":")[1])
                                    #if sbj not in model_sbjs:
                                    overlap_acc.append((overlap, res))
                                    if sbj in model_sbjs:
                                        self_acc.append(res)
                                    elif overlap>0:
                                        other_acc.append(res)
                                    try:
                                        cur_result[sbj].append(res)
                                    except KeyError:
                                        cur_result[sbj] = [res]
                    grid = []
                    for sbj in sbjs:
                        grid.append(cur_result[sbj])

                    # Accuracy Grid plot
                    plt.matshow(np.vstack(grid), cmap=plt.cm.seismic, vmin=0.3, vmax=0.8)
                    for (i, j), z in np.ndenumerate(grid):
                        plt.text(j, i, '{:0.2f}'.format(z), ha='center', va='center', color = 'y')
                    plt.xticks(np.arange(len(models)), models)
                    plt.yticks(np.arange(len(sbjs)), sbjs)
                    plt.title("sbj vs model for time= %i" % t)
                    plt.savefig("C:\\Users\\Nancy\\Downloads\\results\\ellip\\%s_%s_%i_itr%i.png" % (type, jitter, t, itr))
                    plt.clf()

                result_dict[better_types[type]] = (np.mean(self_acc), 1.96*np.std(self_acc)/np.sqrt(len(self_acc)),
                                                           np.mean(other_acc), 1.96*np.std(other_acc)/np.sqrt(len(other_acc)), self_acc, other_acc)
            except IndexError:
                print "index error"
                pass

    # Plot overlap vs accuracy results
    overlap_acc = np.array(overlap_acc)
    r, p = pearsonr(overlap_acc[:, 0], overlap_acc[:, 1])
    plt.scatter(overlap_acc[:, 0], overlap_acc[:, 1], alpha=0.5)
    plt.xlabel("Overlapping Fraction")
    plt.ylabel("Accuracy")
    print r, p
    # plt.title("Overlapping Coverage vs Accuracy for time = %i \n r= %f, p=%f" % (t, r, p))
    plt.text(0.1,0.8,"r=0.493\np=3.604e-10", fontsize=14,
        verticalalignment='top', horizontalalignment='left')

    plt.savefig("C:\\Users\\Nancy\\Downloads\\results\\ellip\\overlap.png" )
    plt.clf()

    # Original AJILE result
    ajile = pd.DataFrame([0.668, 0.586, 0.664, 0.667, 0.657,0.646, 0.661, 0.683, 0.832, 0.568, 0.547, 0.573])
    ax = sns.barplot(data=ajile, alpha=0.7, color="yellow")
    center = ax.patches[0].get_x() + ax.patches[0].get_width()/2.
    ax.patches[0].set_x(center-0.3/2.)
    ax.patches[0].set_width(0.3)
    sns.swarmplot(data=ajile, dodge=True,color="black")
    plt.ylim([0.4, 0.75])
    plt.ylabel("Testing Day Accuracy")
    plt.xlabel("Model Type")
    ax.set_xticklabels(["testing111111"])
    plt.setp(ax.get_xticklabels(), rotation=45)
    plt.title(r"Train = Test Day Subject")
    plt.tight_layout()
    plt.savefig("C:\\Users\\Nancy\\Downloads\\results\\ellip\\ajile_acc.png")
    plt.clf()

    # Concatenate model accuracies
    bar_ind = 0
    swarm_data = pd.DataFrame(columns=["Model", "Accuracy", "Unseen"])
    for label, result in result_dict.iteritems():
        swarm_data = pd.concat([swarm_data, pd.DataFrame(np.array([[label]*len(result[4]), result[4], ["Self"] * len(result[4])]).T, columns=["Model", "Accuracy", "Unseen"])])
        swarm_data = pd.concat([swarm_data, pd.DataFrame(np.array([[label]*len(result[5]), result[5], ["Other"]*len(result[5])]).T, columns=["Model", "Accuracy", "Unseen"])])
        plt.ylabel("Testing Day Accuracy")
        plt.xlabel("Model Type")
        bar_ind += 1
    swarm_data["Model"] = swarm_data["Model"].astype("category")
    swarm_data["Unseen"] = swarm_data["Unseen"].astype("category")
    swarm_data["Accuracy"] = swarm_data["Accuracy"].astype("float")


    #Swarm and barplot for seen subject accuracy

    plot_type = "Self"
    #Stat tests
    print scipy.stats.ttest_ind(swarm_data["Accuracy"][(swarm_data["Model"]=="Deep") & (swarm_data["Unseen"]==plot_type)],
                                swarm_data["Accuracy"][(swarm_data["Model"]=="Interpolate") & (swarm_data["Unseen"]==plot_type)])
    print scipy.stats.ttest_1samp(swarm_data["Accuracy"][(swarm_data["Model"]=="Deep") & (swarm_data["Unseen"]==plot_type)], 0.5)
    print scipy.stats.ttest_1samp(swarm_data["Accuracy"][(swarm_data["Model"] == "Interpolate") & (swarm_data["Unseen"]==plot_type)], 0.5)
    print scipy.stats.ttest_1samp(swarm_data["Accuracy"][(swarm_data["Model"] == "Zero") & (swarm_data["Unseen"]==plot_type)], 0.5)

    ax = sns.barplot(x="Model", y="Accuracy", data=swarm_data[swarm_data["Unseen"]==plot_type], alpha=0.7)
    sns.swarmplot(x="Model", y="Accuracy", data=swarm_data[swarm_data["Unseen"]==plot_type], dodge=True, color="black")
    plt.ylim([0.4, 0.75])
    plt.ylabel("Testing Day Accuracy")
    plt.title(r"Train = Test Day Subject")
    plt.setp(ax.get_xticklabels(), rotation=45)
    plt.tight_layout()
    plt.savefig("C:\\Users\\Nancy\\Downloads\\results\\ellip\\overall_acc_itr%i_seen.png" % (itr))
    plt.clf()

    # Swarm and barplot for unseen subject accuracy
    plot_type = "Other"
    print scipy.stats.ttest_ind(swarm_data["Accuracy"][(swarm_data["Model"]=="Deep") & (swarm_data["Unseen"]==plot_type)],
                                swarm_data["Accuracy"][(swarm_data["Model"]=="Interpolate") & (swarm_data["Unseen"]==plot_type)])
    print scipy.stats.ttest_1samp(swarm_data["Accuracy"][(swarm_data["Model"]=="Deep") & (swarm_data["Unseen"]==plot_type)], 0.5)
    print scipy.stats.ttest_1samp(swarm_data["Accuracy"][(swarm_data["Model"] == "Interpolate") & (swarm_data["Unseen"]==plot_type)], 0.5)
    print scipy.stats.ttest_1samp(swarm_data["Accuracy"][(swarm_data["Model"] == "Zero") & (swarm_data["Unseen"]==plot_type)], 0.5)

    ax = sns.barplot(x="Model", y="Accuracy", data=swarm_data[swarm_data["Unseen"] == plot_type], alpha=0.7)
    sns.swarmplot(x="Model", y="Accuracy", data=swarm_data[swarm_data["Unseen"] == plot_type], dodge=True,
                  color="black")
    plt.ylim([0.4, 0.75])
    plt.ylabel("Testing Day Accuracy")
    plt.title(r"Train $\neq$ Test Day Subject")
    plt.setp(ax.get_xticklabels(), rotation=45)
    plt.tight_layout()
    plt.savefig("C:\\Users\\Nancy\\Downloads\\results\\ellip\\overall_acc_itr%i_seen.png" % (itr))
    plt.clf()

if __name__ == "__main__":
    main()