import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import pdb

for type in ["deep", "interp", "zero"]:
    for jitter in ["True", "False"]:
        try:
            files = np.array(sorted(glob.glob("C:\\Users\\Nancy\\Downloads\\results\\ecog_model_mni_%s_jitter_%s_*itr_2_t.txt" % (type, jitter))))[[1,4,2,3,0]]

            sbjs = ["a0f", "d65", "c95", "cb4"]
            for t in [2700,3300,3900]:
                models = []
                cur_result = {}
                for f, file in enumerate(files):
                    with open(file) as results:
                        model = "_".join(os.path.basename(file).split('.')[0].split("_")[6:])
                        models.append(model)
                        for line in results:
                            if line[:7] == "subject":
                                sbj = line.split(':')[1][1:4]
                                time = int(line.split('time:')[1][1:5])
                            if line[:7] == "average" and time == t:
                                res = float(line.split(":")[1])
                                try:
                                    cur_result[sbj].append(res)
                                except KeyError:
                                    cur_result[sbj] = [res]
                grid = []
                for sbj in sbjs:
                    grid.append(cur_result[sbj])

                plt.matshow(np.vstack(grid), cmap=plt.cm.seismic, vmin=0.3, vmax=0.8)
                for (i, j), z in np.ndenumerate(grid):
                    plt.text(j, i, '{:0.2f}'.format(z), ha='center', va='center', color = 'y')
                plt.xticks(np.arange(len(models)), models)
                plt.yticks(np.arange(len(sbjs)), sbjs)
                plt.title("sbj vs model for time= %i" % t)
                #plt.show()
                plt.savefig("C:\\Users\\Nancy\\Downloads\\results\\%s_%s_%i_itr2.png" % (type, jitter, t))
                plt.clf()
                plt.cla()
                plt.close()
        except:
            pass