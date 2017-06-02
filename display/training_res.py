import matplotlib
matplotlib.use("agg")
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
import pdb
import os
import glob
import scipy.io

"""Plot training and validation accuracies over training epochs.


Example:
        $ python training_res.py

"""

filenames=glob.glob("/home/wangnxr/BCIcomp_results/*history*.p")
save_folder="/home/wangnxr/BCIcomp_results/"
# Max epoch to plot
limit = 100
for filename in filenames:
    plt.clf()
    history = pickle.load(open(filename))

    #accuracy plot
    plt.plot(history["acc"][:limit], label="training")
    plt.plot(history["val_acc"][:limit], label="validation")
    plt.legend()
    plt.title("Accuracy over training epochs")
    plt.ylim([0,1])
    plt.xlabel("epochs")
    plt.savefig(save_folder+os.path.basename(filename).split(".")[0]+"acc.png")
    plt.clf()
    #loss plot
    plt.plot(history["loss"][:limit], label="training")
    plt.plot(history["val_loss"][:limit], label="validation")
    plt.legend()
    plt.title("Loss over training epochs")
    plt.ylim([0,5])
    plt.xlabel("epochs")
    plt.savefig(save_folder+os.path.basename(filename).split(".")[0]+"loss.png")
    #pdb.set_trace()
    scipy.io.savemat(save_folder+os.path.basename(filename).split(".")[0]+".mat", mdict={'data': history})
