import matplotlib
matplotlib.use("agg")
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
import pdb
import os
import glob


filenames=glob.glob("/home/wangnxr/history_ecog_*.p")
save_folder="/home/wangnxr/"

for filename in filenames:
        plt.clf()
	history = pickle.load(open(filename))

	#accuracy plot
	plt.plot(history["acc"], label="training")
	plt.plot(history["val_acc"], label="validation")
	plt.legend()
	plt.title("Accuracy over training epochs")
	plt.ylim([0,1])
	plt.xlabel("epochs")
	plt.savefig(save_folder+os.path.basename(filename).split(".")[0]+"acc.png")
	plt.clf()
	#loss plot
	plt.plot(history["loss"], label="training")
	plt.plot(history["val_loss"], label="validation")
	plt.legend()
	plt.title("Loss over training epochs")
	plt.ylim([0,5])
	plt.xlabel("epochs")
	plt.savefig(save_folder+os.path.basename(filename).split(".")[0]+"loss.png")
	#pdb.set_trace()
