from sklearn.manifold import TSNE
import numpy as np
import glob
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import argparse

def main(data_source):
    files = sorted(glob.glob(data_source + "/*/*"))

    X = np.array([np.flatten(np.load(file)) for file in files])
    y = np.array([file.split("/")[0]=="mv_0" for file in files])
    model = TSNE(n_components=2, random_state=0)
    transforms = model.fit_transform(X)

    plt.scatter(transforms[np.where(y==0)[0], 0], transforms[np.where(y==0)[0],1], label="Move")
    plt.scatter(transforms[np.where(y==1)[0], 0], transforms[np.where(y==1)[0],1], label="No move")
    plt.savefig(data_source + "tsne_graph.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_source', required=True, help="embedding transformed directory")
    args = parser.parse_args()
    main(args.data_source)
