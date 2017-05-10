from sklearn.manifold import TSNE
import numpy as np
import glob
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import argparse
from ecogdeep.train.sbj_parameters import *
import pdb
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import cv2
from PIL import Image

def load_img_seq(path, target_mode=None, resize_size=None, num_frames=12, keep_frames=None):
    #print(path)
    img_orig = Image.open(path)
    width, height = img_orig.size
    img = img_orig.crop(((num_frames-2)*width/num_frames,0,(num_frames-1)*width/num_frames, height))
    if resize_size:
        img = img.resize((resize_size[1], resize_size[0]))

    return img

def imscatter(x, y, image, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    try:
        image = plt.imread(image)
    except TypeError:
        # Likely already an array...
        pass

    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0, im0 in zip(x, y, image):
        im = load_img_seq(im0, resize_size=(50,50))
        im = OffsetImage(im, zoom=zoom)
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists

def main(data_source):
    for time in start_times:
        files = sorted(glob.glob(data_source + "/*/*_%i.npy" % time))
        image_files = ["/".join(file.split("/")[:-2]) + "/train/" + "_".join(file.split("/")[-1].split("_")[:-1]) + ".png"]

        X = np.array([np.ndarray.flatten(np.load(file)) for file in files])
        y = np.array([file.split("/")[-2]=="mv_0" for file in files])
        model = TSNE(n_components=2, random_state=0)
        transforms = model.fit_transform(X)
        plt.scatter(transforms[np.where(y==0)[0], 0], transforms[np.where(y==0)[0],1], c = "b", s = 0.5, label="Move")
        plt.scatter(transforms[np.where(y==1)[0], 0], transforms[np.where(y==1)[0],1], c = 'r', s = 0.5, label="No move")
        imscatter(transforms[:,0], transforms[:,1], image_files)
        plt.savefig(data_source + "/%i_tsne_graph.png" % time)
        plt.clf()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_source', required=True, help="embedding transformed directory")
    args = parser.parse_args()
    main(args.data_source)
