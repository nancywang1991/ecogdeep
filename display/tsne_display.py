from sklearn.manifold import TSNE
import numpy as np
import glob
import matplotlib
matplotlib.use("agg")
import matplotlib.cm
import matplotlib.pyplot as plt
import argparse
from ecogdeep.train.sbj_parameters import *
import pdb
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import cv2
from PIL import Image, ImageOps


def load_img_seq(path, target_mode=None, resize_size=None, color=None,num_frames=12, keep_frames=None):
    #print(path)
    img_orig = Image.open(path)
    width, height = img_orig.size
    img = img_orig.crop(((num_frames-2)*width/num_frames,0,(num_frames-1)*width/num_frames, height))
    if resize_size:
        img = img.resize((resize_size[1], resize_size[0]))
    if color:
        img = ImageOps.expand(img,border=5,fill=color)

    return img

def imscatter(x, y, image, ax=None, zoom=1, frameon=False, color=None, days=None):
    if ax is None:
        ax = plt.gca()
    try:
        image = plt.imread(image)
    except TypeError:
        # Likely already an array...
        pass

    x, y = np.atleast_1d(x, y)
    artists = []
    cmap = matplotlib.cm.get_cmap('Spectral')
    for x0, y0, im0 in zip(x, y, image):
        if days:
            color = cmap(((int(im0.split("_")[1])-days[0])*700+int(im0.split("_")[2]))/((days[1]-days[0])*700.0))
        im = load_img_seq(im0, resize_size=(60,60), color=color)
        im = OffsetImage(im, zoom=2)
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=frameon)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists

def main(data_source):
    for time in start_times[:1]:
        plt.figure(figsize=(150,150))
        files = sorted(glob.glob(data_source + "/*/*_%i.npy" % time))
        image_files = ["/".join(file.split("/")[:-3]) + "/train/" + file.split("/")[-2] + "/" + "_".join(file.split("/")[-1].split("_")[:-1]) + ".png" for file in files]

        X = np.array([np.ndarray.flatten(np.load(file)) for file in files])
        y = np.array([file.split("/")[-2]=="mv_0" for file in files])
        model = TSNE(n_components=2, random_state=0)
        transforms = model.fit_transform(X)
        #plt.scatter(transforms[np.where(y==0)[0], 0], transforms[np.where(y==0)[0],1], c = "b", s = 0.5, label="Move")
        #plt.scatter(transforms[np.where(y==1)[0], 0], transforms[np.where(y==1)[0],1], c = 'r', s = 0.5, label="No move")
        days = (int(files[0].split("_")[1]), int(files[-1].split("_")[1]))
        imscatter(transforms[np.where(y==0)[0],0], transforms[np.where(y==0)[0],1], np.array(image_files)[np.where(y==0)[0]], days=days, color="blue")
        imscatter(transforms[np.where(y==1)[0],0], transforms[np.where(y==1)[0],1], np.array(image_files)[np.where(y==1)[0]], days=days, color="red")
        #plt.scatter(transforms[np.where(y==0)[0], 0], transforms[np.where(y==0)[0],1], c = "b", s = 50, label="Move")
        #plt.scatter(transforms[np.where(y==1)[0], 0], transforms[np.where(y==1)[0],1], c = 'r', s = 50, label="No move")

        plt.savefig(data_source + "/%i_tsne_graph.pdf" % time)
        plt.clf()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_source', required=True, help="embedding transformed directory")
    args = parser.parse_args()
    main(args.data_source)
