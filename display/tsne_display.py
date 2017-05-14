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

#days = [5,6,7,8,11]
def load_img_seq(path, target_mode=None, resize_size=None, color=None,num_frames=12, keep_frames=None):
    #print(path)
    img_orig = Image.open(path)
    width, height = img_orig.size
    img = img_orig.crop(((num_frames-2)*width/num_frames,0,(num_frames-1)*width/num_frames, height))
    if resize_size:
        img = img.resize((resize_size[1], resize_size[0]))
    if color:
        #img = ImageOps.expand(img,border=2,fill=tuple([int(c*256) for c in color]))
	img = ImageOps.expand(img,border=2,fill=color)

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
    cmap = matplotlib.cm.get_cmap('nipy_spectral')
    for x0, y0, im0 in zip(x, y, image):
        if days:
            color = cmap((days.index(int(im0.split("/")[-1].split("_")[1]))*700+int(im0.split("/")[-1].split("_")[2]))/((len(days))*700.0))
        im = load_img_seq(im0, resize_size=(1,1), color=color)
        im = OffsetImage(im, zoom=2)
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=frameon)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists

def main(data_source):
    for time in start_times[1:]:
        plt.figure(figsize=(20,20))
        files = (glob.glob(data_source + "/*/*_%i.npy" % time))
	np.random.shuffle(files)
        image_files = ["/".join(file.split("/")[:-3]) + "/train/" + file.split("/")[-2] + "/" + "_".join(file.split("/")[-1].split("_")[:-1]) + ".png" for file in files]

        X = np.array([np.ndarray.flatten(np.load(file)) for file in files])
        y = np.array([file.split("/")[-2]=="mv_0" for file in files])
        model = TSNE(n_components=2, random_state=0)
        transforms = model.fit_transform(X)
        #plt.scatter(transforms[np.where(y==0)[0], 0], transforms[np.where(y==0)[0],1], c = "b", s = 0.5, label="Move")
        #plt.scatter(transforms[np.where(y==1)[0], 0], transforms[np.where(y==1)[0],1], c = 'r', s = 0.5, label="No move")
        #days = (int(files[0].split("/")[-1].split("_")[1]), int(files[-1].split("/")[-1].split("_")[1]))
        imscatter(transforms[np.where(y==0)[0],0], transforms[np.where(y==0)[0],1], np.array(image_files)[np.where(y==0)[0]], color="blue")
        imscatter(transforms[np.where(y==1)[0],0], transforms[np.where(y==1)[0],1], np.array(image_files)[np.where(y==1)[0]], color="red")
        #plt.scatter(transforms[np.where(y==0)[0], 0], transforms[np.where(y==0)[0],1], c = "b", s = 50, label="Move")
        #plt.scatter(transforms[np.where(y==1)[0], 0], transforms[np.where(y==1)[0],1], c = 'r', s = 50, label="No move")

        plt.savefig(data_source + "/%i_tsne_graph.pdf" % time)
        plt.clf()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_source', required=True, help="embedding transformed directory")
    args = parser.parse_args()
    main(args.data_source)
