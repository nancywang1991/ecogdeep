import argparse
import glob
import cPickle as pickle
import csv
import numpy as np
import os
import pdb
import matplotlib.pyplot as plt
from pyESig2.vid.my_video_capture import my_video_capture
import cv2

"""Video data generation based on selecting frames from the ecog data.


Example:
        $ python movement_video_data_gen_v2.py -n ecog file names -v video directory -s save directory


"""

def save_imgs(imgs, offsets, main_name):
    """Save concatenated frames.

    Args:
        imgs (list of PIL images): consecutive frames.
        offsets (list of int): the list of frame numbers.
    Returns:
        None
    """
    img = np.concatenate(imgs, axis=1)
    a,b,c,d=offsets[-4],offsets[-3],offsets[-2],offsets[-1]
    cv2.imwrite(os.path.join("%s_%i_%i_%i_%i.png" % (main_name,a,b,c,d)), img)

def main(npy_file, vid_dir, save_dir):
    """Main function.

    Args:
        npy_file (str): File containing list of files to extract from video based on ECoG results
        vid_dir (str): Video directory to extract videos from
        save_dir (str): Directory to save video data into.
    Returns:
        None
    """
    npy_files = open(npy_file).readlines()
    vid_dict = {}
    for file in npy_files:
        vid_name = "_".join(os.path.split(file)[-1].split("_")[:3])
        frame = int(os.path.split(file)[-1].split("_")[-1].split(".")[0])
        if vid_name not in vid_dict:
            vid_dict[vid_name] = []
        vid_dict[vid_name].append(frame)
    for vid, frames in vid_dict.iteritems():
        sorted_f = sorted(np.array(frames))
        keep = [sorted_f[0]]
        for f in range(1,len(sorted_f)):
            if sorted_f[f]-sorted_f[f-1] > 5:
                keep.append(sorted_f[f])
        vid_dict[vid] = keep
    for vid, frames in vid_dict.iteritems():
        vid_file = my_video_capture("/".join([vid_dir, "_".join(vid.split("_")[:2]),vid]) + ".avi", 30)
        total_frames = vid_file.get_total_frames()
        for f in frames:
            imgs = []
            if f>(6*10+1) and f < total_frames-8:
                for f2 in range(f-6*10+1, f+7+1, 6):
                    vid_file.forward_to(f2)
                    imgs.append(cv2.resize(vid_file.read(), (340,256)))
                    save_imgs(imgs, range(f - 6 * 10, f +6, 6),os.path.join(save_dir, vid))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--npy_file', required=True, help="ecog file names")
    parser.add_argument('-v', '--vid_dir', required=True, help="video directory")
    parser.add_argument('-s', '--save_dir', required=True, help="Save directory")
    args = parser.parse_args()

    main(args.npy_file, args.vid_dir, args.save_dir)

