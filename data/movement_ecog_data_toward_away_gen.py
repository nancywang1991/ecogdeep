import numpy as np
import cPickle as pickle
import glob
import os
import shutil
import pdb
from pyESig2.movement.joint_movement_norm import *

main_class_dir = "/home/wangnxr/dataset/ecog_vid_combined_a0f_day11/"
joint_dir = "/home/wangnxr/pose/"
crop_loc = "/home/wangnxr/pose/crop_coords/"
save_dir = "/home/wangnxr/dataset_toward_away/" + main_class_dir.split("/")[-2]
print save_dir
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    os.makedirs(save_dir + "/train")
    os.makedirs(save_dir + "/train/toward")
    os.makedirs(save_dir + "/train/away")
    os.makedirs(save_dir + "/test")
    os.makedirs(save_dir + "/test/toward")
    os.makedirs(save_dir + "/test/away")
    os.makedirs(save_dir + "/val")
    os.makedirs(save_dir + "/val/toward")
    os.makedirs(save_dir + "/val/away")

types = ["train", "test", "val"]

def calc_dist(a,b):
    final_dist = []
    for i, coord in enumerate(b):
        if np.all(np.array(coord) > -1000) and np.all(np.array(a[i]) > -1000):
            final_dist.append(np.sqrt((coord[0]-a[i][0])**2 + (coord[1]-a[i][1])**2))
        else:
            final_dist.append(-1000)
    return final_dist


for type in types:
    for file in glob.glob(main_class_dir + "%s/*_arm_1/*.npy" % type):
        filename = file.split("/")[-1].split(".")[0]
        sbj_id, day, vid, _, frame = filename.split("_")
        start = int(frame) + 15 - 30 * 5
        if start < 0:
            joints_file_prev = "%s/%s_%s/%s_%s_%04i.txt" % (joint_dir, sbj_id, day, sbj_id, day, int(vid) - 1)
            try:
                crop_coords_prev = np.array([np.array([int(coord) for coord in crop_coord.split(',')]) for crop_coord in
                                             open(os.path.normpath("%s/%s_%s_%04i.txt" % (
                                             crop_loc, sbj_id, day, int(vid) - 1))).readlines()])
            except IOError:
                print "Crop coords for %s not found" % (filename)
            poses = np.array([numerate_coords(row) for row in (open(joints_file_prev)).readlines()])
            poses_normalized_prev = np.array(
                [normalize_to_camera(row, crop_coord) for row, crop_coord in zip(poses, crop_coords_prev)])
            poses_normalized_filtered = filter_confidence(poses_normalized_prev, poses[:, :, 2])
            poses_normalized_prev = my_savgol_filter(poses_normalized_filtered, 21, 3, axis=0)

        joints_file = "%s/%s_%s/%s_%s_%s.txt" % (joint_dir, sbj_id, day, sbj_id, day, vid)
        try:
            crop_coords = np.array([np.array([int(coord) for coord in crop_coord.split(',')]) for crop_coord in
                                    open(os.path.normpath(
                                        "%s/%s_%s_%04i.txt" % (crop_loc, sbj_id, day, int(vid)))).readlines()])
        except IOError:
            print "Crop coords for %s not found" % (filename)
        poses = np.array([numerate_coords(row) for row in (open(joints_file)).readlines()])
        poses_normalized = np.array(
            [normalize_to_camera(row, crop_coord) for row, crop_coord in zip(poses, crop_coords)])
        poses_normalized_filtered = filter_confidence(poses_normalized, poses[:, :, 2])
        poses_normalized = my_savgol_filter(poses_normalized_filtered, 21, 3, axis=0)
        if file.split("/")[-2] == "r_arm_1":
            start_dist = calc_dist(poses_normalized[:(start + 4 * 30 + 15), 1], poses_normalized[:(start + 4 * 30 + 15), 0])
            end_dist = calc_dist(poses_normalized[:(start + 5 * 30), 1], poses_normalized[:(start + 5 * 30), 0])
        elif file.split("/")[-2] == "l_arm_1":
            start_dist = calc_dist(poses_normalized[:(start + 4 * 30 + 15), 2], poses_normalized[:(start + 4 * 30 + 15), 0])
            end_dist = calc_dist(poses_normalized[:(start + 5 * 30), 2], poses_normalized[:(start + 5 * 30), 0])
        else:
            print "Error: Not working with %s yet" % file.split("/")[-2]
            break
        if end_dist < start_dist:
            shutil.copy(file, "%s/%s/toward/" % (save_dir, type))
        else:
            shutil.copy(file, "%s/%s/away/" % (save_dir, type))
