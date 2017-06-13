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
save_dir = "/home/wangnxr/dataset_xy_reg/" + main_class_dir.split("/")[-2]
print save_dir
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    os.makedirs(save_dir + "/train")
    os.makedirs(save_dir + "/train/X")
    os.makedirs(save_dir + "/train/Y")
    os.makedirs(save_dir + "/test")
    os.makedirs(save_dir + "/test/X")
    os.makedirs(save_dir + "/test/Y")
    os.makedirs(save_dir + "/val")
    os.makedirs(save_dir + "/val/X")
    os.makedirs(save_dir + "/val/Y")

types = ["train", "test", "val"]

for type in types:
    for file in glob.glob(main_class_dir + "%s/*_arm_1/*.npy" % type):
        filename = file.split("/")[-1].split(".")[0]
        sbj_id, day, vid, _, frame = filename.split("_")
        start = int(frame) + 15 - 30 * 5
        if start < 0:
            joints_file_prev = "%s/%s_%s/%s_%s_%04i.txt" % (joint_dir, sbj_id, day, sbj_id, day, int(vid)-1)
            try:
                crop_coords_prev = np.array([np.array([int(coord) for coord in crop_coord.split(',')]) for crop_coord in
                                        open(os.path.normpath("%s/%s_%s_%04i.txt" % (crop_loc, sbj_id, day, int(vid)-1))).readlines()])
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
                                         open(os.path.normpath("%s/%s_%s_%04i.txt" % (crop_loc, sbj_id, day, int(vid)))).readlines()])
        except IOError:
            print "Crop coords for %s not found" % (filename)
        poses = np.array([numerate_coords(row) for row in (open(joints_file)).readlines()])
        poses_normalized = np.array(
            [normalize_to_camera(row, crop_coord) for row, crop_coord in zip(poses, crop_coords)])
        poses_normalized_filtered = filter_confidence(poses_normalized, poses[:, :, 2])
        poses_normalized = my_savgol_filter(poses_normalized_filtered, 21, 3, axis=0)
        if file.split("/")[-2] == "r_arm_1":
            if start<0:
		
                movement_array = np.vstack([poses_normalized_prev[start:,1], poses_normalized[:(start+5*30),1]])
            else:
                movement_array = poses_normalized[start:(start+5*30), 1]
        elif file.split("/")[-2] == "l_arm_1":
            if start<0:
                movement_array = np.vstack([poses_normalized_prev[start:,2], poses_normalized[:(start+5*30),2]])
            else:
                movement_array = poses_normalized[start:(start+5*30), 2]
        else:
            print "Error: Not working with %s yet" % file.split("/")[-2]
            break
        np.save("%s/%s/Y/%s.npy" % (save_dir, type, filename), movement_array)
        shutil.copy(file, "%s/%s/X/" % (save_dir, type))
