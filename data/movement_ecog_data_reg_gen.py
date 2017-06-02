import numpy as np
import cPickle as pickle
import glob
import os
import shutil
import pdb

main_class_dir = "/home/wangnxr/dataset/ecog_vid_combined_a0f_day11/"
movement_dir = "/home/wangnxr/mvmt/"
save_dir = "/home/wangnxr/dataset_reg/" + main_class_dir.split("/")[-2]
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

for file in glob.glob(main_class_dir + "train/*_arm_1/*.npy"):
    filename = file.split("/")[-1].split(".")[0]
    sbj_id, day, vid, _, frame = filename.split("_")
    start = int(frame) + 15 - 30 * 5
    if start < 0:
        movement_file_prev = pickle.load(
            open("%s/%s_%s/%s_%s_%04i_movement.p" % (movement_dir, sbj_id, day, sbj_id, day, int(vid)-1)))
    movement_file = pickle.load(open("%s/%s_%s/%s_%s_%s_movement.p" % (movement_dir, sbj_id, day, sbj_id, day, vid)))
    if file.split("/")[-2] == "r_arm_1":
        if start<0:
            movement_array = np.hstack([movement_file_prev[start:,1], movement_file[:(start+5*30),1]])
        else:
            movement_array = movement_file[start:(start+5*30), 1]
    elif file.split("/")[-2] == "l_arm_1":
        if start<0:
            movement_array = np.hstack([movement_file_prev[start:,2], movement_file[:(start+5*30),2]])
        else:
            movement_array = movement_file[start:(start+5*30), 2]
    else:
        print "Error: Not working with %s yet" % file.split("/")[-2]
        break
    np.save("%s/train/Y/%s.npy" % (save_dir, filename), movement_array)
    shutil.copy(file, "%s/train/X/" % save_dir)

for file in glob.glob(main_class_dir + "test/*_arm_1/*.npy"):
    filename = file.split("/")[-1].split(".")[0]
    sbj_id, day, vid, _, frame = filename.split("_")
    start = int(frame) + 15 - 30 * 5
    if start < 0:
        movement_file_prev = pickle.load(
            open("%s/%s_%s/%s_%s_%04i_movement.p" % (movement_dir, sbj_id, day, sbj_id, day, int(vid)-1)))
    movement_file = pickle.load(open("%s/%s_%s/%s_%s_%s_movement.p" % (movement_dir, sbj_id, day, sbj_id, day, vid)))
    if file.split("/")[-2] == "r_arm_1":
        if start<0:
            movement_array = np.hstack([movement_file_prev[start:,1], movement_file[:(start+5*30),1]])
        else:
            movement_array = movement_file[start:(start+5*30), 1]
    elif file.split("/")[-2] == "l_arm_1":
        if start<0:
            movement_array = np.hstack([movement_file_prev[start:,2], movement_file[:(start+5*30),2]])
        else:
            movement_array = movement_file[start:(start+5*30), 2]
    else:
        print "Error: Not working with %s yet" % file.split("/")[-2]
        break
    np.save("%s/test/Y/%s.npy" % (save_dir, filename), movement_array)
    shutil.copy(file, "%s/test/X/" % save_dir)

for file in glob.glob(main_class_dir + "val/*_arm_1/*.npy"):
    filename = file.split("/")[-1].split(".")[0]
    sbj_id, day, vid, _, frame = filename.split("_")
    start = int(frame) + 15 - 30 * 5
    if start < 0:
        movement_file_prev = pickle.load(
            open("%s/%s_%s/%s_%s_%04i_movement.p" % (movement_dir, sbj_id, day, sbj_id, day, int(vid)-1)))
    movement_file = pickle.load(open("%s/%s_%s/%s_%s_%s_movement.p" % (movement_dir, sbj_id, day, sbj_id, day, vid)))
    if file.split("/")[-2] == "r_arm_1":
        if start<0:
            movement_array = np.hstack([movement_file_prev[start:,1], movement_file[:(start+5*30),1]])
        else:
            movement_array = movement_file[start:(start+5*30), 1]
    elif file.split("/")[-2] == "l_arm_1":
        if start<0:
            movement_array = np.hstack([movement_file_prev[start:,2], movement_file[:(start+5*30),2]])
        else:
            movement_array = movement_file[start:(start+5*30), 2]
    else:
        print "Error: Not working with %s yet" % file.split("/")[-2]
        break
    np.save("%s/val/Y/%s.npy" % (save_dir, filename), movement_array)
    shutil.copy(file, "%s/val/X/" % save_dir)

