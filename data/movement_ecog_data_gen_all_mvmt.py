import argparse
import glob
import cPickle as pickle
import matplotlib
matplotlib.use("agg")
import csv
import numpy as np
import os
import pdb
import matplotlib.pyplot as plt
import mne.io
import gc
from pyESig2.vid.video_sync.vid_start_end import get_disconnected_times
import scipy.sparse

"""ECoG data generation of a single patient from a single day based on selecting all movement sections from the movement data.


Example:
        $ python movement_ecog_data_gen.py
                -m Joint movement directory
                -e edf directory
                -t directory with video start and end time python files
                -df disconnect text file for this patient and day
                -s save directory for final ecog sections
                -o how many frames to offset movement frame (ex. 15 means getting segments that span t-5s:t where t is the time of movement)
                -n optional file containing mean and standard deviation values for normalization
"""


def main(mv_file, edf, save_dir, vid_start_end, start_time, offset):
    """extract ECoG segment where there was definitely movement and definitely not movement

    Args:
        mv_file (string-pickle file): file containing amount of movement for each frame
        edf (array[channels, time]): loaded edf file
        save_dir (str): root folder to save ecog chunks
        start_time (datetime): Start time of the day file
        offset (int): frames to offset movement (target) frame
    Returns:
        None
    """
    vid_name = os.path.split(mv_file)[-1].split('.')[0]
    print vid_name
    mv_file = pickle.load(open(mv_file))
    sbj_id, day, vid_num, _ = vid_name.split('_')

    start_sec = (vid_start_end["start"][int(vid_num)] - start_time).total_seconds()
    end_sec = (vid_start_end["end"][int(vid_num)] - start_time).total_seconds()
    if int((start_sec-5)*1000) < 0:
        # Clip is too early
        return
    edf_clip = edf[:,int((start_sec-5)*1000):int((end_sec+1)*1000)]
    left_arm_mvmt = mv_file[:,2]
    right_arm_mvmt = mv_file[:,1]
    head_mvmt = mv_file[:,0]
    train_dir = os.path.join(save_dir, "train")

    # Set up folders
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(os.path.join(train_dir, "l_arm_1")):
        os.makedirs(os.path.join(train_dir,"head_0"))
        os.makedirs(os.path.join(train_dir,"head_1"))
        os.makedirs(os.path.join(train_dir,"r_arm_0"))
        os.makedirs(os.path.join(train_dir,"r_arm_1"))
        os.makedirs(os.path.join(train_dir,"l_arm_0"))
        os.makedirs(os.path.join(train_dir,"l_arm_1"))
        os.makedirs(os.path.join(train_dir,"mv_1"))
        os.makedirs(os.path.join(train_dir,"mv_0"))

    cur_dir = train_dir

    # Save if meeting movement or non movement conditions
    for f in range(0,len(mv_file), 2):
        flag = 0
        offset_f = int((f-offset-15)*1000/30.0)
        # Add 5 seconds because edf clip is offset by 5 seconds in the other direction
        edf_part = edf_clip[:,(offset_f  - 4000+5000):(offset_f  + 1000+5000)]
        if edf_part.shape[1] == 5000:
            if np.mean(left_arm_mvmt[f:f+5])>1 and np.all(left_arm_mvmt[f-10:f] >= 0) and np.mean(left_arm_mvmt[f-10:f]) < 0.5:
                flag = 1
                save_filename = os.path.join(cur_dir, "l_arm_1", "%s_%i" % (vid_name, f ))
                np.save(save_filename + ".npy", edf_part)
            if np.mean(right_arm_mvmt[f:f+5])>1 and np.all(right_arm_mvmt[f-10:f] >= 0) and np.mean(right_arm_mvmt[f-10:f]) < 0.5:
                flag = 1
                save_filename = os.path.join(cur_dir, "r_arm_1", "%s_%i" % (vid_name, f ))
                np.save(save_filename + ".npy", edf_part)
            if np.mean(head_mvmt[f:f+5])>1 and np.all(head_mvmt[f-10:f] >= 0) and np.mean(head_mvmt[f-10:f]) < 0.5:
                flag = 1
                save_filename = os.path.join(cur_dir, "head_1", "%s_%i" % (vid_name, f ))
                np.save(save_filename + ".npy", edf_part)
            if (f ) % 10 == 0:
                if np.all(left_arm_mvmt[f-30:f+30] >= 0) and np.mean(left_arm_mvmt[f-30:f + 30]) < 0.5:
                    flag+=1
                if np.all(right_arm_mvmt[f-30:f + 30] >= 0) and np.mean(right_arm_mvmt[f-30:f + 30]) < 0.5:
                    flag+=1
                if np.all(head_mvmt[f-30:f + 30] >= 0) and np.mean(head_mvmt[f-30:f + 30]) < 0.5:
                    flag+=1
                if flag==3:
                    save_filename = os.path.join(cur_dir, "mv_0", "%s_%i" % (vid_name, f ))
                    np.save(save_filename + ".npy", edf_part)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mv_dir', required=True, help="Joint movement directory")
    parser.add_argument('-e', '--edf_dir', required=True, help="edf directory")
    parser.add_argument('-t', '--vid_time_dir', required=True, help="video start and end time directory")
    parser.add_argument('-df', '--disconnect_file', required=True, help="video disconnect times")
    parser.add_argument('-s', '--save_dir', required=True, help="Save directory")
    parser.add_argument('-o', '--offset', default=15, type=int, help="how many frames into the future")
    parser.add_argument('-n', '--norm_factors_file', help="optional normalization mean and stdev")
    args = parser.parse_args()

    # Assumes mv_dir only has one day of one subject
    files = glob.glob(args.mv_dir + "/*.p")
    sbj_id, day, vid_num, _ = os.path.split(files[0])[-1].split(".")[0].split("_")
    vid_start_end = pickle.load(open(os.path.join(args.vid_time_dir, "%s_%s.p" % (sbj_id, day))))
    ecog_name = os.path.join(args.edf_dir, "%s_%s.edf" %( sbj_id, day))
    edf = mne.io.read_raw_edf(ecog_name)
    n_channels = edf.ch_names.index('EOGL')-2

    edf_data = np.zeros(shape=(n_channels, len(edf)))
    if args.norm_factors_file is None:
        norm_factors = np.zeros(shape=(n_channels,2))
    else:
        norm_factors = pickle.load(open(args.norm_factors_file))

    # If no norm factor file, create new one
    if args.norm_factors_file is None:
        for c in range(n_channels):
            print "normalization:%i" % (c+1)
            temp_data,_ = edf[c+1,:]
            if args.norm_factors_file is None:
                norm_factors[c,0] = np.mean(temp_data)
                norm_factors[c,1] = np.std(temp_data)
        pickle.dump(norm_factors, open("%s/%s_%s_norm_factors.p" % (args.save_dir, sbj_id, day), "wb"))

    start_time, end_time, start, end = get_disconnected_times(args.disconnect_file)

    # Load and process daily data in thirds

    for c in range(n_channels):
        print "edf_data_part1:%i" % (c+1)
        edf_data[c,:int(0.35*len(edf))], _ = (edf[c+1,:int(0.35*len(edf))]-norm_factors[c,0])/norm_factors[c,1]

    for file in sorted(files)[:len(files)/3]:
        #pdb.set_trace()
        sbj_id, day, vid_num, _ = os.path.split(file)[-1].split(".")[0].split("_")
        main(file, edf_data, args.save_dir, vid_start_end, start_time, args.offset)

    for c in range(n_channels):
        print "edf_data_part2:%i" % (c+1)
        edf_data[c,int(0.3*len(edf)):int(0.7*len(edf))],_ = (edf[c+1,int(0.3*len(edf)):int(0.7*len(edf))]-norm_factors[c,0])/norm_factors[c,1]
    for file in sorted(files)[len(files)/3:2*len(files)/3]:
        #pdb.set_trace()
        sbj_id, day, vid_num, _ = os.path.split(file)[-1].split(".")[0].split("_")
        main(file, edf_data, args.save_dir, vid_start_end, start_time, args.offset)
    
    edf_data = []
    gc.collect()
    edf_data = np.zeros(shape=(n_channels, len(edf)))

    for c in range(n_channels):
        print "edf_data_part3:%i" % (c+1)
        edf_data[c,int(0.65*len(edf)):],_ = (edf[c+1,int(0.65*len(edf)):]-norm_factors[c,0])/norm_factors[c,1]
    for file in sorted(files)[2*len(files)/3:]:
        #pdb.set_trace()
        sbj_id, day, vid_num, _ = os.path.split(file)[-1].split(".")[0].split("_")
        main(file, edf_data, args.save_dir, vid_start_end, start_time, args.offset)

