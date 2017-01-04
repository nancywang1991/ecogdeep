import argparse
import glob
import cPickle as pickle
import csv
import numpy as np
import os
import pdb
import matplotlib.pyplot as plt
import mne.io

def scoring(truth, predicted):
    plt.plot(np.array(range(len(truth)))/30.0, truth*5, label="truth")
    plt.plot(np.array(range(len(truth)))/30.0, predicted*2, label="predicted")
    plt.ylim([0,6])
    plt.legend()
    plt.show()
    pred_locs = np.where(predicted==1)[0]
    for loc in pred_locs:
        predicted[loc-15:loc+15] = 1
    true_correct = sum(np.logical_and(predicted,truth==1))
    true = sum(truth)
    detected = sum(predicted)

    precision = true_correct/float(detected)
    recall = true_correct/true

    return [precision, recall, true/30.0]

def main(mv_file, edf, save_dir, vid_start_end, offset):
    vid_name = os.path.split(mv_file)[-1].split('.')[0]
    print vid_name
    mv_file = pickle.load(open(mv_file))
    sbj_id, day, vid_num, _ = vid_name.split('_')
    #print edf_file
    #edf = pyedflib.EdfReader(edf_file)

    start_sec = (vid_start_end["start"][int(vid_num)] - vid_start_end["start"][0]).total_seconds()
    end_sec = (vid_start_end["end"][int(vid_num)] - vid_start_end["start"][0]).total_seconds()
    #pdb.set_trace()
#    n_channels = len(edf.ch_names)
#    edf_clip_len = int(end_sec*1000)-int(start_sec*1000)
#    edf_clip = np.zeros(shape=(n_channels, edf_clip_len))
#    for c in xrange(n_channels): 
#        pdb.set_trace()
#        edf_clip[c,:] = edf[c][0][0,int(start_sec*1000):int(end_sec*1000)]
#        print c
    edf_clip = edf[:,int(start_sec*1000):int((end_sec+1)*1000)]
    left_arm_mvmt = np.sum(mv_file[:,(2,4,6)], axis=1)
    right_arm_mvmt = np.sum(mv_file[:,(1,3,5)], axis=1)
    head_mvmt = mv_file[:,0]
    train_dir = os.path.join(save_dir, "train")
    test_dir = os.path.join(save_dir, "test")

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    if not os.path.exists(os.path.join(train_dir, "l_arm_1")):
        os.makedirs(os.path.join(train_dir,"head_0"))
        os.makedirs(os.path.join(train_dir,"head_1"))
        os.makedirs(os.path.join(train_dir,"r_arm_0"))
        os.makedirs(os.path.join(train_dir,"r_arm_1"))
        os.makedirs(os.path.join(train_dir,"l_arm_0"))
        os.makedirs(os.path.join(train_dir,"l_arm_1"))
        os.makedirs(os.path.join(train_dir,"mv_1"))
        os.makedirs(os.path.join(train_dir,"mv_0"))

    if not os.path.exists(os.path.join(test_dir,"l_arm_1")):
        os.makedirs(os.path.join(test_dir,"head_0"))
        os.makedirs(os.path.join(test_dir,"head_1"))
        os.makedirs(os.path.join(test_dir,"r_arm_0"))
        os.makedirs(os.path.join(test_dir,"r_arm_1"))
        os.makedirs(os.path.join(test_dir,"l_arm_0"))
        os.makedirs(os.path.join(test_dir,"l_arm_1"))
        os.makedirs(os.path.join(test_dir,"mv_1"))
        os.makedirs(os.path.join(test_dir,"mv_0"))
    if np.random.randint(100) < 75:
        cur_dir = train_dir
    else:
        cur_dir = test_dir

    for f in range(offset+1+15,len(mv_file)-1, 10):
        edf_part = edf_clip[:,int((f-offset-15)*(1000/30.0)):int((f-offset-15)*(1000/30.0)+500)]
        
        if np.mean(left_arm_mvmt[f:f+5]>5):
            #cv2.imwrite(os.path.join(cur_dir, "l_arm_1", "%s_%i.png" %(vid_name,f - offset)), img)
            pickle.dump(edf_part, open(os.path.join(cur_dir, "mv_1", "%s_%i.p" % (vid_name, f - offset)), "wb"))
        elif np.mean(right_arm_mvmt[f:f+5]>5):
            #cv2.imwrite(os.path.join(cur_dir, "r_arm_1", "%s_%i.png" % (vid_name, f - offset)), img)
            pickle.dump(edf_part, open(os.path.join(cur_dir, "mv_1", "%s_%i.p" % (vid_name, f - offset)), "wb"))
        elif np.mean(head_mvmt[f:f+5]>3):
            #cv2.imwrite(os.path.join(cur_dir, "head_1", "%s_%i.png" % (vid_name, f - offset)), img)
            pickle.dump(edf_part, open(os.path.join(cur_dir, "mv_1", "%s_%i.p" % (vid_name, f - offset)), "wb"))

    for f in range(offset+1+15, len(mv_file)-1, 60):
        edf_part = edf_clip[:, int((f - offset - 15) * (1000 / 30.0)):int((f - offset - 15) * (1000 / 30.0)+500)]
        flag = 0

        if np.all(left_arm_mvmt[f:f+5] >= 0) and np.mean(left_arm_mvmt[f:f + 5]) < 1:
            #cv2.imwrite(os.path.join(cur_dir, "l_arm_0", "%s_%i.png" % (vid_name, f - offset)), img)
            flag+=1
        if np.all(right_arm_mvmt[f:f + 5] >= 0) and np.mean(right_arm_mvmt[f:f + 5]) < 1:
            #cv2.imwrite(os.path.join(cur_dir, "r_arm_0", "%s_%i.png" % (vid_name, f - offset)), img)
            flag+=1
        if np.all(head_mvmt[f:f + 5] >= 0) and np.mean(head_mvmt[f:f + 5]) < 1:
            #cv2.imwrite(os.path.join(cur_dir, "head_0", "%s_%i.png" % (vid_name, f - offset)), img)
            flag+=1
        if flag==3:
            pickle.dump(edf_part, open(os.path.join(cur_dir, "mv_0", "%s_%i.p" % (vid_name, f - offset)), "wb"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mv_dir', required=True, help="Joint movement directory")
    parser.add_argument('-e', '--edf_dir', required=True, help="edf directory")
    parser.add_argument('-t', '--vid_time_dir', required=True, help="video start and end time directory")
    parser.add_argument('-s', '--save_dir', required=True, help="Save directory")
    parser.add_argument('-o', '--offset', default=15, type=int, help="how many frames into the future")
    args = parser.parse_args()
    files = glob.glob(args.mv_dir + "/*.p")
    sbj_id, day, vid_num, _ = os.path.split(files[0])[-1].split(".")[0].split("_")  
    ecog_name = os.path.join(args.edf_dir, "%s_%s.edf" %( sbj_id, day))
    edf = mne.io.read_raw_edf(ecog_name)
    n_channels = len(edf.ch_names)
    edf_data = np.zeros(shape=(96, len(edf)))
    for c in xrange(96):
        print c
        edf_data[c,:] = edf[c][0][0,:]
    for file in sorted(files):
        #pdb.set_trace()
        sbj_id, day, vid_num, _ = os.path.split(file)[-1].split(".")[0].split("_")
        vid_start_end = pickle.load(open(os.path.join(args.vid_time_dir, "%s_%s.p" % (sbj_id, day))))
        main(file, edf_data, args.save_dir, vid_start_end, args.offset)
        
