import glob
import shutil
import pdb

dest_dir = "dataset/ecog_vid_combined_a0f/"
ecog_files = glob.glob("dataset/ecog_offset_15_arm_a0f/*/*/*")
#pdb.set_trace()
for ecog_file in ecog_files:
        #pdb.set_trace()
        vid = "_".join(ecog_file.split("/")[-1].split("_")[:3])
        frame = ecog_file.split("/")[-1].split("_")[-1].split(".")[0]
        #frame = str(int(frame)-10)
        vid_file = glob.glob("dataset/vid_offset_0_a0f/*/*/%s_*_%s.png" % (vid, frame))
        if len(vid_file)>0:
                shutil.copy(vid_file[0], "%s/%s.png" % (dest_dir, "/".join(ecog_file.split(".")[0].split("/")[-3:])))
                shutil.copy(ecog_file, "%s/%s" % (dest_dir, "/".join(ecog_file.split("/")[-3:])))

