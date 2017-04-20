import glob
import shutil
import pdb
import os

dest_dir = "/home/wangnxr/dataset/ecog_vid_combined_d65_day9/"
ecog_files = glob.glob("/home/wangnxr/dataset/ecog_offset_0_d65/*/*/*")
#pdb.set_trace()
for ecog_file in ecog_files:
        #pdb.set_trace()
	sub_subfolder =  "%s/%s" % (dest_dir, "/".join(ecog_file.split(".")[0].split("/")[-3:-1]))
        subfolder =  "%s/%s" % (dest_dir, ecog_file.split(".")[0].split("/")[-3])
        vid = "_".join(ecog_file.split("/")[-1].split("_")[:3])
        frame = ecog_file.split("/")[-1].split("_")[-1].split(".")[0]
        #frame = str(int(frame)-10)
        if not os.path.exists(subfolder):
	    os.makedirs(subfolder)
	if not os.path.exists(sub_subfolder):
            os.makedirs(sub_subfolder)

        vid_file = glob.glob("/home/wangnxr/dataset/vid_offset_0_d65/%s_*_%s.png" % (vid, frame))
        if len(vid_file)>0:
                shutil.copy(vid_file[0], "%s/%s.png" % (dest_dir, "/".join(ecog_file.split(".")[0].split("/")[-3:])))
                shutil.copy(ecog_file, "%s/%s" % (dest_dir, "/".join(ecog_file.split("/")[-3:])))

