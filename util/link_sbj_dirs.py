import subprocess
import glob
import pdb

#sbjs = ["a0f", "d65", "cb4", "c95"]
sbjs = ["a0f", "d65"]
new_dirname = "/data2/users/wangnxr/dataset/ecog_mni_ellipv2_deep_impute_%s" % "_".join(sbjs)
old_dirroot = "/data2/users/wangnxr/dataset/ecog_mni_ellipv2_deep_impute"

for sbj in sbjs:
    old_dirname = old_dirroot + "_" + sbj
    subprocess.call("mkdir %s" % new_dirname, shell=True)
    for subfolder in ["train", "test", "val"]:
	subprocess.call("mkdir %s/%s" % (new_dirname, subfolder), shell=True)
	subprocess.call("mkdir %s/%s/mv_0/" % (new_dirname, subfolder), shell=True)
	subprocess.call("mkdir %s/%s/r_arm_1/" % (new_dirname, subfolder), shell=True)
	files = glob.glob("%s/%s/mv_0/*.npy" % (old_dirname, subfolder))
	for file in files:
		subprocess.call("sudo ln -s %s %s/%s/mv_0/" % (file, new_dirname, subfolder), shell=True)

	files = glob.glob("%s/%s/*_arm_1/*.npy" % (old_dirname, subfolder))
        for file in files:	
		subprocess.call("sudo ln -s %s %s/%s/r_arm_1" % (file, new_dirname, subfolder), shell=True)

