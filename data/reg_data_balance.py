import numpy as np
import glob
import shutil
import argparse

def y_loc(file):
    end = int((3500 + 999) * (30 / 1000.0))
    ydata = np.load(file)[(end - 15):end]
    ydata_start = ydata[0]
    ydata_end = ydata[-1]
    t = 0
    while ydata_start[0] < 0:
        t += 1
        ydata_start = ydata[t]
    t = -1
    while ydata_end[0] < 0:
        t -= 1
        ydata_end = ydata[t]
    ydata_end[0] = int(round(ydata_end[0] * (256 / 640.0) / 4.0))
    ydata_end[1] = int(round(ydata_end[1] * (256 / 480.0) / 4.0))
    return ydata_end

def main(main_dir):
    total_y_cnt = np.array(shape=(56,56))
    file_y_dict = {}
    for file in glob.glob("%s/Y/*.npy"% main_dir):
        y_data_end = y_loc(file)
        file_y_dict[file] = y_data_end
        total_y_cnt[y_data_end] += 1

    max_y_cnt = max(total_y_cnt)
    for file, y_data_end in file_y_dict.iteritems():
        copies = max_y_cnt/total_y_cnt[y_data_end]
        filename = file.split("/")[-1].split(".")[0]
        for c in xrange(copies):
            shutil.copyfile("%s/X/%s.png" % (main_dir, filename), "%s/X/%s_copy_%i.png" % (main_dir, filename, c))
            shutil.copyfile("%s/Y/%s.npy" % (main_dir, filename), "%s/X/%s_copy_%i.npy" % (main_dir, filename, c))
            shutil.copyfile("%s/X/%s.npy" % (main_dir, filename), "%s/X/%s_copy_%i.npy" % (main_dir, filename, c))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', required=True, help="Main data directory to rebalance")
    args = parser.parse_args()
    main(args.data_dir)

