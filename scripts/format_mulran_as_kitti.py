import numpy as np
import re
import os
import shutil


def copy_mulran_to_kitti_format(f_bin_info, dir_as_kitti):
    with open(f_bin_info, "r") as f1:
        bins = f1.readlines()
        print(bins)

    for i, bin_src in enumerate(bins):
        src = bin_src.strip()
        tgt = os.path.join(dir_as_kitti, "%06d.bin" % i)

        shutil.copy2(src, tgt)
        print("Copied ", src, " to ", tgt)


if __name__ == "__main__":
    bin_file = "/home/lewis/catkin_ws2/src/contour-context/results/mulran_to_kitti/used_bins.txt"
    tgt_dir = "/media/lewis/S7/Datasets/mulran_as_kitti/sequences/6==2/velodyne"

    copy_mulran_to_kitti_format(bin_file, tgt_dir)
