#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import re
import os


# required file format:
#  1) timestamp and gt pose of the sensor. Ordered by gt ts. (13 elements per line)
#  2) timestamp, seq, and the path (no space) of each lidar scan bin file.
#  Ordered by lidar ts AND seq. (3 elements per line)


def gen_kitti(dir_bins, f_pose, f_times, f_calib, sav_pos, sav_lid, addr_bin_beg=0):
    """

    :param dir_bins: the dir of lidar `.bin` files. Should be compatible with bins from `raw`
    :param f_pose: the file of gt left camera poses
    :param f_times: the relative time stamps for the odom sequence (assume lidar and left cam are concurrent)
    :param f_calib: the calibration file to get lidar-left_cam transform: `T_leftcam_velod_`
    :param sav_pos:
    :param sav_lid:
    :param addr_bin_beg:
    :return:
    """

    bin_files = [os.path.join(dir_bins, f) for f in os.listdir(dir_bins) if
                 os.path.isfile(os.path.join(dir_bins, f)) and f[-3:] == "bin"]
    bin_files.sort()
    print("# bin files: ", len(bin_files))
    for bf in bin_files:
        assert len(bf.split()) <= 1  # no space in the path string

    with open(f_pose, "r") as f1:
        gt_poses = [[eval(p) for p in line.strip().split()] for line in f1.readlines() if line.strip()]
        print("# gt poses: ", len(gt_poses))

    with open(f_times, "r") as f1:
        gt_times = [eval(line.strip()) for line in f1.readlines() if line.strip()]
        print("# gt times: ", len(gt_times))

    T_leftcam_velod_ = np.hstack([np.identity(3), np.zeros((3, 1))])
    with open(f_calib, "r") as f1:
        for line in f1.readlines():
            segs = line.strip().split(" ")
            if segs and segs[0] == "Tr:":
                assert len(segs) == 13
                T_leftcam_velod_ = np.array([eval(x) for x in segs[1:]]).reshape((3, 4))

    T_leftcam_velod_ = np.vstack([T_leftcam_velod_, np.array([[0, 0, 0, 1]])])
    print("T_leftcam_velod_:\n", T_leftcam_velod_)

    assert len(gt_poses) == len(gt_times)
    assert len(bin_files) >= len(gt_poses) + addr_bin_beg

    # Create files:
    all_pose_12col = []
    lid_lines = []
    all_seq = [x for x in range(len(gt_times))]  # other possibilities?

    for i, pose_line in enumerate(gt_poses):
        # The tf provided as gt in KITTI
        tmp_T_lc0_lc = np.vstack([np.array(pose_line).reshape((3, 4)), np.array([[0, 0, 0, 1]])])
        T_w_velod = np.linalg.inv(T_leftcam_velod_) @ tmp_T_lc0_lc @ T_leftcam_velod_

        all_pose_12col.append(T_w_velod[:3, :].reshape(1, 12))
        lid_lines.append("%.6f %d %s" % (gt_times[i], all_seq[i], bin_files[i + addr_bin_beg]))

    np.savetxt(sav_pos, np.hstack([np.array(gt_times).reshape((len(gt_times), 1)), np.vstack(all_pose_12col)]), "%.6f")

    with open(sav_lid, "w") as f1:
        f1.write("\n".join(lid_lines))


if __name__ == "__main__":
    # # KITTI00
    # # dir_lid_bin = "/home/lewis/Downloads/datasets/kitti_raw/2011_10_03/2011_10_03_drive_0027_sync/velodyne_points/data"
    # dir_lid_bin = "/media/lewis/S7/Datasets/kitti/odometry/dataset/sequences/00/velodyne"
    # fp_pose = "/media/lewis/S7/Datasets/kitti/odometry/poses/dataset/poses/00.txt"
    # fp_times = "/media/lewis/S7/Datasets/kitti/odometry/dataset/sequences/00/times.txt"
    # fp_calib = "/media/lewis/S7/Datasets/kitti/odometry/dataset/sequences/00/calib.txt"
    # sav_1 = "/home/lewis/catkin_ws2/src/contour-context/sample_data/ts-sens_pose-kitti00.txt"
    # sav_2 = "/home/lewis/catkin_ws2/src/contour-context/sample_data/ts-lidar_bins-kitti00.txt"

    # KITTI08
    dir_lid_bin = "/media/lewis/S7/Datasets/kitti/odometry/dataset/sequences/08/velodyne"
    fp_pose = "/media/lewis/S7/Datasets/semantic_kitti/odometry/dataset/sequences/08/poses.txt"
    fp_times = "/media/lewis/S7/Datasets/kitti/odometry/dataset/sequences/08/times.txt"
    fp_calib = "/media/lewis/S7/Datasets/kitti/odometry/dataset/sequences/08/calib.txt"
    sav_1 = "/home/lewis/catkin_ws2/src/contour-context/sample_data/ts-sens_pose-kitti08.txt"
    sav_2 = "/home/lewis/catkin_ws2/src/contour-context/sample_data/ts-lidar_bins-kitti08.txt"

    gen_kitti(dir_lid_bin, fp_pose, fp_times, fp_calib, sav_1, sav_2, 0)
