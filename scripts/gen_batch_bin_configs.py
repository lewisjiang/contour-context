import numpy as np
import re
import os
import csv


# required file format:
#  1) timestamp and gt pose of the sensor. Ordered by gt ts. (13 elements per line)
#  2) timestamp, seq, and the path (no space) of each lidar scan bin file.
#  Ordered by lidar ts AND seq. (3 elements per line)

def gen_mulran(dir_bins, f_global_pose, sav_pos, sav_lid):
    def rotx(t, deg=False):
        if deg:
            t = t * np.pi / 180
        ct = np.cos(t)
        st = np.sin(t)

        return np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])

    def roty(t, deg=False):
        if deg:
            t = t * np.pi / 180
        ct = np.cos(t)
        st = np.sin(t)

        return np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])

    def rotz(t, deg=False):
        if deg:
            t = t * np.pi / 180
        ct = np.cos(t)
        st = np.sin(t)

        return np.array([[ct, -st, 0], [st, ct, 0], [0, 0, 1]])

    # set the calibration
    se3_6d = [1.7042, -0.021, 1.8047, 0.0001, 0.0003, 179.6654]  # calib: lidar_to_base_init_se3
    trans = np.array(se3_6d[0:3]).reshape((3, 1))

    roll = se3_6d[3]
    pitch = se3_6d[4]
    yaw = se3_6d[5]

    rot = rotz(yaw, True) * roty(pitch, True) @ rotx(roll, True)

    lidar_to_base_init_se3 = np.vstack([np.hstack([rot, trans]), np.array([[0, 0, 0, 1]])])
    print(lidar_to_base_init_se3)

    # calculate sensor gt poses
    tss = []
    poses = []
    T_wl0 = None
    tws0_set = False
    with open(f_global_pose, newline='') as cf:
        reader = csv.reader(cf, delimiter=',')
        cnt_lines = 0
        cnt_lines_valid = 0
        for row in reader:
            cnt_lines += 1
            if len(row) == 13:
                try:
                    ts_sec = float(row[0]) * 1e-9
                    tf12_base = np.array([float(a) for a in row[1:]])

                    cnt_lines_valid += 1
                    tss.append(ts_sec)

                    T_wb = np.vstack([tf12_base.reshape((3, 4)), np.array([0, 0, 0, 1])])
                    T_wl = T_wb @ np.linalg.inv(lidar_to_base_init_se3)

                    if not tws0_set:
                        T_wl0 = T_wl
                        tws0_set = True

                    T_l0l = np.linalg.inv(T_wl0) @ T_wl
                    poses.append(T_l0l[0:3, :].reshape(-1))

                except ValueError:
                    print("Not a float in line: ", row)

        print("valid gt lines read: %d/%d" % (cnt_lines_valid, cnt_lines))

    np_file_pose_dat = np.hstack([np.array(tss).reshape((-1, 1)), np.vstack(poses)])
    np.savetxt(sav_pos, np_file_pose_dat, "%.6f")

    # handle file paths. Assumption: file name is ts in nano sec
    bin_files = [f for f in os.listdir(dir_bins) if
                 os.path.isfile(os.path.join(dir_bins, f)) and f[-4:] == ".bin"]
    bin_files.sort()

    print("valid bin file names under dir: %d/%d" % (len(bin_files), len([f for f in os.listdir(dir_bins)])))

    lid_lines = []
    for i, fn in enumerate(bin_files):
        lid_lines.append("%.6f %d %s" % (eval(fn.split(".")[0]) * 1e-9, i, os.path.join(dir_bins, fn)))
    with open(sav_lid, "w") as f1:
        f1.write("\n".join(lid_lines))


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
                 os.path.isfile(os.path.join(dir_bins, f)) and f[-4:] == ".bin"]
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
    # =============================== KITTI Odometry ====================================
    # # KITTI00
    # # dir_lid_bin = "/home/lewis/Downloads/datasets/kitti_raw/2011_10_03/2011_10_03_drive_0027_sync/velodyne_points/data"
    # dir_lid_bin = "/media/lewis/S7/Datasets/kitti/odometry/dataset/sequences/00/velodyne"
    # fp_pose = "/media/lewis/S7/Datasets/kitti/odometry/poses/dataset/poses/00.txt"
    # fp_times = "/media/lewis/S7/Datasets/kitti/odometry/dataset/sequences/00/times.txt"
    # fp_calib = "/media/lewis/S7/Datasets/kitti/odometry/dataset/sequences/00/calib.txt"
    # sav_1 = "/home/lewis/catkin_ws2/src/contour-context/sample_data/ts-sens_pose-kitti00.txt"
    # sav_2 = "/home/lewis/catkin_ws2/src/contour-context/sample_data/ts-lidar_bins-kitti00.txt"

    ksq = "08"
    # KITTI08
    # dir_lid_bin = "/media/lewis/S7/Datasets/kitti/odometry/dataset/sequences/%s/velodyne" % ksq
    # # dir_lid_bin = "/home/lewis/Downloads/datasets/sequences/velodyne"
    # # dir_lid_bin = "/media/lewis/OS/downloads/sequences/velodyne"
    # fp_pose = "/media/lewis/S7/Datasets/sematic_kitti/odometry/dataset/sequences/%s/poses.txt" % ksq
    # fp_times = "/media/lewis/S7/Datasets/kitti/odometry/dataset/sequences/%s/times.txt" % ksq
    # fp_calib = "/media/lewis/S7/Datasets/kitti/odometry/dataset/sequences/%s/calib.txt" % ksq
    # sav_1 = "/home/lewis/catkin_ws2/src/contour-context/sample_data/ts-sens_pose-kitti%s.txt" % ksq
    # sav_2 = "/home/lewis/catkin_ws2/src/contour-context/sample_data/ts-lidar_bins-kitti%s.txt" % ksq

    # Onboard ssd:
    dir_lid_bin = "/media/lewis/DC36269C362677A2/d/cont2_dataset/%s/velodyne" % ksq
    fp_pose = "/media/lewis/DC36269C362677A2/d/cont2_dataset/%s/poses.txt" % ksq
    fp_times = "/media/lewis/DC36269C362677A2/d/cont2_dataset/%s/times.txt" % ksq
    fp_calib = "/media/lewis/DC36269C362677A2/d/cont2_dataset/%s/calib.txt" % ksq
    sav_1 = "/home/lewis/catkin_ws2/src/contour-context/sample_data/ts-sens_pose-kitti%s.txt" % ksq
    sav_2 = "/home/lewis/catkin_ws2/src/contour-context/sample_data/ts-lidar_bins-kitti%s.txt" % ksq

    # # mulran as KITTI: KAIST01
    # dir_lid_bin = "/media/lewis/S7/Datasets/mulran_as_kitti/sequences/51/velodyne"
    # fp_pose = "/media/lewis/S7/Datasets/mulran_as_kitti/sequences/51/poses.txt"
    # fp_times = "/media/lewis/S7/Datasets/mulran_as_kitti/sequences/51/times.txt"
    # fp_calib = "/media/lewis/S7/Datasets/mulran_as_kitti/sequences/51/calib.txt"
    # sav_1 = "/home/lewis/catkin_ws2/src/contour-context/sample_data/ts-sens_pose-kitti51.txt"
    # sav_2 = "/home/lewis/catkin_ws2/src/contour-context/sample_data/ts-lidar_bins-kitti51.txt"

    # # mulran as KITTI: RS02
    # dir_lid_bin = "/media/lewis/S7/Datasets/mulran_as_kitti/sequences/62/velodyne"
    # fp_pose = "/media/lewis/S7/Datasets/mulran_as_kitti/sequences/62/poses.txt"
    # fp_times = "/media/lewis/S7/Datasets/mulran_as_kitti/sequences/62/times.txt"
    # fp_calib = "/media/lewis/S7/Datasets/mulran_as_kitti/sequences/62/calib.txt"
    # sav_1 = "/home/lewis/catkin_ws2/src/contour-context/sample_data/ts-sens_pose-kitti62.txt"
    # sav_2 = "/home/lewis/catkin_ws2/src/contour-context/sample_data/ts-lidar_bins-kitti62.txt"

    # # mulran as KITTI: DCC02
    # dir_lid_bin = "/media/lewis/S7/Datasets/mulran_as_kitti/sequences/72/velodyne"
    # fp_pose = "/media/lewis/S7/Datasets/mulran_as_kitti/sequences/72/poses.txt"
    # fp_times = "/media/lewis/S7/Datasets/mulran_as_kitti/sequences/72/times.txt"
    # fp_calib = "/media/lewis/S7/Datasets/mulran_as_kitti/sequences/72/calib.txt"
    # sav_1 = "/home/lewis/catkin_ws2/src/contour-context/sample_data/ts-sens_pose-kitti72.txt"
    # sav_2 = "/home/lewis/catkin_ws2/src/contour-context/sample_data/ts-lidar_bins-kitti72.txt"

    gen_kitti(dir_lid_bin, fp_pose, fp_times, fp_calib, sav_1, sav_2)

    # =============================== Mulran Odometry ====================================

    # # KAIST02
    # dir_lid_bin = "/media/lewis/S7/Datasets/mulran/KAIST02/Ouster"
    # fp_gt_ts_pose = "/media/lewis/S7/Datasets/mulran/KAIST02/global_pose.csv"
    # sav_1 = "/home/lewis/catkin_ws2/src/contour-context/sample_data/ts-sens_pose-mulran-kaist02.txt"
    # sav_2 = "/home/lewis/catkin_ws2/src/contour-context/sample_data/ts-lidar_bins-mulran-kaist02.txt"

    # # Riverside02
    # dir_lid_bin = "/media/lewis/S7/Datasets/mulran/Riverside02/Ouster"
    # fp_gt_ts_pose = "/media/lewis/S7/Datasets/mulran/Riverside02/global_pose.csv"
    # sav_1 = "/home/lewis/catkin_ws2/src/contour-context/sample_data/ts-sens_pose-mulran-rs02.txt"
    # sav_2 = "/home/lewis/catkin_ws2/src/contour-context/sample_data/ts-lidar_bins-mulran-rs02.txt"

    # # DCC02
    # dir_lid_bin = "/media/lewis/S7/Datasets/mulran/DCC02/Ouster"
    # fp_gt_ts_pose = "/media/lewis/S7/Datasets/mulran/DCC02/global_pose.csv"
    # sav_1 = "/home/lewis/catkin_ws2/src/contour-context/sample_data/ts-sens_pose-mulran-dcc02.txt"
    # sav_2 = "/home/lewis/catkin_ws2/src/contour-context/sample_data/ts-lidar_bins-mulran-dcc02.txt"
    # #
    # #
    #
    # gen_mulran(dir_lid_bin, fp_gt_ts_pose, sav_1, sav_2)
