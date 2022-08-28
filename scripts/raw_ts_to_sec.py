import datetime


def raw_kitti_ts_to_seconds(ts_path, float_path):
    """
    Transforms KITTI raw dataset's human-readable time stamp str into float number timestamp
    :param ts_path:
    :param float_path:
    :return:
    """
    with open(ts_path, 'r') as f0:
        lines = f0.readlines()
        stamps = []
        for line in lines:
            dt = datetime.datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
            stamps.append(str(dt.timestamp()) + "\n")
            print(dt.timestamp())
        with open(float_path, 'w') as f1:
            f1.writelines(stamps)


if __name__ == "__main__":
    print("")

    # kitti_raw_ts_file = "/home/lewis/Downloads/datasets/kitti_raw/2011_09_30/2011_09_30_drive_0018_sync/velodyne_points/timestamps.txt"
    # processed_file = "../results/kitti_seq05_seconds.txt"

    kitti_raw_ts_file = "/home/lewis/Downloads/datasets/kitti_raw/2011_10_03/2011_10_03_drive_0027_sync/velodyne_points/timestamps.txt"
    processed_file = "../results/kitti_seq00_seconds.txt"

    raw_kitti_ts_to_seconds(kitti_raw_ts_file, processed_file)
