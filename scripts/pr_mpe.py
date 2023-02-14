import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib
import math
from scipy.spatial import KDTree

thres_dist = 5.0


def get_gt_sens_poses(fpath_gt_sens_poses):
    """

    :return: 2d matrix, each row is a 12 dim elements
    """

    with open(fpath_gt_sens_poses, "r") as fp:
        lines = fp.readlines()

    res = []
    for line in lines:
        assert len(line.strip().split()) == 13
        res.append([eval(x) for x in line.strip().split()[1:]])

    return np.vstack(res)


def get_maxf1_idx(data):
    max_f1 = 0
    idx = -1
    max_pt = None
    for d in data:
        cur = 2 * d[0] * d[1] / (d[0] + d[1]) if (d[0] + d[1]) > 0 else 0

        if max_f1 < cur:
            max_f1 = cur
            idx = d[2]
            max_pt = d
    print("Max f1 point: ", max_pt)
    return max_f1, idx


class SimpleRMSE:
    def __init__(self):
        self.sum_sqs = 0
        self.sum_abs = 0
        self.cnt_sqs = 0

    def add_one_error(self, err_vec):
        self.cnt_sqs += 1
        tmp = 0
        for i in err_vec:
            tmp += i ** 2
        self.sum_sqs += tmp
        self.sum_abs += math.sqrt(tmp)

    def get_rmse(self):
        if self.cnt_sqs:
            return math.sqrt(self.sum_sqs / self.cnt_sqs)
        else:
            return -1

    def get_mean(self):
        if self.cnt_sqs:
            return self.sum_abs / self.cnt_sqs
        else:
            return -1


def get_points_ours2(fp_gt_sens_poses, fp_outcome):
    print("In ours2")
    plots_data = []

    print(fp_gt_sens_poses)
    print(fp_outcome)
    pr_points = []

    gt_pose = get_gt_sens_poses(fp_gt_sens_poses)  # the sensor poses must be ordered by time/creation/acquisition
    gt_positive = np.zeros(gt_pose.shape[0])
    gt_points = gt_pose[:, [3, 7, 11]]
    tree = KDTree(gt_points)

    for i in range(gt_pose.shape[0]):
        near_points = tree.query_ball_point(gt_points[i, :], thres_dist)
        for j in near_points:
            if j < i - 150:
                gt_positive[i] = 1
                break

    with open(fp_outcome, "r") as f1:
        lines = f1.readlines()
        est = []
        for line in lines:
            line_info = line.strip().split()
            assert len(line_info) > 3

            pairing = line_info[1].split('-')
            idx_curr = int(pairing[0])

            est_line = [eval(line_info[2]), 0, 0, idx_curr]
            if pairing[1] != 'x':
                idx_best = int(pairing[1])
                if np.linalg.norm(gt_pose[idx_curr].reshape(3, 4)[:, 3] -
                                  gt_pose[idx_best].reshape(3, 4)[:, 3]) < thres_dist:
                    est_line[1] = 1

                # 3. if the overall is P
            est_line[2] = gt_positive[idx_curr]

            est.append(est_line)
            # print(est_line)

        orig_est = est

        est = np.vstack(est)
        est = est[(-est[:, 0]).argsort()]  # sort by correlation, larger better

        tp = 0
        fp = 0

        for i in range(est.shape[0]):
            if est[i, 1]:
                tp += 1
            else:
                fp += 1

            fn = 0
            for j in range(i, est.shape[0]):
                if est[j, 2]:
                    fn += 1

            pr_points.append([tp / (tp + fn), tp / (tp + fp), est[i, 3]])
        # pr_points.append([0, 1])

        points = np.vstack(pr_points)[:, 0:2]
        points = points[points[:, 0].argsort()]
        plots_data.append(points)

        # get max F1
        max_f1, f1_pose_idx = get_maxf1_idx(pr_points)
        print("Max F1 score: %f @%d " % (max_f1, int(f1_pose_idx)))

        # calc rmse for scores above max f1 sim
        sim_thres = eval(lines[int(f1_pose_idx)].split()[2])
        print("sim thres for Max F1 score: %f" % sim_thres)

        sr_trans = SimpleRMSE()
        sr_rot = SimpleRMSE()
        for i, line in enumerate(lines):
            line_info = line.strip().split()
            assert len(line_info) > 5

            # if current is TP
            if eval(line_info[2]) >= sim_thres and orig_est[i][1] == 1 and orig_est[i][2] == 1:
                sr_trans.add_one_error([eval(line_info[3]), eval(line_info[4])])
                sr_rot.add_one_error([eval(line_info[5]), ])

        print("TP count: ", sr_rot.cnt_sqs)
        print("Rot mean err: ", sr_rot.get_mean() / np.pi * 180)
        print("Rot rmse    : ", sr_rot.get_rmse() / np.pi * 180)
        print("Trans mean err: ", sr_trans.get_mean())
        print("Trans rmse    :  ", sr_trans.get_rmse())

    return plots_data


def main(fp_gt_sens_poses, fp_outcome):
    fig, axes = plt.subplots(1, 1, figsize=(18, 6))

    data_res = [
        get_points_ours2(fp_gt_sens_poses, fp_outcome)
    ]
    data_names = [
        "Ours"
    ]

    assert len(data_res) == len(data_names)

    titles = [fp_outcome]

    for i in range(1):
        ax = axes

        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')

        ax.set_xlim([0, 1.02])
        ax.set_ylim([0, 1.02])

        ax.set_title(titles[i])
        ax.tick_params(axis="y", direction="in")
        ax.tick_params(axis="x", direction="in")

        used_names = []
        used_colors = []

        for j, data1 in enumerate(data_res):
            if data1[i].size == 0:
                continue
            ax.plot(data1[i][:, 0], data1[i][:, 1], color="C%d" % (9 - j))
            used_names.append(data_names[j])
            used_colors.append("C%d" % (9 - j))

        ax.legend(used_names, loc=3)

    plt.show()


if __name__ == "__main__":
    file_gt_sens_poses = "../sample_data/ts-sens_pose-kitti08.txt"
    file_outcome = "../results/outcome_txt/outcome-kitti08.txt"
    main(file_gt_sens_poses, file_outcome)
