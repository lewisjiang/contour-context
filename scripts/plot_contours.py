#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib.patches import Ellipse
import matplotlib.lines as lines
import matplotlib.transforms as transforms


def confidence_ellipse_2d(cov_xy, mean_xy, ax, n_std=3.0, facecolor='none', edgecolor='black', **kwargs):
    """
    New cov->ellipse that directly operates on init parameters
    :param cov_xy:
    :param mean_xy:
    :param ax:
    :param n_std:
    :param facecolor:
    :param edgecolor:
    :param kwargs:
    :return:
    """
    cov = np.array(cov_xy)
    cov = (cov + cov.T) / 2

    if cov[0, 0] * cov[1, 1] == 0:
        return None
    J, V = np.linalg.eigh(cov)  # The eigenvalues in ascending order

    width = n_std * np.sqrt(J[1]) * 2
    height = n_std * np.sqrt(J[0]) * 2
    angle = np.arctan2(V[1, 1], V[1, 0])  # row (first dim) is the order parameter

    # direct parameters to use in other plot script
    # print("(%f, %f), %f, %f, %f" % (mean_xy[0], mean_xy[1], width, height, angle / np.pi * 180))

    ellipse = Ellipse((mean_xy[0], mean_xy[1]),
                      width=width,
                      height=height,
                      angle=angle / np.pi * 180,
                      facecolor=facecolor,
                      edgecolor=edgecolor,
                      **kwargs)
    return ax.add_patch(ellipse)


def confidence_ellipse_fromcov_2d(cov_xy, mean_xy, ax, n_std=3.0, facecolor='none', edgecolor='black', **kwargs):
    cov = np.array(cov_xy)
    if cov[0, 0] * cov[1, 1] == 0:
        return
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
                      width=ell_radius_x * 2,
                      height=ell_radius_y * 2,
                      facecolor=facecolor,
                      edgecolor=edgecolor,
                      **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_xy[0], mean_xy[1])

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def main_axis_from_cov_2d(cov_xy, mean_xy, ax, color='black', **kwargs):
    cov = np.array(cov_xy)
    if cov[0, 0] * cov[1, 1] == 0:
        return
    eig_val, eig_vec = np.linalg.eigh(cov)  # default using lower triangle, eigval in ascending order
    end_xy = mean_xy + eig_vec[:, 1] * np.sqrt(eig_val[1])

    line = lines.Line2D([mean_xy[0], end_xy[0]], [mean_xy[1], end_xy[1]],
                        lw=1, color=color, axes=ax)
    ax.add_line(line)


def read_data_from_file(fpath):
    data = []
    with open(fpath, 'r') as f0:
        flg_beg = False
        flg_end = False
        for line in f0:
            elems = line.strip().split()
            if len(elems) > 0 and elems[0] == "DATA_START":
                flg_beg = True
                continue
            if len(elems) > 0 and elems[0] == "DATA_END":
                flg_end = True
            if flg_end:
                break
            if not flg_beg:
                continue
            # 20 elements per line:
            # 0:level, 1:cell_cnt, 2:pos_mean, 4:pos_cov, 8:eig_vals,
            #  10:eig_vecs, 14:eccen, 15:vol3_mean, 16:com, 18,19:..
            tmp_data = []
            for i in range(20):
                tmp_data.append(eval(elems[i]))
            data.append(tmp_data)
    return np.array(data)


def plot_contours(raw_data=None, levels=None, legends=()):
    """
    Plot ellipsoids for given datasets
    :param raw_data:
    :param levels:
    :param legends:
    :return:
    """
    if levels is None:
        levels = []
    if raw_data is None:
        raw_data = []

    assert len(raw_data) == len(legends)

    if not len(levels):
        print("No levels selected")
        return

    print(data_names)

    levels.sort()
    level_map = dict()
    for i, l in enumerate(levels):
        level_map[l] = i

    fig, axs = plt.subplots(1, len(levels), figsize=(len(levels) * 3, 4))
    # fig, axs = plt.subplots(2, len(levels)//2, figsize=(8, 4))
    # axs = [axs[i//(len(levels)//2), i%3] for i in range(len(levels))]
    #
    # fig = plt.figure(figsize=(9, 5))
    # axs = [fig.add_subplot(231), fig.add_subplot(232), fig.add_subplot(233),
    #         fig.add_subplot(234), fig.add_subplot(235), fig.add_subplot(236)]

    cmap = mpl.cm.get_cmap('jet')
    used_colors = []

    max_xy = [0, 0]
    tf_ed = True  # whether we use a transform T_delta (T_tgt = T_delta * T_src) to have a clearer view in tgt frame
    T_delta_str = """
  -0.999999 -0.00140861     145.041
 0.00140861   -0.999999     149.279
          0           0           1
    """
    T_delta_elem = [eval(i) for i in T_delta_str.split()]
    assert not tf_ed or len(T_delta_elem) == 9
    T_delta = np.array(T_delta_elem).reshape([3, 3])  # bev to bev transform (orig: pixel 0, 0), not sensor to sensor

    for i in range(len(raw_data)):
    # for i in range(1, -1,-1):
        data_color = cmap((i + 0.5) / len(raw_data))
        if i == 0:
            data_color = cm.get_cmap('Greens')(0.85)
        if i == 1:
            data_color = cm.get_cmap('Reds')(0.85)
        # print(data_color)
        used_colors.append(data_color)
        level_beg = 0
        last_level = -1
        for j in range(raw_data[i].shape[0]):
            raw_level = int(raw_data[i][j, 0])
            if raw_level not in levels:
                continue
            if raw_level != last_level:
                level_beg = j
                last_level = raw_level
            mean_xy = raw_data[i][j, 2:4]
            # cov_xy = raw_data[i][j, 4:8].reshape((2, 2)).T  # raw cov, some data maybe invisible, since a line
            J = np.diag(raw_data[i][j, 8:10])
            V = raw_data[i][j, 10:14].reshape((2, 2,)).T
            cov_xy = V @ J @ V.T

            if i == 0 and tf_ed:  # manual transform, T_tgt = T_delta * T_src and the first one is src
                m_rot = T_delta[0:2, 0:2]
                m_trans = T_delta[0:2, 2:3]  # T_delta[0:2, 2] has shape (2,)
                mean_xy = m_rot @ np.expand_dims(mean_xy, axis=1) + m_trans
                mean_xy = np.squeeze(mean_xy)
                cov_xy = m_rot @ cov_xy @ m_rot.T

            selected_idx = [_ for _ in range(20)]
            # selected_idx = [1, 3, 5, 7]
            # selected_idx = [2, 4, 6, 8]

            max_xy[0] = max_xy[0] if max_xy[0] > mean_xy[0] else mean_xy[0]
            max_xy[1] = max_xy[1] if max_xy[1] > mean_xy[1] else mean_xy[1]

            if j - level_beg in selected_idx:
                print("scan", i, "lev ", raw_level, "#", j - level_beg)
                confidence_ellipse_2d(cov_xy, mean_xy, axs[level_map[raw_level]], 2.0,
                                      linestyle='-', edgecolor='none', facecolor=data_color, alpha=0.5)
                # main_axis_from_cov_2d(cov_xy, mean_xy, axs[level_map[raw_level]], data_color)
            # 0. text: ranking in all contours of the scan
            # axs[level_map[raw_level]].text(mean_xy[0], mean_xy[1], str(j), color=data_color, fontsize=6)

            if j - level_beg in selected_idx:
                # 1. text: the cell count ranking place of the data
                axs[level_map[raw_level]].text(mean_xy[0], mean_xy[1], str(j - level_beg), color=data_color, fontsize=6)

                # 2. text: the cellcount i.e. [1]
                # axs[level_map[raw_level]].text(mean_xy[0], mean_xy[1], str(raw_data[i][j, 1]), color=data_color,
                #                                fontsize=6)

    # border = [0, 100, 0, 100]
    # border = [0, 150, 0, 150]
    border = [0, max_xy[0] + 10, 0, max_xy[1] + 10]
    for i in range(len(levels)):
        # axs[i].set_title('Level %d' % levels[i])
        axs[i].set_xlabel('Level %d' % levels[i])
        axs[i].set_aspect(aspect=1)
        axs[i].axis(border)

        # https://pythonguides.com/matplotlib-remove-tick-labels/
        axs[i].yaxis.set_ticklabels([])
        axs[i].yaxis.set_ticks([])
        axs[i].xaxis.set_ticklabels([])
        axs[i].xaxis.set_ticks([])

    l_clr = [Ellipse((0, 0),
                     width=2,
                     height=2,
                     facecolor='none',
                     edgecolor=c, linestyle='-') for c in used_colors]
    axs[-1].legend(l_clr, legends)
    # axs[-1].legend()

    img_name = "".join([i for i in "-".join(data_names) if i.isalnum() or i in "-_."])
    if tf_ed:
        img_name += "-tf_ed"

    plt.savefig('../results/ellipse_img/ellipse_%s.svg' % img_name, format='svg', dpi=600, bbox_inches='tight',
                pad_inches=0)

    plt.show()


if __name__ == "__main__":
    # f1 = "../results/contours_orig_1317357625557814.txt"   # -s 0
    # f2 = "../results/contours_orig_1317357625661737.txt"
    #
    # f91 = "../results/contours_orig_1317357736560660.txt"
    # f92 = "../results/contours_orig_1317357736664336.txt"  # -s 111.1
    #
    # # data_names = ("t=0", "t=1", "t=91", "t=92")
    # data_names = ("t=0", "t=92")
    # # data_names = ("t=0", "t=1")
    #
    # dat1 = read_data_from_file(f1)
    # # dat2 = read_data_from_file(f2)
    # # dat3 = read_data_from_file(f91)
    # dat4 = read_data_from_file(f92)
    #
    # # plot_contours([dat1, dat2, dat3, dat4], [2, 3, 4], data_names)
    # plot_contours([dat1, dat4], [0, 1, 2, 3, 4, 5], data_names)
    # # plot_contours([dat1, dat2], [1, 2, 3, 4], data_names)

    # read data saved from bin
    # f_new = "../results/contours_orig-0000000034.txt"
    # f_old = "../results/contours_orig-0000002437.txt"
    # f_new = "../results/contours_orig-0000000119.txt"
    # f_old = "../results/contours_orig-0000002511.txt"

    # f_new = "../results/contours_orig-0000001561.txt"
    # f_old = "../results/contours_orig-0000002576.txt"

    # f_new = "../results/contours_accu-0000000119.txt"
    # f_old = "../results/contours_accu-0000002511.txt"

    # f_old = "../results/contours_orig-0000000558.txt"
    # f_new = "../results/contours_orig-0000001316.txt"
    # data_names = ("t=1561", "t=2576")
    # data_names = ("t=565", "t=1316")
    # data_names = ("t=0558", "t=1316")

    # f_old = "../results/contours_orig-0000000769.txt"
    # f_new = "../results/contours_orig-0000001512.txt"
    # data_names = ("t=0769", "t=1512")

    # #########################################################
    # Case 3: Two-scan comparison:
    # seq_old = 895
    # seq_new = 2632
    # seq_old = 905
    # seq_new = 2636

    # seq_old = 890  # final bus fn
    # seq_new = 2632

    # seq_old = 806  # seek farther
    # seq_new = 1562

    # # Sequence 00:
    # seq_old = 486  # seek farther
    # seq_new = 1333

    # f_old = "../results/contours_orig-000000%04d.txt" % seq_old
    # f_new = "../results/contours_orig-000000%04d.txt" % seq_new

    # #########################################################
    # Sequence 08, odom seq:
    seq_old = 237
    seq_new = 1648
    f_old = "../results/contours_orig-assigned_id_0000%04d.txt" % seq_old
    f_new = "../results/contours_orig-assigned_id_0000%04d.txt" % seq_new

    data_names = ("t=%04d" % seq_old, "t=%04d" % seq_new)
    cont_data = [read_data_from_file(f_old), read_data_from_file(f_new)]

    plot_contours(cont_data, [0, 1, 2, 3, 4, 5], data_names)
