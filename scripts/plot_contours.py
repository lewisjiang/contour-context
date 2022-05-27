#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


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

    levels.sort()
    level_map = dict()
    for i, l in enumerate(levels):
        level_map[l] = i

    fig, axs = plt.subplots(1, len(levels), figsize=(len(levels) * 4, 4))
    cmap = mpl.cm.get_cmap('jet')
    used_colors = []

    for i in range(len(raw_data)):
        data_color = cmap((i + 0.5) / len(raw_data))
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

            confidence_ellipse_fromcov_2d(cov_xy, mean_xy, axs[level_map[raw_level]], 2.0, edgecolor=data_color,
                                          linestyle='--')
            # 0. text: ranking in all contours of the scan
            # axs[level_map[raw_level]].text(mean_xy[0], mean_xy[1], str(j), color=data_color, fontsize=6)
            # 1. text: the cell count ranking place of the data
            if j - level_beg < 9:
                axs[level_map[raw_level]].text(mean_xy[0], mean_xy[1], str(j - level_beg), color=data_color, fontsize=6)
            # 2. text: the area
            # axs[level_map[raw_level]].text(mean_xy[0], mean_xy[1], str(raw_data[i][j,1]), color=data_color, fontsize=6)

    border = [0, 100, 0, 100]
    for i in range(len(levels)):
        axs[i].set_title('Level %d' % levels[i])
        axs[i].set_aspect(aspect=1)
        axs[i].axis(border)

    lines = [Ellipse((0, 0),
                     width=2,
                     height=2,
                     facecolor='none',
                     edgecolor=c, linestyle='--') for c in used_colors]
    axs[-1].legend(lines, legends)
    # axs[-1].legend()

    plt.savefig('../results/cont0.svg', format='svg', dpi=600, bbox_inches='tight', pad_inches=0)

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

    f_new = "../results/contours_orig_1317357723149910000.txt"
    f_old = "../results/contours_orig_1317357630437796000.txt"
    data_names = ("t=new", "t=old")
    data1 = read_data_from_file(f_new)
    data2 = read_data_from_file(f_old)
    plot_contours([data1, data2], [ 1, 2, 3, 4], data_names)
