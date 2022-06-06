#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib.patches import Ellipse
import matplotlib.lines as lines
import matplotlib.transforms as transforms


def plot_1d_gmm(means, stds):
    num_divs = 64
    xs = x = np.linspace(-np.pi, np.pi, num_divs)
    ys = np.zeros_like(xs)

    for m, sd in zip(means, stds):
        for i in range(xs.shape[0]):
            ys[i] += np.exp(-0.5 * ((m - xs[i]) / sd) ** 2) / np.sqrt(2 * np.pi * sd * sd)

    plt.plot(xs, ys)
    plt.show()


if __name__ == "__main__":
    # ms = [0.38, 0.25, -0.08, -0.06, 0.2, -0.77, 0.11, 0.24, ]
    ms = [0.38, 1.4454, 1.4678, 0.9192, 1.4716, -0.0112, 0.7654, 2.4218, 2.9742, 1.4914, 2.268, ]
    vs = [np.pi / 32 for i in ms]
    plot_1d_gmm(ms, vs)
