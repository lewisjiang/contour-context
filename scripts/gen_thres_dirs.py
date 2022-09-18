import numpy as np
import os

config_str = """
i_ovlp_sum          %d       %d
i_ovlp_max_one      %d       %d
i_in_ang_rng        %d       %d

i_indiv_sim         %d       %d
i_orie_sim          %d       %d

correlation         %f    %f
area_perc           %f    %f
neg_est_dist        %f    %f
"""


def create_config_folders(beg_idx=0):
    cfg_constell = [3, 4, 5, 6]
    cfg_corr = [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7]
    cfg_area = [0.01, 0.03, 0.05, 0.10]
    cfg_ndist = [-10.01, -8.01, -6.01, -4.01, -3.01]
    cfg_idx = beg_idx

    rng = [3, 3, 3]
    divs = len(cfg_constell)

    for i in range(divs):
        beg_corr = int(len(cfg_corr) / divs * i)
        beg_corr = min(beg_corr, len(cfg_corr) - rng[0])
        for i1 in range(beg_corr, beg_corr + rng[0]):

            beg_area = int(len(cfg_area) / divs * i)
            beg_area = min(beg_area, len(cfg_area) - rng[1])
            for i2 in range(beg_area, beg_area + rng[1]):

                beg_ndist = int(len(cfg_ndist) / divs * i)
                beg_ndist = min(beg_ndist, len(cfg_ndist) - rng[1])
                for i3 in range(beg_ndist, beg_ndist + rng[2]):
                    print("%03d" % cfg_idx, cfg_constell[i], cfg_corr[i1], cfg_area[i2], cfg_ndist[i3])

                    cfg = config_str % (cfg_constell[i], cfg_constell[i] + 3,
                                        cfg_constell[i], cfg_constell[i] + 3,
                                        cfg_constell[i], cfg_constell[i] + 3,
                                        cfg_constell[i], cfg_constell[i] + 3,
                                        cfg_constell[i], cfg_constell[i] + 3,
                                        cfg_corr[i1], cfg_corr[i1] + 0.15,
                                        cfg_area[i2], cfg_area[i2] + 0.1,
                                        cfg_ndist[i3], cfg_ndist[i3] + 0.01)
                    cfg_dir = os.path.join("../results/batch_pr_tests", "%03d" % cfg_idx)
                    os.makedirs(cfg_dir, exist_ok=True)

                    cfg_path = os.path.join(cfg_dir, "batch_pr.cfg")
                    if os.path.isfile(cfg_path):
                        print("overwriting existing cofig")
                        exit(-1)
                    with open(cfg_path, "w") as f:
                        f.write(cfg)

                    cfg_idx += 1


def create_config_folders_manual(beg_idx=0):
    cfg_idx = beg_idx
    threses = [
        [3, 0.1, 0.01, -10],
        [3, 0.2, 0.01, -10],
        [3, 0.2, 0.01, -12],
        [3, 0.1, 0.01, -12],
        [7, 0.75, 0.15, -4],
        [7, 0.80, 0.15, -3],
    ]
    for thres in threses:
        cfg = config_str % (int(thres[0]), int(thres[0]) + 3,
                            int(thres[0]), int(thres[0]) + 3,
                            int(thres[0]), int(thres[0]) + 3,
                            int(thres[0]), int(thres[0]) + 3,
                            int(thres[0]), int(thres[0]) + 3,
                            thres[1], thres[1] + 0.15,
                            thres[2], thres[2] + 0.1,
                            thres[3], thres[3] + 0.01)
        cfg_dir = os.path.join("../results/batch_pr_tests", "%03d" % cfg_idx)
        os.makedirs(cfg_dir, exist_ok=True)

        cfg_path = os.path.join(cfg_dir, "batch_pr.cfg")
        if os.path.isfile(cfg_path):
            print("overwriting existing cofig")
            exit(-1)
        with open(cfg_path, "w") as f:
            f.write(cfg)
        cfg_idx += 1


if __name__ == "__main__":
    print()
    # create_config_folders(0)
    create_config_folders_manual(109)
