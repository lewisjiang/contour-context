import sys
import numpy as np
from plot_contours import read_data_from_file


def compare_contours(posid_src, posid_tgt, lev, seq_src, seq_tgt):
    assert lev >= 0 and seq_tgt >= 0 and seq_src >= 0
    f_src = "../results/contours_orig-000000%04d.txt" % (int(posid_src))
    f_tgt = "../results/contours_orig-000000%04d.txt" % (int(posid_tgt))

    dat_src = read_data_from_file(f_src)
    dat_tgt = read_data_from_file(f_tgt)

    print(dat_src.shape)
    print(dat_tgt.shape)

    line_src = None
    line_tgt = None

    seq_cnt = 0
    for line in dat_src:
        if int(line[0]) < lev:
            continue
        elif int(line[0]) > lev:
            break
        else:
            if seq_cnt == seq_src:
                line_src = line
                break
            else:
                seq_cnt += 1

    seq_cnt = 0
    for line in dat_tgt:
        if int(line[0]) < lev:
            continue
        elif int(line[0]) > lev:
            break
        else:
            if seq_cnt == seq_tgt:
                line_tgt = line
                break
            else:
                seq_cnt += 1

    assert line_src is not None and line_tgt is not None
    # print(line_src)
    # print(line_tgt)

    # 20 elements per line:
    str_struct = {0: "level", 1: "cell_cnt", 2: "pos_mean", 4: "pos_cov", 8: "eig_vals",
                  10: "eig_vecs", 14: "eccen", 15: "vol3_mean", 16: "com", 18: "ecc_feat", 19: "com_feat"}
    pos_struct = list(str_struct)
    int_idx_set = {0, 1, 14, 18, 19}

    fmt = "%10s | %24s | %24s |"
    print(fmt % ("property", str(posid_src), str(posid_tgt)))
    for i in range(len(pos_struct)):
        idx_beg = pos_struct[i]
        if i == len(pos_struct) - 1:
            idx_end = idx_beg + 1
        else:
            idx_end = pos_struct[i + 1]

        data_str1 = ""
        data_str2 = ""

        for j in range(idx_beg, idx_end):
            if idx_beg in int_idx_set:
                data_str1 += "%d " % int(line_src[j])
                data_str2 += "%d " % int(line_tgt[j])
            else:
                data_str1 += "%.2f " % line_src[j]
                data_str2 += "%.2f " % line_tgt[j]

        data_str1 = data_str1.strip()
        data_str2 = data_str2.strip()

        print(fmt % (str_struct[idx_beg], data_str1, data_str2))


if __name__ == "__main__":
    print(sys.argv)
    if len(sys.argv) == 6:
        print("Comparing %d(%d, %d) with %d(%d, %d)" % (int(sys.argv[1]), int(sys.argv[3]), int(sys.argv[4]),
                                                        int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[5])))
        compare_contours(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    else:
        print("Dummy example")
        compare_contours(769, 1512, 2, 3, 1)
