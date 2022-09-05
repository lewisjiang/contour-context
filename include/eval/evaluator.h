//
// Created by lewis on 8/22/22.
//

#ifndef CONT2_EVALUATOR_ROS_H
#define CONT2_EVALUATOR_ROS_H

#include "cont2/contour_db.h"
#include "tools/pointcloud_util.h"
#include "tools/algos.h"

template<int dim>
struct SimpleRMSE {
  double sum_sqs = 0;
  double sum_abs = 0;
  int cnt_sqs = 0;

  SimpleRMSE() = default;

  void addOneErr(const double *d) {
    cnt_sqs++;
    double tmp = 0;
    for (int i = 0; i < dim; i++) {
      tmp += d[i] * d[i];
    }
    sum_sqs += tmp;
    sum_abs += std::sqrt(tmp);
  }

  double getRMSE() const { return cnt_sqs ? std::sqrt(sum_sqs / cnt_sqs) : -1; }

  double getMean() const { return cnt_sqs ? sum_abs / cnt_sqs : -1; }
};

struct PredictionOutcome {
  enum Res {
    TP, FP, TN, FN
  };

  int id_src = -1;  //
  int id_tgt = -1;
  Res tfpn = Res::TN; // the most insignificant type
  double est_err[3]{};  // TP, FP: the error param on SE2, else: all zero
};

// Definition of loader
// 1. use script to generate 2 files: (all ts are in ns)
//  1) timestamp and gt pose (z up) of the sensor. Ordered by gt ts. (13 elements per line)
//  2) timestamp, seq, and the path (no space) of each lidar scan bin file.
//    Ordered by lidar ts AND seq. (3 elements per line)
// 2. load the .bin data in sequence, and find the gt pose
class ContLCDEvaluator {
  struct LaserScanInfo {
//  bool has_gt_pose = false;  // default/useless lidar
    bool has_gt_positive_lc = false;
    Eigen::Isometry3d sens_pose;
    int seq = 0;
    double ts = 0;
    std::string fpath;

    LaserScanInfo() = default;
  };

  // valid input info about lidar scans
  std::vector<LaserScanInfo> laser_info_;  // use predefined {seq-id, bin file} for better traceability
  std::vector<int> assigned_seqs_;  // actually a seq:addr map

  // param:
  const double ts_diff_tol = 10e-3;  // 10ms. In Mulran, the gt is given at an interval of about 10ms
  const double min_time_excl = 15.0;  // exclude 15s

  // bookkeeping variables
  int p_lidar_curr = -1;
//  std::map<int, LaserScanInfo>::iterator it_lidar_curr;  // int index is a little safer than iterator/pointer?

  // benchmark recorders
  SimpleRMSE<2> tp_trans_rmse, all_trans_rmse;
  SimpleRMSE<1> tp_rot_rmse, all_rot_rmse;
  std::vector<PredictionOutcome> pred_records;

public:
  ContLCDEvaluator(const std::string &fpath_pose, const std::string &fpath_laser) {
    std::fstream infile1, infile2;
    std::string sbuf, pname;
    // TODO:
    //  1. read data from the 2 aforementioned files
//    std::vector<PoseDatum> pose_data;
    std::vector<double> gt_tss;
    std::vector<Eigen::Isometry3d> gt_poses;

    infile1.open(fpath_pose, std::ios::in);
    if (infile1.rdstate() != std::ifstream::goodbit) {
      std::cerr << "Error opening gt pose file: " << fpath_pose << std::endl;
      return;
    }

    const int line_len = 13; // timestamp and the 12-element sensor gt pose
    while (std::getline(infile1, sbuf)) {
      std::istringstream iss(sbuf);

      double tmp;
      Eigen::Vector3d tmp_trans;
      Eigen::Matrix3d tmp_rot_mat;
      Eigen::Quaterniond tmp_rot_q;
      Eigen::Isometry3d tmp_tf;
      for (int i = 0; i < line_len; i++) {
        CHECK(iss >> tmp);
        if (i == 0)
          gt_tss.push_back(tmp);
        else {
          if ((i - 1) % 4 == 3)
            tmp_trans((i - 1) / 4) = tmp;
          else
//            tmp_rot_mat((i - 1) / 4, (i - 1) % 4);  // `-Wuninitialized` cannot detect this bug!
            tmp_rot_mat((i - 1) / 4, (i - 1) % 4) = tmp;
        }
      }

      tmp_rot_q = Eigen::Quaterniond(tmp_rot_mat);
      tmp_tf.setIdentity();
      tmp_tf.rotate(tmp_rot_q);
      tmp_tf.pretranslate(tmp_trans);

      gt_poses.emplace_back(tmp_tf);
    }
    infile1.close();
    CHECK_EQ(gt_poses.size(), gt_tss.size());
    printf("Added %lu stamped gt poses.\n", gt_poses.size());

    std::vector<int> sort_permu(gt_poses.size());
    std::iota(sort_permu.begin(), sort_permu.end(), 0);
    std::sort(sort_permu.begin(), sort_permu.end(), [&](const int &a, const int &b) {
      return gt_tss[a] < gt_tss[b];
    });

    apply_sort_permutation(sort_permu, gt_poses);
    apply_sort_permutation(sort_permu, gt_tss);

    //  2. Align to each laser scan a gt pose the closest ts. Log the scan files that have no associated gt
    //  poses and skip them at this stage.
    infile2.open(fpath_laser, std::ios::in);
    if (infile2.rdstate() != std::ifstream::goodbit) {
      std::cerr << "Error opening laser info file: " << fpath_laser << std::endl;
      return;
    }

    std::vector<double> lidar_ts;
    std::vector<int> assigned_seq;
    std::vector<std::string> bin_paths;
    while (std::getline(infile2, sbuf)) {
      std::istringstream iss(sbuf);

      double ts;
      int seq;
      std::string bin_path;

      if (iss >> ts) {
        lidar_ts.emplace_back(ts);

        iss >> seq;
        assigned_seq.emplace_back(seq);

        iss >> bin_path;
        bin_paths.emplace_back(bin_path);

//        printf("%.6f %d %s\n", ts, seq, bin_path.c_str());
      }
    }
    infile2.close();
    CHECK_EQ(lidar_ts.size(), assigned_seq.size());
    CHECK_EQ(lidar_ts.size(), bin_paths.size());
    printf("Added %lu laser bin paths.\n", bin_paths.size());

    // filter the lidar bin paths
    int cnt_valid_scans = 0;
    for (int i = 0; i < lidar_ts.size(); i++) {
      int gt_idx = lookupNN<double>(lidar_ts[i], gt_tss, ts_diff_tol);
      if (gt_idx < 0)
        continue;
      LaserScanInfo tmp_info;
      tmp_info.sens_pose = gt_poses[gt_idx];
      tmp_info.fpath = bin_paths[i];
      tmp_info.ts = lidar_ts[i];
      tmp_info.seq = assigned_seq[i];
      cnt_valid_scans++;
      laser_info_.emplace_back(tmp_info);
      assigned_seqs_.emplace_back(assigned_seq[i]);
    }
    printf("Found %d laser scans with gt poses.\n", cnt_valid_scans);

    // check ordering
    for (auto it2 = laser_info_.begin(); it2 != laser_info_.end();) {
      auto it1 = it2++;
      if (it2 != laser_info_.end()) {
        CHECK_LT(it1->seq, it2->seq);
        CHECK_LT(it1->ts, it2->ts);
      }
    }
    printf("Ordering check passed\n");

    // add info about gt loop closure
    int cnt_gt_lc = 0;
    for (auto &it_fast: laser_info_) {  // not necessarily ordered by assigned seq
      for (auto &it_slow: laser_info_) {
        if (it_fast.ts < it_slow.ts + min_time_excl)
          break;
        double dist = (it_fast.sens_pose.translation() - it_slow.sens_pose.translation()).norm();
        if (dist < 4.0) {
          it_fast.has_gt_positive_lc = true;
          cnt_gt_lc++;
        }
      }
    }
    printf("Found %d laser with gt loops.\n", cnt_gt_lc);

  }

  bool loadNewScan() {
    p_lidar_curr++;
    // Load the scan into the cache so that it can be retrieved by calling related functions
    CHECK(p_lidar_curr >= 0);
    if (p_lidar_curr >= laser_info_.size()) {
      printf("\n===\ncurrent addr %d exceeds boundary\n", p_lidar_curr);
      return false;
    }

    printf("\n===\nloaded scan addr %d, seq: %d, fpath: %s\n", p_lidar_curr, laser_info_[p_lidar_curr].seq,
           laser_info_[p_lidar_curr].fpath.c_str());
    return true;
  }

  // 1. loader
  const LaserScanInfo &getCurrScanInfo() const {
    CHECK(p_lidar_curr < laser_info_.size());
    CHECK(p_lidar_curr >= 0);

    return laser_info_[p_lidar_curr];
  }

  std::shared_ptr<ContourManager> getCurrContourManager(const ContourManagerConfig &config) const {
    // assumption:
    // 1. The returned cont_mng is matched to the data in `laser_info_`
    // 2. The int index of every cont_mng is matched to the index of every item of `laser_info`
    //   (so that we can use cont_mng as input to index gt 3d poses in the "recorder" below)

    std::shared_ptr<ContourManager> cmng_ptr(new ContourManager(config, laser_info_[p_lidar_curr].seq));
    pcl::PointCloud<pcl::PointXYZ>::ConstPtr out_ptr = readKITTIPointCloudBin<pcl::PointXYZ>(
        laser_info_[p_lidar_curr].fpath);

    std::string str_id = std::to_string(laser_info_[p_lidar_curr].seq);
    str_id = "assigned_id_" + std::string(8 - str_id.length(), '0') + str_id;

    cmng_ptr->makeBEV<pcl::PointXYZ>(out_ptr, str_id);
    cmng_ptr->makeContoursRecurs();

    return cmng_ptr;
  }

  // 2. recorder.
  PredictionOutcome
  addPrediction(const std::shared_ptr<const ContourManager> &q_mng,
                const std::shared_ptr<const ContourManager> &cand_mng = nullptr,
                const Eigen::Isometry2d &T_est_delta_2d = Eigen::Isometry2d::Identity()) {
    int id_tgt = q_mng->getIntID();  // q: src, cand: tgt
    int addr_tgt = lookupNN<int>(id_tgt, assigned_seqs_, 0);
    CHECK_GE(addr_tgt, 0);

    PredictionOutcome curr_res;
    curr_res.id_tgt = id_tgt;

    if (cand_mng) {
      // The prediction is positive
      int id_src = cand_mng->getIntID();
      int addr_src = lookupNN<int>(id_src, assigned_seqs_, 0);
      CHECK_GE(addr_src, 0);

      curr_res.id_src = id_src;

      const auto gen_bev_config = q_mng->getConfig();  // the config used to generate BEV
      Eigen::Isometry2d tf_err = ConstellCorrelation::evalMetricEst(T_est_delta_2d, laser_info_[addr_src].sens_pose,
                                                                    laser_info_[addr_tgt].sens_pose, gen_bev_config);
      double est_trans_norm2d = ConstellCorrelation::getEstSensTF(T_est_delta_2d, gen_bev_config).translation().norm();
      double gt_trans_norm3d = (laser_info_[addr_src].sens_pose.translation() -
                                laser_info_[addr_tgt].sens_pose.translation()).norm();
      printf(" Dist: Est2d: %.2f; GT3d: %.2f\n", est_trans_norm2d, gt_trans_norm3d);

      double err_vec[3] = {tf_err.translation().x(), tf_err.translation().y(), std::atan2(tf_err(1, 0), tf_err(0, 0))};
      printf(" Error: dx=%f, dy=%f, dtheta=%f\n", err_vec[0], err_vec[1], err_vec[2]);

      memcpy(curr_res.est_err, err_vec, sizeof(err_vec));
      if (laser_info_[addr_tgt].has_gt_positive_lc && gt_trans_norm3d < 4.0) {  // TP
        curr_res.tfpn = PredictionOutcome::TP;

        tp_trans_rmse.addOneErr(err_vec);
        tp_rot_rmse.addOneErr(err_vec + 2);
      } else {  // FP
        curr_res.tfpn = PredictionOutcome::FP;
      }

      all_trans_rmse.addOneErr(err_vec);
      all_rot_rmse.addOneErr(err_vec + 2);

    } else {
      // The prediction is negative
      if (laser_info_[addr_tgt].has_gt_positive_lc)  // FN
        curr_res.tfpn = PredictionOutcome::FN;
      else  // TN
        curr_res.tfpn = PredictionOutcome::TN;
    }

    pred_records.emplace_back(curr_res);
    return curr_res;
  }


  // 3. Result saver
  void savePredictionResults(const std::string &sav_path) const {
    std::fstream res_file(sav_path, std::ios::out);

    if (res_file.rdstate() != std::ifstream::goodbit) {
      std::cerr << "Error opening " << sav_path << std::endl;
      return;
    }

    // tgt before src
    for (const auto &rec: pred_records) {
      int addr_tgt = lookupNN<int>(rec.id_tgt, assigned_seqs_, 0);
      CHECK_GE(addr_tgt, 0);

      res_file << rec.tfpn << "\t";

      std::string str_rep_tgt = laser_info_[addr_tgt].fpath, str_rep_src;

      if (rec.id_src < 0) {
        res_file << rec.id_tgt << "-x" << "\t";
        str_rep_src = "x";
      } else {
        int addr_src = lookupNN<int>(rec.id_src, assigned_seqs_, 0);
        CHECK_GE(addr_src, 0);

        res_file << rec.id_tgt << "-" << rec.id_src << "\t";
        str_rep_src = laser_info_[addr_src].fpath;
      }

      res_file << rec.est_err[0] << "\t" << rec.est_err[1] << "\t" << rec.est_err[2] << "\t";

//      // case 1: path
//      res_file << str_rep_tgt << "\t" << str_rep_src << "\n"; // may be too long

      // case 2: shortened
      int str_max_len = 32;
      int beg_tgt = std::max(0, (int) str_rep_tgt.length() - str_max_len);
      int beg_src = std::max(0, (int) str_rep_src.length() - str_max_len);
      res_file << str_rep_tgt.substr(beg_tgt, str_rep_tgt.length() - beg_tgt) << "\t"
               << str_rep_src.substr(beg_src, str_rep_src.length() - beg_src) << "\n";

    }
    // rmse and mean error can be calculated from this file. So we will not record it.


    printf("In outcome file:\n");
    printf("TP is %d\n", static_cast<std::underlying_type<PredictionOutcome::Res>::type>(PredictionOutcome::TP));
    printf("FP is %d\n", static_cast<std::underlying_type<PredictionOutcome::Res>::type>(PredictionOutcome::FP));
    printf("TN is %d\n", static_cast<std::underlying_type<PredictionOutcome::Res>::type>(PredictionOutcome::TN));
    printf("FN is %d\n", static_cast<std::underlying_type<PredictionOutcome::Res>::type>(PredictionOutcome::FN));


    res_file.close();
    printf("Outcome saved successfully.\n");


  }

  inline double getTPMeanTrans() const { return tp_trans_rmse.getMean(); }

  inline double getTPMeanRot() const { return tp_rot_rmse.getMean(); }

  inline double getTPRMSETrans() const { return tp_trans_rmse.getRMSE(); }

  inline double getTPRMSERot() const { return tp_rot_rmse.getRMSE(); }

  // 4. related public util
  static void loadCheckThres(const std::string &fpath, CandidateScoreEnsemble &thres_lb,
                             CandidateScoreEnsemble &thres_ub);


};

#endif //CONT2_EVALUATOR_ROS_H
