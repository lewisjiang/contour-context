#include<iostream>
#include<string>
#include <ros/ros.h>
#include "cont2/contour_mng.h"
#include "cont2_ros/io_ros.h"
#include "cont2/contour_db.h"
#include "eval/evaluator.h"
#include "cont2_ros/spinner_ros.h"

#include <nav_msgs/Path.h>
#include <visualization_msgs/MarkerArray.h>

#include <geometry_msgs/PoseStamped.h>
#include "tf2/transform_datatypes.h"
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/convert.h>
#include <tf2_eigen/tf2_eigen.h>

const std::string PROJ_DIR = std::string(PJSRCDIR);


class RosBagPlayLoopTest : public BaseROSSpinner {

  // ros stuff
//  ros::NodeHandle nh;
//  ros::Publisher pub_path;
//  ros::Publisher pub_lc_connections;
//  ros::Publisher pub_pose_idx;
//  nav_msgs::Path path_msg;
//  visualization_msgs::MarkerArray line_array;
//  visualization_msgs::MarkerArray idx_text_array;
//  tf2_ros::Buffer tfBuffer;
//  tf2_ros::TransformListener tfListener;
//
//  // The data used for general purpose (w/o gt files, etc.) in the work flow
//  std::map<int, GlobalPoseInfo> g_poses;

  // --- Added members for evaluation and LC module running ---
  //  std::vector<std::shared_ptr<ContourManager>> scans;
  ContourDB contour_db;

  // The raw data read from extra content of the dataset
  std::vector<double> all_raw_time_stamps;
  std::vector<Eigen::Isometry3d> all_raw_gt_l_poses;

  int seq_beg;
  int seq_end;
  Eigen::Isometry3d T_lc_velod_;    // points in velodyne to left camera, Tr

  Cont2_ROS_IO<pcl::PointXYZ> ros_io;  // data (e.g. point cloud) loader to work with ros.

  CandidateScoreEnsemble thres_lb_, thres_ub_;  // check thresholds

  int lc_tp_pose_cnt{}, lc_fn_pose_cnt{}, lc_fp_pose_cnt{};
  SimpleRMSE<2> trans_rmse;
  SimpleRMSE<1> rot_rmse;


  bool time_beg_set = false;
  ros::Time time_beg;


public:
  explicit RosBagPlayLoopTest(ros::NodeHandle &nh_, int idx_beg, int idx_end,
                              uint64_t t_seq_beg_ns = 12345) : BaseROSSpinner(nh_), //nh(nh_),
                                                               seq_beg(idx_beg), seq_end(idx_end),
                                                               ros_io(0, "/velodyne_points", nh_, t_seq_beg_ns),
      // tfListener(tfBuffer),
                                                               contour_db(ContourDBConfig(),
                                                                          std::vector<int>({1, 2, 3})) {

    T_lc_velod_.setIdentity();
  }

  // call before spin 1/3
  void loadRawTs(const std::string &f_raw_ts) {
    std::fstream ts_file;
    ts_file.open(f_raw_ts, std::ios::in);
    if (ts_file.rdstate() != std::ifstream::goodbit) {
      std::cout << "Cannot open ts" << f_raw_ts << ", returning..." << std::endl;
      return;
    }
    std::string strbuf;
    while (std::getline(ts_file, strbuf)) {
      std::istringstream iss(strbuf);
      std::string ts;
      if (iss >> ts) {
        all_raw_time_stamps.push_back(std::stod(ts));
      }
    }
    printf("%lu timestamps loaded.\n", all_raw_time_stamps.size());
  }

  // call before spin 2/3
  // TODO: load gt and use it in
  void loadSeqGT(const std::string &f_seq_calib, const std::string &f_seq_gt_pose) {
    // 1. calibration
    std::fstream calib_file;
    calib_file.open(f_seq_calib, std::ios::in);
    if (calib_file.rdstate() != std::ifstream::goodbit) {
      std::cout << "Cannot open calibration" << f_seq_calib << ", returning..." << std::endl;
      return;
    }
    bool calib_set = false;
    Eigen::Quaterniond calib_rot_q;
    Eigen::Vector3d calib_trans;
    std::string strbuf;
    while (std::getline(calib_file, strbuf)) {
      std::istringstream iss(strbuf);
      std::string pname;
      if (iss >> pname) {
        if (pname == "Tr:") {
          Eigen::Matrix3d rot_mat;
          for (int i = 0; i < 12; i++) {
            if (i % 4 == 3)
              iss >> calib_trans(i / 4);
            else
              iss >> rot_mat(i / 4, i % 4);
          }
          calib_rot_q = Eigen::Quaterniond(rot_mat);
          calib_set = true;
          break;
        }
      }
    }
    CHECK(calib_set);
    T_lc_velod_.setIdentity();
    T_lc_velod_.rotate(calib_rot_q);
    T_lc_velod_.pretranslate(calib_trans);

    // 2. gt left camera pose
    std::fstream gt_pose_file;
    gt_pose_file.open(f_seq_gt_pose, std::ios::in);
    if (gt_pose_file.rdstate() != std::ifstream::goodbit) {
      std::cout << "Cannot open gt pose" << f_seq_gt_pose << ", returning..." << std::endl;
      return;
    }

    Eigen::Isometry3d T_w_lc0;
    T_w_lc0.setIdentity();

    // moving along z to moving along y (The difference between init left cam pose (lc0) and the fixed z-up world frame)
    T_w_lc0.rotate(Eigen::AngleAxisd(-M_PI / 2, Eigen::Vector3d(1, 0, 0)));  // TODO: for vis

    while (std::getline(gt_pose_file, strbuf)) {
      std::istringstream iss(strbuf);
      Eigen::Isometry3d tmp_T_lc0_lc;  // raw gt poses are T_lc0_lc
      Eigen::Vector3d tmp_trans;
      Eigen::Matrix3d tmp_rot_mat;
      Eigen::Quaterniond tmp_rot_q;

      for (int i = 0; i < 12; i++) {
        if (i % 4 == 3)
          iss >> tmp_trans(i / 4);
        else
          iss >> tmp_rot_mat(i / 4, i % 4);
      }

      tmp_rot_q = Eigen::Quaterniond(tmp_rot_mat);
      tmp_T_lc0_lc.setIdentity();
      tmp_T_lc0_lc.rotate(tmp_rot_q);
      tmp_T_lc0_lc.pretranslate(tmp_trans);

//      all_raw_gt_l_poses.emplace_back(T_w_lc0 * tmp_T_lc0_lc * T_lc_velod_);
      // The first lidar pose is the world frame:
      all_raw_gt_l_poses.emplace_back(T_lc_velod_.inverse() * tmp_T_lc0_lc * T_lc_velod_);
//      all_raw_gt_l_poses.emplace_back(tmp_T_lc0_lc);

    }

    printf("%lu gt poses loaded.\n", all_raw_gt_l_poses.size());
  }

  void setCheckThres(const CandidateScoreEnsemble &lb, const CandidateScoreEnsemble &ub) {
    thres_lb_ = lb;
    thres_ub_ = ub;
  }

  // call before spin 3/3
  // load check thres
  void loadCheckThres(const std::string &fpath) {
    ContLCDEvaluator::loadCheckThres(fpath, thres_lb_, thres_ub_);
  }

  // spin
  void processOnce(int &cnt) {

    pcl::PointCloud<pcl::PointXYZ>::ConstPtr out_ptr = nullptr;
    geometry_msgs::TransformStamped tf_gt_curr;
//    out_ptr = ros_io.getLidarPointCloud();
    out_ptr = ros_io.getLidarPointCloud(tf_gt_curr);
    if (!out_ptr)
      return;

    ros::Time time;
    time.fromNSec(out_ptr->header.stamp);
    if (!time_beg_set) {
      time_beg = time;
      time_beg_set = true;
    }

    int scan_seq = lookupNN<double>(time.toSec(), all_raw_time_stamps, 1e-3);
    CHECK_GT(scan_seq, -1);
    if (scan_seq < seq_beg || scan_seq > seq_end) {
      printf("Index %d out of range [%d, %d].\n", scan_seq, seq_beg, seq_end);
      return;
    }

//    // case 1: use count as id
//    int curr_id = cnt;
    // case 2: use dataset frame seq id
    int curr_id = scan_seq;

    // case 1: use gps-imu data as pose (comment the line below)
    tf_gt_curr = tf2::eigenToTransform(all_raw_gt_l_poses[scan_seq - seq_beg]); // case 2: use dataset sequence gt

    printf("---\nour counted seq: %d, kitti seq: %d, curr_id: %d, stamp: %lu\n", cnt, scan_seq, curr_id, time.toNSec());

    broadcastCurrPose(tf_gt_curr);

    std::cout << "trans z: " << tf_gt_curr.transform.translation.z << std::endl;

    Eigen::Isometry3d T_gt_curr; // velodyne to world transform
    T_gt_curr = tf2::transformToEigen(tf_gt_curr); // use looked up tf

    Eigen::Vector3d time_translate(0, 0, 1);
    time_translate = time_translate * (time.toSec() - time_beg.toSec()) / 60;   // elevate 1m per minute

    g_poses.insert(std::make_pair(curr_id, GlobalPoseInfo(T_gt_curr, time_translate.z())));
//    gt_poses.emplace_back(T_gt_curr);
//    poses_time_z_shift.emplace_back(time_translate.z());

    tf_gt_curr.transform.translation.z += time_translate.z();// elevate 1m per minute
    publishPath(time, tf_gt_curr);
    publishScanSeqText(time, tf_gt_curr, scan_seq);


    std::vector<std::pair<int, int>> new_lc_pairs;

    ContourManagerConfig config;
    config.lv_grads_ = {1.5, 2, 2.5, 3, 3.5, 4};
    std::shared_ptr<ContourManager> cmng_ptr(new ContourManager(config, curr_id));
    cmng_ptr->makeBEV<pcl::PointXYZ>(out_ptr);
//      cmng_ptr.makeContours();
    cmng_ptr->makeContoursRecurs();


    // save images of layers
    TicToc clk;
    for (int i = 0; i < config.lv_grads_.size(); i++) {
      std::string f_name = PROJ_DIR + "/results/layer_img/contour_" + "lv" + std::to_string(i) + "_" +
                           std::to_string(out_ptr->header.stamp) + ".png";
      cmng_ptr->saveContourImage(f_name, i);
    }
    std::cout << "Time save layers: " << clk.toctic() << std::endl;

    cmng_ptr->clearImage();

    // 2.1.2 query case 2:
    std::vector<std::shared_ptr<const ContourManager>> candidate_loop;
    std::vector<double> cand_corr;
    std::vector<Eigen::Isometry2d> bev_tfs;
    clk.tic();

//    contour_db.queryRangedKNN(cmng_ptr, candidate_loop, cand_corr, bev_tfs);
    contour_db.queryRangedKNN(cmng_ptr, thres_lb_, thres_ub_, candidate_loop, cand_corr, bev_tfs);
    printf("%lu Candidates in %7.5fs: \n", candidate_loop.size(), clk.toc());

    bool has_tp_lc = false;
    bool has_fp_lc = false;

    CHECK(candidate_loop.size() < 2); // TODO: at most one candidate
    for (int j = 0; j < candidate_loop.size(); j++) {
      printf("Matching new: %d with old: %d:", curr_id, candidate_loop[j]->getIntID());
      new_lc_pairs.emplace_back(curr_id, candidate_loop[j]->getIntID());

      // record "valid" loop closure
      auto tf_err = ConstellCorrelation::evalMetricEst(bev_tfs[j], g_poses[new_lc_pairs.back().second].T_wl,
                                                       g_poses[new_lc_pairs.back().first].T_wl, config);
      double est_trans_norm2d = ConstellCorrelation::getEstSensTF(bev_tfs[j], config).translation().norm();
      double gt_trans_norm3d = (g_poses[new_lc_pairs.back().first].T_wl.translation() -
                                g_poses[new_lc_pairs.back().second].T_wl.translation()).norm();
      printf(" Dist: Est2d: %.2f; GT3d: %.2f\n", est_trans_norm2d, gt_trans_norm3d);

      // metric error benchmark
      double err_vec[3] = {tf_err.translation().x(), tf_err.translation().y(), std::atan2(tf_err(1, 0), tf_err(0, 0))};
      printf(" Error: dx=%f, dy=%f, dtheta=%f\n", err_vec[0], err_vec[1], err_vec[2]);
      trans_rmse.addOneErr(err_vec);
      rot_rmse.addOneErr(err_vec + 2);
      printf(" Error mean: t:%7.4f, r:%7.4f of %d lc\n", trans_rmse.getMean(), rot_rmse.getMean(), trans_rmse.cnt_sqs);
      printf(" Error rmse: t:%7.4f, r:%7.4f\n", trans_rmse.getRMSE(), rot_rmse.getRMSE());

      if (gt_trans_norm3d < 4.0) {
//           gt_poses[new_lc_pairs.back().second].translation()).norm() < 4.0 || tf_err.translation().norm() < 2.0) {
        // TODO: this judgement is based on the assumption that only one candidate is returned.
        has_tp_lc = true;
      } else {
        if (est_trans_norm2d < 4.0)
          has_fp_lc = true;
      }

      // write file
      std::string f_name =
          PROJ_DIR + "/results/match_comp_img/lc_" + cmng_ptr->getStrID() + "-" + candidate_loop[j]->getStrID() +
          ".png";
      ContourManager::saveMatchedPairImg(f_name, *cmng_ptr, *candidate_loop[j]);
      printf("Image saved: %s-%s\n", cmng_ptr->getStrID().c_str(), candidate_loop[j]->getStrID().c_str());
    }

    if (has_tp_lc)
      lc_tp_pose_cnt++;
    else if (has_fp_lc)
      lc_fp_pose_cnt++;
    else {
      bool has_fn = false;
      for (int i = 0; i < curr_id - 150; i++) {
        double dist = (g_poses[i].T_wl.translation() - g_poses[curr_id].T_wl.translation()).norm();
        if (dist < 4.0) {
          printf("False Negative: %d-%d, %f\n", i, cnt, dist);
          has_fn = true;
        }
      }
      if (has_fn)
        lc_fn_pose_cnt++;
    }

    // 2.2 add new
    contour_db.addScan(cmng_ptr, time.toSec());
    // 2.3 balance
    clk.tic();
    contour_db.pushAndBalance(curr_id, time.toSec());
    printf("Rebalance tree cost: %7.5f\n", clk.toc());

    // plot
    publishLCConnections(new_lc_pairs, time);
    cnt++;

    printf("Accumulated tp poses: %d\n", lc_tp_pose_cnt);
    printf("Accumulated fn poses: %d\n", lc_fn_pose_cnt);
    printf("Accumulated fp poses: %d\n", lc_fp_pose_cnt);

  }

};

int main(int argc, char **argv) {
  ros::init(argc, argv, "rosbag_play_test");
  ros::NodeHandle nh;

  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);

  printf("tag 0\n");

  // Check thres path
  std::string cand_score_config = "/home/lewis/catkin_ws2/src/contour-context/config/score_thres_kitti_bag_play.cfg";

  // KITTI seq 05:
  uint64_t t_seq_beg = 100000;
  int p1 = 0, p2 = 2760;
  std::string ts_path = "/home/lewis/catkin_ws2/src/contour-context/results/kitti_seq05_seconds.txt";
  std::string gt_pose_path = "/home/lewis/Downloads/KITTI_data/dataset_gt/poses/05.txt";
  std::string seq_calib_path = "/home/lewis/Downloads/KITTI_data/dataset/sequences/05/calib.txt";

  // KITTI seq 00:
//  uint64_t t_seq_beg = 100000; // any small ts
//  int p1 = 0, p2 = 4540;
//  std::string ts_path = "/home/lewis/catkin_ws2/src/contour-context/results/kitti_seq00_seconds.txt";
//  std::string gt_pose_path = "/home/lewis/Downloads/KITTI_data/dataset_gt/poses/00.txt";
//  std::string seq_calib_path = "/home/lewis/Downloads/KITTI_data/dataset/sequences/00/calib.txt";

  RosBagPlayLoopTest o(nh, p1, p2, t_seq_beg);
  o.loadRawTs(ts_path);
  o.loadSeqGT(seq_calib_path, gt_pose_path);

  // case 1: load config file
  o.loadCheckThres(cand_score_config);

//  // case 2: manually set
//  CandidateScoreEnsemble thres_lb, thres_ub;
//  // a.1 constellation similarity
//  thres_lb.sim_constell.i_ovlp_sum = 5;
//  thres_ub.sim_constell.i_ovlp_sum = 10;
//
//  thres_lb.sim_constell.i_ovlp_max_one = 4;
//  thres_ub.sim_constell.i_ovlp_max_one = 6;
//
//  thres_lb.sim_constell.i_in_ang_rng = 4;
//  thres_ub.sim_constell.i_in_ang_rng = 6;
//
//  // a.2 pairwise similarity
//  thres_lb.sim_pair.i_indiv_sim = 4;
//  thres_ub.sim_pair.i_indiv_sim = 6;
//
//  thres_lb.sim_pair.i_orie_sim = 4;
//  thres_ub.sim_pair.i_orie_sim = 6;
//
//  thres_lb.sim_pair.f_area_perc = 5; // 0.05;
//  thres_ub.sim_pair.f_area_perc = 10; // 0.15;
//
//  // a.3 correlation
//  thres_lb.correlation = 0.65;
//  thres_ub.correlation = 0.75;
//
//  thres_lb.area_perc = 0.05;
//  thres_ub.area_perc = 0.10;
//  o.setCheckThres(thres_lb, thres_ub);

  ros::Rate rate(50);
  int cnt = 0;

  while (ros::ok()) {
    ros::spinOnce();

    o.processOnce(cnt);

    rate.sleep();
  }

  return 0;
}
