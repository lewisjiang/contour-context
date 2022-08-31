//
// Created by lewis on 5/30/22.
//
#include<iostream>
#include<string>
#include <utility>
#include <ros/ros.h>
#include "cont2/contour_mng.h"
#include "cont2/io_bin.h"
#include "cont2/contour_db.h"
#include "cont2/correlation.h"

#include <nav_msgs/Path.h>
#include <visualization_msgs/MarkerArray.h>

#include <geometry_msgs/PoseStamped.h>
#include "tf2/transform_datatypes.h"
//#include <tf2_ros/transform_broadcaster.h>
//#include <tf2_ros/transform_listener.h>
#include <tf2/convert.h>
#include <tf2_eigen/tf2_eigen.h>

const std::string PROJ_DIR = std::string(PJSRCDIR);

class KittiBinDataVis {
  ros::NodeHandle nh;
  ros::Publisher pub_path;
  ros::Publisher pub_index;
  nav_msgs::Path path_msg;
  visualization_msgs::MarkerArray idx_text_array;

  std::vector<std::pair<int, Eigen::Isometry3d>> gt_poses;

public:
  explicit KittiBinDataVis(ros::NodeHandle &nh_, std::vector<std::pair<int, Eigen::Isometry3d>> gt_poses_)
      : nh(nh_), gt_poses(std::move(gt_poses_)) {
    ros::Time t_curr = ros::Time::now();
    path_msg.header.frame_id = "world";
    path_msg.header.stamp = t_curr;

    pub_path = nh_.advertise<nav_msgs::Path>("/gt_path", 10000);
    pub_index = nh_.advertise<visualization_msgs::MarkerArray>("/pose_index", 10000);

    double time_shift = 1.0 / 60;  // shift per sec
    for (int i = 0; i < gt_poses.size(); i++) {
      // pose
      geometry_msgs::PoseStamped ps;
      ps.header = path_msg.header;
      auto q_rot = Eigen::Quaterniond(gt_poses[i].second.rotation());
      auto trans = Eigen::Vector3d(gt_poses[i].second.translation());
      ps.pose.orientation.x = q_rot.x();
      ps.pose.orientation.y = q_rot.y();
      ps.pose.orientation.z = q_rot.z();
      ps.pose.orientation.w = q_rot.w();
      ps.pose.position.x = trans.x();
      ps.pose.position.y = trans.y();
      ps.pose.position.z = trans.z() + time_shift * i / 10;

      path_msg.poses.emplace_back(ps);

      // index
      visualization_msgs::Marker marker;
      marker.header.frame_id = "world";
      marker.header.stamp = t_curr;
      marker.ns = "myns";
      marker.id = i;

      marker.action = visualization_msgs::Marker::ADD;
      marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;

      marker.text = std::to_string(gt_poses[i].first);
      marker.scale.z = 0.25;
      marker.lifetime = ros::Duration();

      marker.pose = ps.pose;

      marker.color.a = 1.0; // Don't forget to set the alpha!
      marker.color.r = 0.0f;
      marker.color.g = 1.0f;
      marker.color.b = 1.0f;

      idx_text_array.markers.emplace_back(marker);
    }

    pub_index.publish(idx_text_array);
    pub_path.publish(path_msg);

  }

  void dummyLoopOnce() {
    ros::Time t_curr = ros::Time::now();
    path_msg.header.stamp = t_curr;

    pub_path.publish(path_msg);
    pub_index.publish(idx_text_array);

    printf("spin %f\n", t_curr.toSec());
  }
};

int main(int argc, char **argv) {
  ros::init(argc, argv, "read_kitti_bin_test");
  ros::NodeHandle nh;

  printf("read bin start 0\n");

  // kitti 05
//  std::string kitti_raw_dir = "/home/lewis/Downloads/datasets/kitti_raw", date = "2011_09_30", seq = "2011_09_30_drive_0018_sync";

  // kitti 00
  std::string kitti_raw_dir = "/home/lewis/Downloads/datasets/kitti_raw", date = "2011_10_03", seq = "2011_10_03_drive_0027_sync";

  // case raw(1/3): read from raw data
  //  ReadKITTILiDAR reader(kitti_raw_dir, date, seq);

  // case odom(1/3): use manually appointed data

//  // visualize gt poses and index
//  KittiBinDataVis data_test(nh, reader.getGNSSImuPoses());
//  ros::Rate rate(1);
//  while (ros::ok()) {
//    ros::spinOnce();
//    data_test.dummyLoopOnce();
//    rate.sleep();
//  }

  // analysis data
//  int idx_old = 34, idx_new = 2437;
//  int idx_old = 119, idx_new = 2511;
//  int idx_old = 1561, idx_new = 2576;
//  int idx_old = 805, idx_new = 2576;
//  int idx_old = 80, idx_new = 2481; // the return to the first turning

//  int idx_old = 558, idx_new = 1316;
//  int idx_old = 769, idx_new = 1512;

//  int idx_old = 895, idx_new = 2632;  // final straight road loop
//  int idx_old = 905, idx_new = 2636;  // final straight road loop
//  int idx_old = 890, idx_new = 2632;  // final straight road loop

//  int idx_old = 806, idx_new = 1562;  // seek farther. 1562:793, 1563:816, nearest old: 808

  // sequence 00
//  int idx_old = 486, idx_new = 1333;  //

  // kitti 08, in odom seq, not raw
  int idx_old = 237, idx_new = 1648;  //


  std::string s_old, s_new;

//  // case raw(2/3):
//  pcl::PointCloud<pcl::PointXYZ>::ConstPtr out_ptr_old = reader.getLidarPointCloud<pcl::PointXYZ>(idx_old, s_old);
//  pcl::PointCloud<pcl::PointXYZ>::ConstPtr out_ptr_new = reader.getLidarPointCloud<pcl::PointXYZ>(idx_new, s_new);

  // case odom(2/3):
  s_old = std::string(6 - std::to_string(idx_old).length(), '0') + std::to_string(idx_old);
  s_new = std::string(6 - std::to_string(idx_new).length(), '0') + std::to_string(idx_new);
  std::string path_bin_old = PROJ_DIR + "/sample_data/" + s_old + ".bin";
  std::string path_bin_new = PROJ_DIR + "/sample_data/" + s_new + ".bin";
  pcl::PointCloud<pcl::PointXYZ>::ConstPtr out_ptr_old = readKITTIPointCloudBin<pcl::PointXYZ>(path_bin_old);
  pcl::PointCloud<pcl::PointXYZ>::ConstPtr out_ptr_new = readKITTIPointCloudBin<pcl::PointXYZ>(path_bin_new);

  ContourManagerConfig config;
  config.lv_grads_ = {1.5, 2, 2.5, 3, 3.5, 4};
//  config.reso_row_ = 1.0f;
//  config.reso_col_ = 1.0f;
//  config.n_row_ = 150;
//  config.n_col_ = 150;

  std::shared_ptr<ContourManager> cmng_ptr_old(new ContourManager(config, idx_old));
  cmng_ptr_old->makeBEV<pcl::PointXYZ>(out_ptr_old, s_old);
  cmng_ptr_old->makeContoursRecurs();
  std::shared_ptr<ContourManager> cmng_ptr_new(new ContourManager(config, idx_new));
  cmng_ptr_new->makeBEV<pcl::PointXYZ>(out_ptr_new, s_new);
  cmng_ptr_new->makeContoursRecurs();

  printf("Analysing %s-%s\n", cmng_ptr_old->getStrID().c_str(), cmng_ptr_new->getStrID().c_str());

  // test 1. manual compare
  // Given a pair of scans, find the best possible LC param, via all possible matching combinations.
  // That is to say, traverse over all possible pairs of keys, and find the best possible LC.
  // The workflow should be similar to `queryRangedKNN()`'s checking
  Eigen::Isometry2d T_init;
  T_init.setIdentity();

  // init similarity thres params
  CandidateScoreEnsemble thres_lb, thres_ub;
  // a.1 constellation similarity
  thres_lb.sim_constell.i_ovlp_sum = 5;
  thres_ub.sim_constell.i_ovlp_sum = 10;

  thres_lb.sim_constell.i_ovlp_max_one = 4;
  thres_ub.sim_constell.i_ovlp_max_one = 6;

  thres_lb.sim_constell.i_in_ang_rng = 4;
  thres_ub.sim_constell.i_in_ang_rng = 6;

  // a.2 pairwise similarity
  thres_lb.sim_pair.i_indiv_sim = 4;
  thres_ub.sim_pair.i_indiv_sim = 6;

  thres_lb.sim_pair.i_orie_sim = 4;
  thres_ub.sim_pair.i_orie_sim = 6;

//  thres_lb.sim_pair.f_area_perc = 5; // 0.05;
//  thres_ub.sim_pair.f_area_perc = 15; // 0.15;

  // a.3 correlation
//  thres_lb.correlation = 0.65;
//  thres_ub.correlation = 0.75;
//
//  thres_lb.area_perc = 0.05;
//  thres_ub.area_perc = 0.10;

  thres_lb.sim_post.correlation = 0.60;
  thres_ub.sim_post.correlation = 0.65;

  thres_lb.sim_post.area_perc = 0.05;
  thres_ub.sim_post.area_perc = 0.10;

  thres_lb.sim_post.neg_est_dist = -8.0;
  thres_ub.sim_post.neg_est_dist = -4.5;


  CandidateManager cand_mng(cmng_ptr_new, thres_lb, thres_ub);

  printf("Keys:\n"); // as if adding retrieved keys (search results) in `queryRangedKNN()`.
  for (int ll = 0; ll < config.lv_grads_.size(); ll++) {
    printf("\n---\nPermu Level %d\n", ll);
    auto keys1 = cmng_ptr_old->getLevRetrievalKey(ll);
    auto keys2 = cmng_ptr_new->getLevRetrievalKey(ll);

    // test all possible key pairs
    for (int i1 = 0; i1 < keys1.size(); i1++) {
      for (int i2 = 0; i2 < keys2.size(); i2++) {
        const auto &k1 = keys1[i1];
        const auto &k2 = keys2[i2];

        // basic key checks:
        if (k1.sum() == 0 || k2.sum() == 0)
          continue;
        KeyFloatType tmp_dist = (k1 - k2).squaredNorm();
        if (tmp_dist > 1000.0f)
          continue;

        printf("BF key diff: %d %d %10.4f, ", i1, i2, tmp_dist);

        // check sim step by step:

        ConstellationPair piv_pair(ll, i1, i2);
        CandidateScoreEnsemble ret_score = cand_mng.checkCandWithHint(cmng_ptr_old, piv_pair);

        printf("Each check: ");
        ret_score.sim_constell.print();
        ret_score.sim_pair.print();
        printf("\n");
//        printf("%6f\n", ret_score.correlation);  // of course 0, since we have not calc correlation yet.

      }
    }
  }

//  // manual constellation case 1: combine 2 matches from different anchors
//  std::vector<ConstellationPair> cstl; // for '806', '1562'
//  cstl.emplace_back(1, 0, 0);
//  cstl.emplace_back(1, 3, 1);
//  cstl.emplace_back(2, 6, 4);
//  cstl.emplace_back(2, 9, 8);
//  cstl.emplace_back(3, 2, 1);
//  cstl.emplace_back(4, 0, 0);
//  cstl.emplace_back(4, 8, 7);
//  // Add some "invalid pairs"
//  cstl.emplace_back(1, 8, 9);  // Com radius not pass.
//  cstl.emplace_back(1, 4, 8);  // Cell cnt not pass.
//  cstl.emplace_back(2, 3, 1);  // Cell cnt not pass.
////  cstl.emplace_back(1, 4, 8);  //
////  cstl.emplace_back(1, 4, 8);  //
//
//  T_init = ContourManager::getTFFromConstell(*cmng_ptr_old, *cmng_ptr_new, cstl.begin(), cstl.end());

  printf("\n===\nPolling over candidate keys finished.\n\n");

  cand_mng.tidyUpCandidates();

  const int max_fine_opt = 5;
  std::vector<std::shared_ptr<const ContourManager>> res_cand_ptr;
  std::vector<double> res_corr;
  std::vector<Eigen::Isometry2d> res_T;

  int num_best_cands = cand_mng.fineOptimize(max_fine_opt, res_cand_ptr, res_corr, res_T);
  if (!res_T.empty()) {
    printf("Pair prediction: Positive.\n");
    T_init = res_T[0];  // actually the fine optimized
  }


  std::string f_name =
      PROJ_DIR + "/results/pair_comp_img/pair_" + cmng_ptr_old->getStrID() + "-" + cmng_ptr_new->getStrID() +
      ".png";
  ContourManager::saveMatchedPairImg(f_name, *cmng_ptr_old, *cmng_ptr_new);


  // test 4. calculate GMM L2 optimization using ceres
  GMMOptConfig gmm_config;

// For int idx_old = 1561, idx_new = 2576;
//  Transform matrix:
//  0.0760232 -0.997106   142.958
//  0.997106 0.0760232  -4.46818
//  0         0         1

  if (T_init.matrix() == Eigen::Isometry2d::Identity().matrix()) {
    printf("\n===\nOverall, no valid T init produced from the whole matching.");
    return 0;
  }

  // optimize
//  // case 1: use another opt flow
//  ConstellCorrelation corr_est(gmm_config);
//  corr_est.initProblem(*cmng_ptr_old, *cmng_ptr_new, T_init);
//  std::pair<double, Eigen::Isometry2d> corr_final = corr_est.calcCorrelation();
//  std::cout << "Fine corr: " << corr_final.first << "\nT opt 2d: \n" << corr_final.second.matrix() << std::endl;

  // case 2: use the ensemble output:
  std::pair<double, Eigen::Isometry2d> corr_final;
  corr_final.first = res_corr[0];
  corr_final.second = res_T[0];

  // eval with gt:
  Eigen::Isometry3d gt_pose_old, gt_pose_new;  // sensor poses

//  // case raw(3/3)
//  const auto &gt_poses = reader.getGNSSImuPoses();
//  const auto &T_imu_lidar = reader.get_T_imu_velod();
//  for (const auto &itm: gt_poses) {
//    if (itm.first == idx_old)
//      gt_pose_old = itm.second * T_imu_lidar;
//    else if (itm.first == idx_new)
//      gt_pose_new = itm.second * T_imu_lidar;
//  }

  // case odom(3/3)
  Eigen::Matrix<double, 3, 4> m12_old, m12_new;
  m12_old
      << 0.038273, -0.997186, -0.064466, 17.912492, 0.998634, 0.035872, 0.037996, 187.439841, -0.035577, -0.065832, 0.997196, -5.703635;
  m12_new
      << -0.035202, 0.997522, -0.060911, 17.172279, -0.999374, -0.034909, 0.005879, 183.415044, 0.003738, 0.061079, 0.998126, -5.634574;
  gt_pose_old.setIdentity();
  gt_pose_old.rotate(Eigen::Quaterniond(m12_old.block<3, 3>(0, 0)));
  gt_pose_old.pretranslate(m12_old.block<3, 1>(0, 3));
  gt_pose_new.setIdentity();
  gt_pose_new.rotate(Eigen::Quaterniond(m12_new.block<3, 3>(0, 0)));
  gt_pose_new.pretranslate(m12_new.block<3, 1>(0, 3));


//  std::cout << gt_pose_old.matrix() << std::endl;
//  std::cout << gt_pose_new.matrix() << std::endl;

  std::cout << "T delta bev est 2d:\n" << T_init.matrix() << std::endl;  // the tf fed into python plot

  Eigen::Isometry2d T_est_sens_2d = ConstellCorrelation::getEstSensTF(corr_final.second, config);
  std::cout << "T delta sens est 2d:\n" << T_est_sens_2d.matrix() << std::endl;

  Eigen::Isometry2d T_err_2d = ConstellCorrelation::evalMetricEst(corr_final.second, gt_pose_old, gt_pose_new, config);
  std::cout << "Error 2d:\n" << T_err_2d.matrix() << std::endl;

  return 0;
}
