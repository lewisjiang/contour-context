#include<iostream>
#include<string>
#include <ros/ros.h>
#include "cont2/contour_mng.h"
#include "cont2_ros/io_ros.h"
#include "cont2/contour_db.h"

#include <nav_msgs/Path.h>
#include <visualization_msgs/MarkerArray.h>

#include <geometry_msgs/PoseStamped.h>
#include "tf2/transform_datatypes.h"
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/convert.h>
#include <tf2_eigen/tf2_eigen.h>

const std::string PROJ_DIR = std::string(PJSRCDIR);

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

  double getRMSE() const {
    return std::sqrt(sum_sqs / cnt_sqs);
  }

  double getMean() const {
    return sum_abs / cnt_sqs;
  }
};

class RosBagPlayLoopTest {
  // ros stuff
  ros::NodeHandle nh;
  ros::Publisher pub_path;
  ros::Publisher pub_lc_connections;
  ros::Publisher pub_pose_idx;
  nav_msgs::Path path_msg;
  visualization_msgs::MarkerArray line_array;
  visualization_msgs::MarkerArray idx_text_array;

  std::vector<std::shared_ptr<ContourManager>> scans;
  ContourDB contour_db;
  std::vector<Eigen::Isometry3d> gt_poses;
  std::vector<double> poses_time_z_shift;
  std::vector<double> all_raw_time_stamps;

  Cont2_ROS_IO<pcl::PointXYZ> ros_io;

  tf2_ros::Buffer tfBuffer;
  tf2_ros::TransformListener tfListener;
  int lc_cnt{}, seq_cnt{}, valid_lc_cnt{};
  int lc_pose_cnt{}, lc_fn_pose_cnt{};
  SimpleRMSE<2> trans_rmse;
  SimpleRMSE<1> rot_rmse;


  bool time_beg_set = false;
  ros::Time time_beg;


public:
  explicit RosBagPlayLoopTest(ros::NodeHandle &nh_) : nh(nh_), ros_io(0, "/velodyne_points", nh_, 1234567),
                                                      tfListener(tfBuffer),
                                                      contour_db(ContourDBConfig(), std::vector<int>({1, 2, 3})) {
    path_msg.header.frame_id = "world";
    pub_path = nh_.advertise<nav_msgs::Path>("/gt_path", 10000);
    pub_pose_idx = nh_.advertise<visualization_msgs::MarkerArray>("/pose_index", 10000);
    pub_lc_connections = nh_.advertise<visualization_msgs::MarkerArray>("/lc_connect", 10000);
  }

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

  // given the ts, find the sequence id of the nearest ts
  int lookupTs(double ts_query, double tol = 1e-3) const {
    auto it_low = std::lower_bound(all_raw_time_stamps.begin(), all_raw_time_stamps.end(), ts_query);
    auto it = it_low;
    if (it_low == all_raw_time_stamps.begin()) {
      it = it_low;
    } else if (it_low == all_raw_time_stamps.end()) {
      it = it_low - 1;
    } else {
      it = std::abs(ts_query - *it_low) < std::abs(ts_query - *(it_low - 1)) ?
           it_low : it_low - 1;
    }
    CHECK(std::abs(*it - ts_query) < tol);
    return it_low - all_raw_time_stamps.begin();
  }

  // TODO: load gt and use it in
//  void loadSeqGT(const std::string &f_seq_calib, const std::string &f_seq_ts, const std::string &f_seq_gt_pose,
//                 int seq_beg) {
//    // 1. calibration
//    std::fstream calib_file;
//    calib_file.open(f_seq_calib, std::ios::in);
//    if (calib_file.rdstate() != std::ifstream::goodbit) {
//      std::cout << "Cannot open calibration" << f_seq_calib << ", returning..." << std::endl;
//      return;
//    }
//    bool calib_set = false;
//    Eigen::Isometry3d T_lc_velod_;    // points in velodyne to left camera, Tr
//    Eigen::Quaterniond calib_rot_q;
//    Eigen::Vector3d calib_trans;
//    std::string strbuf;
//    while (std::getline(calib_file, strbuf)) {
//      std::istringstream iss(strbuf);
//      std::string pname;
//      if (iss >> pname) {
//        if (pname == "Tr:") {
//          Eigen::Matrix3d rot_mat;
//          for (int i = 0; i < 12; i++) {
//            if (i % 4 == 3)
//              iss >> calib_trans(i / 4);
//            else
//              iss >> rot_mat(i % 4, i / 4);
//          }
//          calib_rot_q = Eigen::Quaterniond(rot_mat);
//          calib_set = true;
//          break;
//        }
//      }
//    }
//    CHECK(calib_set);
//    T_lc_velod_.setIdentity();
//    T_lc_velod_.rotate(calib_rot_q);
//    T_lc_velod_.pretranslate(calib_trans);
//
//    // 2. ts and gt left camera pose
//    std::fstream ts_file, gt_pose_file;
//    ts_file.open(f_seq_ts, std::ios::in);
//    // the ts with the raw lidar data, and we select a segment that matches the sequence
//    gt_pose_file.open(f_seq_gt_pose, std::ios::in);
//    if (ts_file.rdstate() != std::ifstream::goodbit) {
//      std::cout << "Cannot open ts" << f_seq_ts << ", returning..." << std::endl;
//      return;
//    }
//    if (gt_pose_file.rdstate() != std::ifstream::goodbit) {
//      std::cout << "Cannot open gt pose" << f_seq_gt_pose << ", returning..." << std::endl;
//      return;
//    }
//
////    int ts_cnt = 0;
////    while (std::getline(ts_file, strbuf)) {
////      std::istringstream iss(strbuf);
////      std::string pname;
////      if (iss >> pname) {
////        if (ts_cnt >= seq_beg) {
////
////        }
////        ts_cnt++;
////      }
////    }
//
//  }

  void processOnce(int &cnt) {

    pcl::PointCloud<pcl::PointXYZ>::ConstPtr out_ptr = nullptr;
    geometry_msgs::TransformStamped tf_gt_last;
//    out_ptr = ros_io.getLidarPointCloud();
    out_ptr = ros_io.getLidarPointCloud(tf_gt_last);
    if (!out_ptr)
      return;

//    std::cout << out_ptr->size() << std::endl;
//    std::cout << out_ptr->header.seq << std::endl;

    ros::Time time;
    time.fromNSec(out_ptr->header.stamp);
    if (!time_beg_set) {
      time_beg = time;
      time_beg_set = true;
    }

    Eigen::Isometry3d T_gt_last; // velodyne to world transform

    // TODO: Where do we get the gt?
    // case 1: use gpx
    T_gt_last = tf2::transformToEigen(tf_gt_last); // use looked up tf
    // case 2: use sequence gt

    int scan_seq = lookupTs(time.toSec());

    Eigen::Vector3d time_translate(0, 0, 1);
    time_translate = time_translate * (time.toSec() - time_beg.toSec()) / 60;   // elevate 1m per minute

//    T_gt_last.pretranslate(time_translate);
    gt_poses.emplace_back(T_gt_last);
    poses_time_z_shift.emplace_back(time_translate.z());

    tf_gt_last.transform.translation.z += time_translate.z();// elevate 1m per minute
    publishPath(time, tf_gt_last);
    publishScanSeqText(time, tf_gt_last, scan_seq);

    printf("---\nour curr seq: %d, stamp: %lu\n", cnt, time.toNSec());

    if (cnt == 1291) {
      printf("Stop here\n");
    }

    std::vector<std::pair<int, int>> new_lc_pairs;

    ContourManagerConfig config;
    config.lv_grads_ = {1.5, 2, 2.5, 3, 3.5, 4};
    std::shared_ptr<ContourManager> cmng_ptr(new ContourManager(config, cnt));
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
    contour_db.queryRangedKNN(cmng_ptr, candidate_loop, cand_corr, bev_tfs);
    printf("%lu Candidates in %7.5fs: \n", candidate_loop.size(), clk.toc());

    bool has_valid_lc = false;

    for (int j = 0; j < candidate_loop.size(); j++) {
      printf("Matching new: %d with old: %d:", cnt, candidate_loop[j]->getIntID());
      new_lc_pairs.emplace_back(cnt, candidate_loop[j]->getIntID());

      // record "valid" loop closure
      auto tf_err = ConstellCorrelation::evalMetricEst(bev_tfs[j], gt_poses[new_lc_pairs.back().second],
                                                       gt_poses[new_lc_pairs.back().first], config);

      // metric error benchmark
      double err_vec[3] = {tf_err.translation().x(), tf_err.translation().y(), std::atan2(tf_err(1, 0), tf_err(0, 0))};
      printf(" Error: dx=%f, dy=%f, dtheta=%f\n", err_vec[0], err_vec[1], err_vec[2]);
      trans_rmse.addOneErr(err_vec);
      rot_rmse.addOneErr(err_vec + 2);
      printf(" Error mean: t:%7.4f, r:%7.4f of %d lc\n", trans_rmse.getMean(), rot_rmse.getMean(), trans_rmse.cnt_sqs);
      printf(" Error rmse: t:%7.4f, r:%7.4f\n", trans_rmse.getRMSE(), rot_rmse.getRMSE());

      if ((gt_poses[new_lc_pairs.back().first].translation() -
           gt_poses[new_lc_pairs.back().second].translation()).norm() < 4.0) {
        valid_lc_cnt++;
        has_valid_lc = true;
      }

      // write file
      std::string f_name =
          PROJ_DIR + "/results/match_comp_img/lc_" + cmng_ptr->getStrID() + "-" + candidate_loop[j]->getStrID() +
          ".png";
      ContourManager::saveMatchedPairImg(f_name, *cmng_ptr, *candidate_loop[j]);
      printf("Image saved: %s-%s\n", cmng_ptr->getStrID().c_str(), candidate_loop[j]->getStrID().c_str());
    }

    if (has_valid_lc)
      lc_pose_cnt++;
    else {
      bool has_fn = false;
      for (int i = 0; i < cnt - 150; i++) {
        double dist = (gt_poses[i].translation() - gt_poses[cnt].translation()).norm();
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
    contour_db.pushAndBalance(seq_cnt++, time.toSec());
    printf("Rebalance tree cost: %7.5f\n", clk.toc());

    // plot
    publishLCConnections(new_lc_pairs, time);
    cnt++;

    printf("Accumulated valid lc: %d\n", valid_lc_cnt);
    printf("Accumulated lc poses: %d\n", lc_pose_cnt);
    printf("Accumulated fn poses: %d\n", lc_fn_pose_cnt);

  }


  void publishPath(ros::Time time, const geometry_msgs::TransformStamped &tf_gt_last) {
    path_msg.header.stamp = time;
    geometry_msgs::PoseStamped ps;
    ps.pose.orientation = tf_gt_last.transform.rotation;
    ps.pose.position.x = tf_gt_last.transform.translation.x;
    ps.pose.position.y = tf_gt_last.transform.translation.y;
    ps.pose.position.z = tf_gt_last.transform.translation.z;
    ps.header = path_msg.header;
    path_msg.poses.emplace_back(ps);
    pub_path.publish(path_msg);
  }

  void publishScanSeqText(ros::Time time, const geometry_msgs::TransformStamped &tf_gt_last, int seq) {
    // index
    visualization_msgs::Marker marker;
    marker.header.frame_id = "world";
    marker.header.stamp = time;
    marker.ns = "myns";
    marker.id = seq;

    marker.action = visualization_msgs::Marker::ADD;
    marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;

    marker.text = std::to_string(seq);
    marker.scale.z = 0.25;
    marker.lifetime = ros::Duration();

    marker.pose.orientation = tf_gt_last.transform.rotation;
    marker.pose.position.x = tf_gt_last.transform.translation.x;
    marker.pose.position.y = tf_gt_last.transform.translation.y;
    marker.pose.position.z = tf_gt_last.transform.translation.z;

    marker.color.a = 1.0; // Don't forget to set the alpha!
    marker.color.r = 0.0f;
    marker.color.g = 1.0f;
    marker.color.b = 1.0f;

    idx_text_array.markers.emplace_back(marker);
    pub_pose_idx.publish(idx_text_array);
  }

  void publishLCConnections(const std::vector<std::pair<int, int>> &new_lc_pairs, ros::Time time) {
    printf("Num new pairs: %lu\n", new_lc_pairs.size());
    line_array.markers.clear();

    for (const auto &pr: new_lc_pairs) {
      visualization_msgs::Marker marker;
      marker.header.frame_id = "world";
      marker.header.stamp = time;
      marker.ns = "myns";
      marker.id = lc_cnt++;

      marker.action = visualization_msgs::Marker::ADD;
      marker.type = visualization_msgs::Marker::LINE_STRIP;

      double len = (gt_poses[pr.first].translation() - gt_poses[pr.second].translation()).norm();

      geometry_msgs::Point p1;
      p1.x = gt_poses[pr.first].translation().x();
      p1.y = gt_poses[pr.first].translation().y();
      p1.z = gt_poses[pr.first].translation().z() + poses_time_z_shift[pr.first];
      marker.points.emplace_back(p1);
      p1.x = gt_poses[pr.second].translation().x();
      p1.y = gt_poses[pr.second].translation().y();
      p1.z = gt_poses[pr.second].translation().z() + poses_time_z_shift[pr.second];
      marker.points.emplace_back(p1);

      marker.lifetime = ros::Duration();

      if (len > 10)
        marker.scale.x = 0.01;
      else
        marker.scale.x = 0.1;

      marker.color.a = 1.0; // Don't forget to set the alpha!
      marker.color.r = 0.0f;
      marker.color.g = 1.0f;
      marker.color.b = 0.0f;

      line_array.markers.emplace_back(marker);
    }
    pub_lc_connections.publish(line_array);
  }

};

int main(int argc, char **argv) {
  ros::init(argc, argv, "rosbag_play_test");
  ros::NodeHandle nh;

  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);

  printf("tag 0\n");
  RosBagPlayLoopTest o(nh);

  std::string ts_path = "/home/lewis/catkin_ws2/src/contour-context/results/kitti_seq05_seconds.txt";
  o.loadRawTs(ts_path);

  ros::Rate rate(50);
  int cnt = 0;

  while (ros::ok()) {
    ros::spinOnce();

    o.processOnce(cnt);

    rate.sleep();
  }

  return 0;
}
