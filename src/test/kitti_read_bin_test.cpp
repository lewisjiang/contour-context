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

#include <nav_msgs/Path.h>
#include <visualization_msgs/MarkerArray.h>

#include <geometry_msgs/PoseStamped.h>
#include "tf2/transform_datatypes.h"
//#include <tf2_ros/transform_broadcaster.h>
//#include <tf2_ros/transform_listener.h>
#include <tf2/convert.h>
#include <tf2_eigen/tf2_eigen.h>

const std::string PROJ_DIR = std::string(PJSRCDIR);

class KittuBinDataVis {
  ros::NodeHandle nh;
  ros::Publisher pub_path;
  ros::Publisher pub_index;
  nav_msgs::Path path_msg;
  visualization_msgs::MarkerArray idx_text_array;

  std::vector<std::pair<int, Eigen::Isometry3d>> gt_poses;

public:
  explicit KittuBinDataVis(ros::NodeHandle &nh_, std::vector<std::pair<int, Eigen::Isometry3d>> gt_poses_)
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

  std::string kitti_raw_dir = "/home/lewis/Downloads/datasets/kitti_raw", date = "2011_09_30", seq = "2011_09_30_drive_0018_sync";
  ReadKITTILiDAR reader(kitti_raw_dir, date, seq);

//  // visualize gt poses and index
//  KittuBinDataVis data_test(nh, reader.getGtImuPoses());
//  ros::Rate rate(1);
//  while (ros::ok()) {
//    ros::spinOnce();
//    data_test.dummyLoopOnce();
//    rate.sleep();
//  }

  // analysis data
//  int idx_old = 34, idx_new = 2437;
//  int idx_old = 119, idx_new = 2511;
  int idx_old = 1561, idx_new = 2576;
  std::string s_old, s_new;

  pcl::PointCloud<pcl::PointXYZ>::ConstPtr out_ptr_old = reader.getLidarPointCloud<pcl::PointXYZ>(idx_old, s_old);
  pcl::PointCloud<pcl::PointXYZ>::ConstPtr out_ptr_new = reader.getLidarPointCloud<pcl::PointXYZ>(idx_new, s_new);

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
  // find the nearest keys in each level:
  printf("Keys:\n");
  for (int ll = 0; ll < config.lv_grads_.size(); ll++) {
    printf("\nPermu Level %d\n", ll);
    auto keys1 = cmng_ptr_old->getRetrievalKey(ll);
    auto keys2 = cmng_ptr_new->getRetrievalKey(ll);
    auto bcis1 = cmng_ptr_old->getBCI(ll);
    auto bcis2 = cmng_ptr_new->getBCI(ll);

    RetrievalKey final_k1, final_k2;
    int f1 = 0, f2 = 0;
    KeyFloatType min_diff = 1e6;
    for (int i1 = 0; i1 < keys1.size(); i1++) {
      for (int i2 = 0; i2 < keys2.size(); i2++) {
        const auto &k1 = keys1[i1];
        const auto &k2 = keys2[i2];
        KeyFloatType tmp_dist = (k1 - k2).squaredNorm();
        if (tmp_dist < min_diff) {
          min_diff = tmp_dist;
          final_k1 = k1;
          final_k2 = k2;
          f1 = i1;
          f2 = i2;
        }
      }
    }
    printf("BF compare key finished\n");

    std::vector<ConstellationPair> constell_pairs;
    bool is_constell_sim = BCI::checkConstellSim(bcis1[f1], bcis2[f2], constell_pairs);

    if (is_constell_sim) {
      printf("Found constell\n");

      std::vector<int> sim_idx;
      std::pair<Eigen::Isometry2d, bool> mat_res = ContourManager::calcScanCorresp(*cmng_ptr_old, *cmng_ptr_new,
                                                                                   constell_pairs, sim_idx);
      std::cout << mat_res.second << std::endl;
      std::cout << mat_res.first.matrix() << std::endl;
    } else {
      printf("No constellation found\n");
    }

    printf("Level %d, minimal key diff sq = %f\nkey1bit: ", ll, min_diff);
    for (int key_bit = 0; key_bit < RetrievalKey::SizeAtCompileTime; key_bit++)
      printf("%8.4f\t", final_k1[key_bit]);
    printf("\nkey2bit: ");
    for (int key_bit = 0; key_bit < RetrievalKey::SizeAtCompileTime; key_bit++)
      printf("%8.4f\t", final_k2[key_bit]);
    printf("\n");

  }
  std::string f_name =
      PROJ_DIR + "/results/pair_comp_img/pair_" + cmng_ptr_old->getStrID() + "-" + cmng_ptr_new->getStrID() +
      ".png";
  ContourManager::saveMatchedPairImg(f_name, *cmng_ptr_old, *cmng_ptr_new);

//  // test 2. save slice accumulated from top N contours
////  for (int ll = 0; ll < config.lv_grads_.size(); ll++) {
////
////  }
////  cmng_ptr_old->saveAccumulatedContours(10);
////  cmng_ptr_new->saveAccumulatedContours(10);
//
//  // test 3. show distance description
////  cmng_ptr_old->expShowDists(1, 1, 10);
////  cmng_ptr_new->expShowDists(1, 2, 10);
////
////  cmng_ptr_old->expShowBearing(1, 1, 10);
////  cmng_ptr_new->expShowBearing(1, 2, 10);
//
//  cmng_ptr_old->expShowDists(3, 3, 10);
//  cmng_ptr_new->expShowDists(3, 3, 10);
//
//  cmng_ptr_old->expShowBearing(3, 3, 10);
//  cmng_ptr_new->expShowBearing(3, 3, 10);


  return 0;
}
