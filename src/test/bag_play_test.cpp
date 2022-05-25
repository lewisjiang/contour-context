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

class RosBagPlayLoopTest {
  // ros stuff
  ros::NodeHandle nh;
  ros::Publisher pub_path;
  ros::Publisher pub_lc_connections;
  nav_msgs::Path path_msg;
  visualization_msgs::MarkerArray line_array;

  std::vector<std::shared_ptr<ContourManager>> scans;
  ContourDB contour_db;
  std::vector<Eigen::Isometry3d> gt_poses;

  Cont2_ROS_IO<pcl::PointXYZ> ros_io;

  tf2_ros::Buffer tfBuffer;
  tf2_ros::TransformListener tfListener;
  int lc_cnt{}, seq_cnt{};

  bool time_beg_set = false;
  ros::Time time_beg;


public:
  explicit RosBagPlayLoopTest(ros::NodeHandle &nh_) : nh(nh_), ros_io(0, "/velodyne_points", nh_),
                                                      tfListener(tfBuffer),
                                                      contour_db(ContourDBConfig(), std::vector<int>({1, 2, 3, 4})) {
    path_msg.header.frame_id = "world";
    pub_path = nh_.advertise<nav_msgs::Path>("/gt_path", 10000);
    pub_lc_connections = nh_.advertise<visualization_msgs::MarkerArray>("/lc_connect", 10000);
  }

  void processOnce(int &cnt) {

    pcl::PointCloud<pcl::PointXYZ>::ConstPtr out_ptr = nullptr;
    out_ptr = ros_io.getLidarPointCloud();
    if (!out_ptr)
      return;

    std::cout << out_ptr->size() << std::endl;
    std::cout << out_ptr->header.seq << std::endl;

    ros::Time time;
    time.fromNSec(out_ptr->header.stamp);
    if (!time_beg_set) {
      time_beg = time;
      time_beg_set = true;
    }

    try {
      auto tf_gt_last = tfBuffer.lookupTransform("world", "velodyne", time, ros::Duration(0.2));
      auto T_gt_last = tf2::transformToEigen(tf_gt_last);

      Eigen::Vector3d time_translate(0, 0, 1);
      time_translate = time_translate * (time.toSec() - time_beg.toSec()) / 60;   // elevate 1m per minute

      T_gt_last.pretranslate(time_translate);
      gt_poses.emplace_back(T_gt_last);

      tf_gt_last.transform.translation.z += time_translate.z();// elevate 1m per minute
      publishPath(time, tf_gt_last);

    } catch (tf2::TransformException &ex) {
      ROS_WARN("%s. Returning...", ex.what());
      return;
    }

    printf("our curr seq: %d, stamp: %lu\n", cnt, time.toNSec());

    std::vector<std::pair<int, int>> new_lc_pairs;

    ContourManagerConfig config;
    config.lv_grads_ = {1.5, 2, 2.5, 3, 3.5, 4};
    std::shared_ptr<ContourManager> cmng_ptr(new ContourManager(config, cnt));
    cmng_ptr->makeBEV<pcl::PointXYZ>(out_ptr);
//      cmng_ptr.makeContours();
    cmng_ptr->makeContoursRecurs();

    // save images of layers
//    for (int i = 0; i < config.lv_grads_.size(); i++) {
//      std::string f_name = PROJ_DIR + "/results/layer_img/contour_" + "lv" + std::to_string(i) + "_" +
//                           std::to_string(out_ptr->header.stamp) + ".png";
//      cmng_ptr.saveContourImage(f_name, i);
//    }

//    // case 1: poll over all data
//    scans.emplace_back(cmng_ptr);
//    if (scans.size() > 1) {
//      int i = cnt;
//      for (int j = 0; j < i - 100; j += 5) { // avoid nearby loop closure
//        printf("Matching new: %d with old: %d:", i, j);
//        TicToc clk_match_once;
//        auto lc_detect_res = ContourManager::calcScanCorresp(*scans[i], *scans[j]);  // (src, tgt)
//        printf("Match once time: %f\n", clk_match_once.toc());
//        if (lc_detect_res.second)
//          new_lc_pairs.emplace_back(i, j);
//      }
//    }

    // case 2: use tree
    // 2.1 query
    std::vector<std::shared_ptr<ContourManager>> candidate_loop;
    std::vector<KeyFloatType> dists_sq;
    contour_db.queryCandidates(*cmng_ptr, candidate_loop, dists_sq);
    printf("%lu Candidates, ", candidate_loop.size());
    if (!candidate_loop.empty())
      printf(" dist sq from %7.4f to %7.4f\n", dists_sq.front(), dists_sq.back());
    else
      printf("\n");

    for (int j = 0; j < candidate_loop.size(); j++) {
      printf("Matching new: %d with old: %d:", cnt, candidate_loop[j]->getIntID());
      TicToc clk_match_once;
      auto lc_detect_res = ContourManager::calcScanCorresp(*cmng_ptr, *candidate_loop[j]);  // (src, tgt)
      printf("Match once time: %f\n", clk_match_once.toc());
      if (lc_detect_res.second)
        new_lc_pairs.emplace_back(cnt, j);
    }
    // 2.2 add new
    contour_db.addScan(cmng_ptr, time.toSec());
    // 2.3 balance
    contour_db.pushAndBalance(seq_cnt++, time.toSec());

    // plot
    publishLCConnections(new_lc_pairs, time);
    cnt++;

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

      geometry_msgs::Point p1;
      p1.x = gt_poses[pr.first].translation().x();
      p1.y = gt_poses[pr.first].translation().y();
      p1.z = gt_poses[pr.first].translation().z();
      marker.points.emplace_back(p1);
      p1.x = gt_poses[pr.second].translation().x();
      p1.y = gt_poses[pr.second].translation().y();
      p1.z = gt_poses[pr.second].translation().z();
      marker.points.emplace_back(p1);

      marker.scale.x = 0.1;

      marker.lifetime = ros::Duration();

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

  printf("tag 0\n");
  RosBagPlayLoopTest o(nh);

  ros::Rate rate(20);
  int cnt = 0;

  while (ros::ok()) {
    ros::spinOnce();

    o.processOnce(cnt);

    rate.sleep();
  }

  return 0;
}
