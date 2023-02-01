//
// Created by lewis on 8/27/22.
//

#ifndef CONT2_SPINNER_ROS_H
#define CONT2_SPINNER_ROS_H

#include <thread>
#include <ros/ros.h>
#include <glog/logging.h>

#include <nav_msgs/Path.h>
#include <std_msgs/String.h>
#include <visualization_msgs/MarkerArray.h>

#include <geometry_msgs/PoseStamped.h>
#include <tf2/transform_datatypes.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/convert.h>
#include <tf2_eigen/tf2_eigen.h>

// Definition:
// 1. Main purpose: display path and loop connection by calling either a member function (need to implement) or an outer
//  loop function.
// 2. Optional purpose: add the loop detector and evaluator into the spinner by inheriting it.
struct BaseROSSpinner {
  struct GlobalPoseInfo {
    Eigen::Isometry3d T_wl;
    double z_shift{};

    GlobalPoseInfo(const Eigen::Isometry3d &a, const double &b) : T_wl(a), z_shift(b) {}

    GlobalPoseInfo() = default;
  };

  // ros stuff
  ros::NodeHandle nh;
  ros::Publisher pub_path;
  ros::Publisher pub_lc_connections;
  ros::Publisher pub_pose_idx;
  nav_msgs::Path path_msg;
  visualization_msgs::MarkerArray line_array;
  visualization_msgs::MarkerArray idx_text_array;
//  tf2_ros::Buffer tfBuffer;
//  tf2_ros::TransformListener tfListener;
  ros::Subscriber sub_stop_go;

  // The data used for general purpose (w/o gt files, etc.) in the work flow. Used to draw things. Add on the go.
  std::map<int, GlobalPoseInfo> g_poses;

  // bookkeeping
  u_int64_t lc_line_cnt = 0;
  bool stat_paused = false;
  bool stat_terminated = false;
  std::mutex mtx_status;

  // additional util member variables:


  // Functions:

  explicit BaseROSSpinner(ros::NodeHandle &nh_) : nh(nh_) //, tfListener(tfBuffer)
  {
    path_msg.header.frame_id = "world";
    pub_path = nh_.advertise<nav_msgs::Path>("/gt_path", 10000);
    pub_pose_idx = nh_.advertise<visualization_msgs::MarkerArray>("/pose_index", 10000);
    pub_lc_connections = nh_.advertise<visualization_msgs::MarkerArray>("/lc_connect", 10000);

    // rostopic pub cont2_status std_msgs/String "pause" --once
    // rostopic pub cont2_status std_msgs/String "resume" --once
    // rostopic pub cont2_status std_msgs/String "toggle" --once
    sub_stop_go = nh.subscribe("/cont2_status", 100, &BaseROSSpinner::statusCallback, this);
  }

  void statusCallback(const std_msgs::String::ConstPtr &msg) {
    mtx_status.lock();
    if (msg->data == std::string("pause")) {
      stat_paused = true;
    } else if (msg->data == std::string("resume")) {
      stat_paused = false;
    } else if (msg->data == std::string("toggle")) {
      stat_paused = !stat_paused;
    } else if (msg->data == std::string("terminate")) {
      stat_terminated = true;
    }

//    if (msg->data == std::string("end")) {
//      printf("[H] %d \t Need reset\n", idx++);
//      need_rst = true;
//    } else if (msg->data == std::string("exit")) {
//      printf("[H] %d rounds of simulation have finished cleanly. Exiting...\n", idx);
//      mtx_status.unlock();
//      ros::shutdown();
//    } else {
//      printf("[H] %d \t A new round\n", idx);
//      CHECK(!need_rst);  // must finish reset before new round starts
//    }
    mtx_status.unlock();
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

  ///
  /// \param new_lc_pairs The loop pair indexed by sequence key, consistent with map `g_poses`'s key field
  /// \param time
  void publishLCConnections(const std::vector<std::pair<int, int>> &new_lc_pairs, const std::vector<bool> &TF_positive,
                            ros::Time time) {
    printf("Num new pairs: %lu\n", new_lc_pairs.size());
    line_array.markers.clear();

    DCHECK_EQ(new_lc_pairs.size(), TF_positive.size());

    for (int i = 0; i < new_lc_pairs.size(); i++) {
      const auto &pr = new_lc_pairs[i];
      visualization_msgs::Marker marker;
      marker.header.frame_id = "world";
      marker.header.stamp = time;
      marker.ns = "myns";
      marker.id = lc_line_cnt++;

      marker.action = visualization_msgs::Marker::ADD;
      marker.type = visualization_msgs::Marker::LINE_STRIP;

      double len = (g_poses.at(pr.first).T_wl.translation() - g_poses.at(pr.second).T_wl.translation()).norm();

      geometry_msgs::Point p1;
      p1.x = g_poses[pr.first].T_wl.translation().x();
      p1.y = g_poses[pr.first].T_wl.translation().y();
      p1.z = g_poses[pr.first].T_wl.translation().z() + g_poses[pr.first].z_shift;
      marker.points.emplace_back(p1);
      p1.x = g_poses[pr.second].T_wl.translation().x();
      p1.y = g_poses[pr.second].T_wl.translation().y();
      p1.z = g_poses[pr.second].T_wl.translation().z() + g_poses[pr.second].z_shift;
      marker.points.emplace_back(p1);

      marker.lifetime = ros::Duration();

      marker.scale.x = 0.5;
      if (TF_positive[i]) {
        marker.color.a = 1.0; // Don't forget to set the alpha!
        marker.color.r = 0.0f;
        marker.color.g = 1.0f;
        marker.color.b = 0.0f;
      } else {
        marker.color.a = 1.0; // Don't forget to set the alpha!
        marker.color.r = 1.0f;
        marker.color.g = 0.0f;
        marker.color.b = 0.0f;

      }

      line_array.markers.emplace_back(marker);
    }
    pub_lc_connections.publish(line_array);
  }

  static void broadcastCurrPose(geometry_msgs::TransformStamped tf_gt_last) {
    static tf2_ros::TransformBroadcaster br;
    tf_gt_last.header.stamp = ros::Time::now();
    tf_gt_last.header.frame_id = "world";
    tf_gt_last.child_frame_id = "gt_curr";
    br.sendTransform(tf_gt_last);
  }

};


#endif //CONT2_SPINNER_ROS_H
