//
// Created by lewis on 5/5/22.
//

#ifndef CONT2_IO_ROS_H
#define CONT2_IO_ROS_H

#include <iostream>
#include <string>
#include <mutex>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <glog/logging.h>

#include <geometry_msgs/PoseStamped.h>
#include "tf2/transform_datatypes.h"
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/convert.h>
#include <tf2_eigen/tf2_eigen.h>

/*
 * a collection of adaptors to deal with ROS bag, topics, eyc., for the ros independent package.
 */

template<typename PointType>
class Cont2_ROS_IO {
  std::string lidar_topic_;
  std::string bag_path_;
  int mode_ = 0;

  ros::Subscriber sub_lidar_common_;

  std::vector<typename pcl::PointCloud<PointType>::Ptr> lidar_buffer_;
  std::vector<geometry_msgs::TransformStamped> gt_pose_buffer_;
  std::mutex mtx_lidar_;

  tf2_ros::Buffer tfBuffer;
  tf2_ros::TransformListener tfListener;

  uint64_t start_ns;  // ignore timestamps smaller than this one. Useful to precisely control the messages we want.


protected:
  void commonLidarMsgCallback(const sensor_msgs::PointCloud2::ConstPtr &msg) {
    if (msg->header.stamp.toNSec() < start_ns) {
      ROS_INFO("Waiting the message to reach start time, %7.2f secs left...",
               1e-9 * (start_ns - msg->header.stamp.toNSec()));
      return;
    }
    typename pcl::PointCloud<PointType>::Ptr ptr(new pcl::PointCloud<PointType>());
    pcl::fromROSMsg(*msg, *ptr);
    ptr->header.stamp = msg->header.stamp.toNSec(); // pcl conversion uses us, here we overwrite it with ns
    printf("---\nData received in callback, stamp: %lu\n", msg->header.stamp.toNSec());

    mtx_lidar_.lock();
    lidar_buffer_.emplace_back(ptr);
    ros::Time time;
    time.fromNSec(ptr->header.stamp);

    try {
//      auto tf_gt_last = tfBuffer.lookupTransform("world", "velodyne", time, ros::Duration(0.1));
      auto tf_gt_last = tfBuffer.lookupTransform("world", "velo_link", time, ros::Duration(0.1));
      gt_pose_buffer_.template emplace_back(tf_gt_last);
    } catch (tf2::TransformException &ex) {
      lidar_buffer_.pop_back();
      ROS_WARN("%s. Pop last lidar, returning...", ex.what());
//      return;
    }

    mtx_lidar_.unlock();
  }

public:
  Cont2_ROS_IO(int mode, const std::string &sparam, ros::NodeHandle &nh, const uint64_t &ns_beg) : mode_(mode),
                                                                                                   tfListener(tfBuffer),
                                                                                                   start_ns(ns_beg) {
    if (mode == 0) {
      lidar_topic_ = sparam;
      sub_lidar_common_ = nh.subscribe(lidar_topic_, 2000, &Cont2_ROS_IO::commonLidarMsgCallback, this);
    } else if (mode == 1) {
      bag_path_ = sparam;
    } else
      assert(false);
  }

  typename pcl::PointCloud<PointType>::ConstPtr getLidarPointCloud() {
    mtx_lidar_.lock();
    typename pcl::PointCloud<PointType>::Ptr out_ptr;
    if (lidar_buffer_.empty()) {
      out_ptr = nullptr;

    } else {
      out_ptr = lidar_buffer_.front();
      lidar_buffer_.erase(lidar_buffer_.begin());

//      std::cout << out_ptr->header.stamp << std::endl;  // header ts in usec
//      printf("Buffer size: %lu\n", lidar_buffer_.size());
    }
    mtx_lidar_.unlock();
    return out_ptr;
  }

  typename pcl::PointCloud<PointType>::ConstPtr getLidarPointCloud(geometry_msgs::TransformStamped &gt_pose) {
    mtx_lidar_.lock();
    typename pcl::PointCloud<PointType>::Ptr out_ptr;
    if (lidar_buffer_.empty()) {
      out_ptr = nullptr;

    } else {
      out_ptr = lidar_buffer_.front();
      lidar_buffer_.erase(lidar_buffer_.begin());

      gt_pose = gt_pose_buffer_.front();
      gt_pose_buffer_.erase(gt_pose_buffer_.begin());

//      std::cout << out_ptr->header.stamp << std::endl;  // header ts in usec
//      printf("Buffer size: %lu\n", lidar_buffer_.size());
    }
    mtx_lidar_.unlock();
    return out_ptr;
  }


};


#endif //CONT2_IO_ROS_H
