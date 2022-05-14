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
  std::mutex mtx_lidar_;


protected:
  void commonLidarMsgCallback(const sensor_msgs::PointCloud2::ConstPtr &msg) {
    typename pcl::PointCloud<PointType>::Ptr ptr(new pcl::PointCloud<PointType>());
    pcl::fromROSMsg(*msg, *ptr);

    printf("Data\n");

    mtx_lidar_.lock();
    lidar_buffer_.emplace_back(ptr);
    mtx_lidar_.unlock();
  }

public:
  Cont2_ROS_IO(int mode, const std::string &sparam, ros::NodeHandle &nh) : mode_(mode) {
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


};


#endif //CONT2_IO_ROS_H
