//
// Created by lewis on 8/30/22.
//

#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include "sensor_msgs/PointCloud2.h"
#include "tools/pointcloud_util.h"

const std::string PROJ_DIR = std::string(PJSRCDIR);


int main(int argc, char **argv) {
  ros::init(argc, argv, "pub_a_lidar_msg");
  ros::NodeHandle nh;

  std::string bin_path = PROJ_DIR + "sample_data/001648.bin";

  ros::Publisher pub_cloud = nh.advertise<sensor_msgs::PointCloud2>("/vis_a_pc", 10000);

  pcl::PointCloud<pcl::PointXYZI>::ConstPtr pc_to_show = readKITTIPointCloudBin<pcl::PointXYZI>(bin_path);

  if (!pc_to_show) {
    printf("No pc loaded.\n");
    return -1;
  }

  sensor_msgs::PointCloud2 pc_msg_to_pub;

  pcl::toROSMsg(*pc_to_show, pc_msg_to_pub);

  ros::Rate rate(1);
  while (ros::ok()) {
    ros::spinOnce();
    pc_msg_to_pub.header.stamp = ros::Time::now();
    pc_msg_to_pub.header.frame_id = "world";
    pub_cloud.publish(pc_msg_to_pub);
    printf("%lu\n", pc_msg_to_pub.data.size());
    rate.sleep();
  }


  return 0;
}