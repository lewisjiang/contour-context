#include<iostream>
#include<string>
#include <ros/ros.h>
#include "cont2/contour_mng.h"
#include "cont2_ros/io_ros.h"

int main(int argc, char **argv) {
  ros::init(argc, argv, "assembly_test");
  ros::NodeHandle nh;

  printf("tag 0\n");

  Cont2_ROS_IO<pcl::PointXYZ> ros_io(0, "/velodyne_points", nh);

  std::vector<ContourManager> scans;

  ros::Rate rate(20);
  int cnt = 0;
  while (ros::ok()) {
    ros::spinOnce();

    pcl::PointCloud<pcl::PointXYZ>::ConstPtr out_ptr = nullptr;
    out_ptr = ros_io.getLidarPointCloud();
    if (out_ptr) {
      std::cout << out_ptr->size() << std::endl;
      std::cout << out_ptr->header.seq << std::endl;

      ContourManagerConfig config;
      config.lev_grads_ = {1.5, 2, 2.5, 3, 3.5, 4};
      ContourManager cmng(config);
      cmng.makeBEV<pcl::PointXYZ>(out_ptr);
//      cmng.makeContours();
      cmng.makeContoursRecurs();
      scans.emplace_back(cmng);
      if (scans.size() > 1) {
        for (int i = 0; i < scans.size(); i++) {
          for (int j = 0; j < i; j++) {
            printf("Match old: %d with new: %d:", j, i);
            TicToc clk_match_once;
            ContourManager::calcScanCorresp(scans[j], scans[i]);
            printf("Match once time: %f\n", clk_match_once.toc());
          }
        }
      }
    }
    rate.sleep();
  }
  return 0;
}
