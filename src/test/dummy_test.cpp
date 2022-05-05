#include<iostream>
#include<string>
#include <ros/ros.h>
#include "cont2/contour_mng.h"

int main(int argc, char **argv) {
  ros::init(argc, argv, "assembly_test");
  ros::NodeHandle nh;
  ros::Rate rate(5);
  int cnt = 0;
  while (ros::ok()) {
    std::cout << "haha" << cnt++ << std::endl;
    rate.sleep();
  }
  return 0;
}
