//
// Created by lewis on 5/5/22.
//
#include <ros/ros.h>
#include "tools/bm_util.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

int main(int argc, char **argv) {
  ros::init(argc, argv, "assembly_test");

  std::string image_path = string(PJSRCDIR) + "sample_data/starry_night.jpg";
  Mat img = imread(image_path, IMREAD_GRAYSCALE);
  if (img.empty()) {
    std::cout << "Could not read the image: " << image_path << std::endl;
    return 1;
  }
  cout << img.type() << endl;
  std::cout << img.rows << endl;
  std::cout << img.cols << endl;

  // test a naive connected component searching
  TicToc clk;
  uint8_t ma = 0, mi = 255;
  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
      if (img.at<uint8_t>(i, j) > ma)
        ma = img.at<uint8_t>(i, j);
      if (img.at<uint8_t>(i, j) < mi)
        mi = img.at<uint8_t>(i, j);
    }
  }

  cout << (int) ma << ", " << (int) mi << endl << clk.toc() << endl;


  imshow("Display window", img);
  int k = waitKey(0); // Wait for a keystroke in the window
  if (k == 's') {
    imwrite("starry_night.png", img);
  }
  return 0;
}