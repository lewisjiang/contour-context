//
// Created by lewis on 5/12/22.
//

#ifndef CONT2_IO_BIN_H
#define CONT2_IO_BIN_H

/* Read binary data without ROS interface
 * */

#include <string>
#include <sstream>
#include <cstdio>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "tools/pointcloud_util.h"

// Read KITTI lidar bin (same for MulRan)
// assumptions/approximations:
//  1. every lidar frame has a corresponding gt pose
//  2. timestamps and the extrinsic are insignificant
class ReadKITTILiDAR {
  const std::string kitti_raw_dir_, date_, seq_;

  std::vector<std::pair<int, Eigen::Isometry3d>> imu_gt_poses_;
  std::vector<std::pair<int, std::string>> lidar_ts_, imu_ts_;

  int max_index_num = 10000;
  int seq_name_len = 10; // ".bin", ".txt" excluded

  // T_imu_velodyne = T_imu_w * T_w_velodyne
  Eigen::Isometry3d T_imu_velod_;


public:
  explicit ReadKITTILiDAR(std::string &kitti_raw_dir, std::string &date, std::string &seq) : kitti_raw_dir_(
      kitti_raw_dir), date_(date), seq_(seq) {
    // Read extrinsic

    std::string calib_path = kitti_raw_dir_ + "/" + date_ + "/calib_imu_to_velo.txt";
    std::fstream calib_file;
    calib_file.open(calib_path, std::ios::in);
    if (calib_file.rdstate() != std::ifstream::goodbit) {
      std::cout << "Cannot open " << calib_path << ", failed to initialize..." << std::endl;
      return;
    }
    Eigen::Quaterniond calib_rot_q;
    Eigen::Vector3d calib_trans;

    std::string strbuf;
    while (std::getline(calib_file, strbuf)) {
      std::istringstream iss(strbuf);
      std::string pname;
      if (iss >> pname) {
        if (pname == "R:") {
          Eigen::Matrix3d rot_mat;
          for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
              iss >> rot_mat(i, j);
          calib_rot_q = Eigen::Quaterniond(rot_mat);
        } else if (pname == "T:") {
          for (int i = 0; i < 3; i++)
            iss >> calib_trans(i);
        }
      }
    }
    T_imu_velod_.setIdentity();
    T_imu_velod_.rotate(calib_rot_q);
    T_imu_velod_.pretranslate(calib_trans);

    // Read GNSS-imu poses. Not gt
    double scale = 0;
    Eigen::Vector3d trans_orig(0, 0, 0);
    for (int idx = 0; idx < max_index_num; idx++) {
      std::string idx_str = std::to_string(idx);
      std::string imu_entry_path = kitti_raw_dir_ + "/" + date_ + "/" + seq_ + "/oxts/data/" +
                                   std::string(seq_name_len - idx_str.length(), '0') + idx_str + ".txt";
      std::fstream infile;
      infile.open(imu_entry_path, std::ios::in);
      if (infile.rdstate() != std::ifstream::goodbit) {
        std::cout << "Cannot open " << imu_entry_path << ", breaking loop..." << std::endl;
        break;
      }
      std::string sbuf, pname;
      std::getline(infile, sbuf); // the data has only one line
      std::istringstream iss(sbuf);

      double pose_dat[6];
      for (double &i: pose_dat)
        iss >> i;

      infile.close();

      // https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
      double er = 6378137.0;
      if (scale == 0) {
        scale = std::cos(pose_dat[0] * M_PI / 180.0);
      }

      Eigen::Vector3d trans(scale * pose_dat[1] * M_PI * er / 180,
                            scale * er * std::log(std::tan((90 + pose_dat[0]) * M_PI / 360)),
                            pose_dat[2]);
      Eigen::Quaterniond rot = Eigen::Quaterniond(Eigen::AngleAxisd(pose_dat[5], Eigen::Vector3d::UnitZ())
                                                  * Eigen::AngleAxisd(pose_dat[4], Eigen::Vector3d::UnitY())
                                                  * Eigen::AngleAxisd(pose_dat[3], Eigen::Vector3d::UnitX()));
      if (trans_orig.sum() == 0)
        trans_orig = trans;

      trans = trans - trans_orig;
      Eigen::Isometry3d res;
      res.setIdentity();
      res.rotate(rot);
      res.pretranslate(trans);
      imu_gt_poses_.emplace_back(idx, res);
    }
  };

  // get all gt pose (to display)
  const std::vector<std::pair<int, Eigen::Isometry3d>> &getGNSSImuPoses() const {
    return imu_gt_poses_;
  }

  const Eigen::Isometry3d &get_T_imu_velod() const {
    return T_imu_velod_;
  }

  // get point cloud
  // we may not need to display it in rviz
  template<typename PointType>
  typename pcl::PointCloud<PointType>::ConstPtr getLidarPointCloud(int idx, std::string &str_idx0lead) {
    typename pcl::PointCloud<PointType>::Ptr out_ptr = nullptr;

    std::string idx_str = std::to_string(idx);
    str_idx0lead = std::string(seq_name_len - idx_str.length(), '0') + idx_str;
    std::string lidar_bin_path =
        kitti_raw_dir_ + "/" + date_ + "/" + seq_ + "/velodyne_points/data/" + str_idx0lead + ".bin";

    return readKITTIPointCloudBin<PointType>(lidar_bin_path);
  }


};

// Read PointCloud2 data
class ReadPointCloud2 {
public:
};

#endif //CONT2_IO_BIN_H
