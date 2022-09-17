//
// Created by lewis on 8/24/22.
//

#ifndef CONT2_POINTCLOUD_UTIL_H
#define CONT2_POINTCLOUD_UTIL_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

template<typename PointType>
typename pcl::PointCloud<PointType>::ConstPtr readKITTIPointCloudBin(const std::string &lidar_bin_path) {
  typename pcl::PointCloud<PointType>::Ptr out_ptr = nullptr;

  // allocate 4 MB buffer (only ~130*4*4 KB are needed)
  int num = 1000000;
  auto *data = (float *) malloc(num * sizeof(float));
  // pointers
  float *px = data + 0;
  float *py = data + 1;
  float *pz = data + 2;
  float *pr = data + 3;

  FILE *stream;
  stream = fopen(lidar_bin_path.c_str(), "rb");
  if (stream) {
    num = fread(data, sizeof(float), num, stream) / 4;
    out_ptr.reset(new pcl::PointCloud<PointType>());
    out_ptr->reserve(num);
    for (int32_t i = 0; i < num; i++) {
      PointType pt;
      pt.x = *px;
      pt.y = *py;
      pt.z = *pz;
      out_ptr->push_back(pt);

      px += 4;
      py += 4;
      pz += 4;
      pr += 4;
    }
    fclose(stream);

  } else {
    printf("Lidar bin file %s does not exist.\n", lidar_bin_path.c_str());
    exit(-1);
  }
  free(data);
  return out_ptr;
}

#endif //CONT2_POINTCLOUD_UTIL_H
