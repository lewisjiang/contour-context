//
// Created by lewis on 5/5/22.
//

#ifndef CONT2_CONTOUR_MNG_H
#define CONT2_CONTOUR_MNG_H

#include "cont2/contour.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"

struct ContourManagerConfig {
  std::vector<float> lev_grads_;  // n marks, n+1 levels
  //
  float reso_row_ = 2.0f, reso_col_ = 2.0f;
  int n_row_ = 100, n_col_ = 100;
  float lidar_height_ = 2.0f;  // ground assumption
  float blind_sq_ = 9.0f;

  int cont_cnt_thres_ = 5; // the cell count threshold dividing a shaped blob from a point
};


// manage the collection of contours in a scan
class ContourManager {
  // configuration
  const ContourManagerConfig cfg_;
  const float VAL_ABS_INF_ = 1e3;

  // property
  float x_max_, x_min_, y_max_, y_min_;
  std::string str_id;

  // data
  std::vector<std::vector<std::shared_ptr<ContourView>>> cont_views_;  // TODO: use a parallel vec of vec for points
  cv::Mat bev_;
  std::vector<std::vector<V2F>> c_height_position_;  // downsampled but not discretized point xy position
  float max_bin_val_ = -VAL_ABS_INF_, min_bin_val_ = VAL_ABS_INF_;
  // TODO: se2
  // TODO: z axis pointing down


  // bookkeeping

protected:
  template<typename PointType>
  ///
  /// \tparam PointType
  /// \param pt
  /// \return (row, col) of the matched point
  std::pair<int, int> hashPointToImage(const PointType &pt) {
    std::pair<int, int> res{-1, -1};
    float padding = 1e-2;
    if (pt.x < x_min_ + padding || pt.x > x_max_ - padding || pt.y < y_min_ + padding || pt.y > y_max_ - padding ||
        (pt.y * pt.y + pt.x * pt.x) < cfg_.blind_sq_) {
//      std::cout << pt.x << "\t" << pt.y << std::endl;
      return res;
    }
    res.first = int(std::floor(pt.x / cfg_.reso_row_)) + cfg_.n_row_ / 2;
    res.second = int(std::floor(pt.y / cfg_.reso_col_)) + cfg_.n_col_ / 2;

    DCHECK(res.first >= 0 && res.first < cfg_.n_row_);
    DCHECK(res.second >= 0 && res.second < cfg_.n_col_);

    return res;
  }

public:
  explicit ContourManager(const ContourManagerConfig &config) : cfg_(config) {
    DCHECK(cfg_.n_col_ % 2 == 0);
    DCHECK(cfg_.n_row_ % 2 == 0);
    DCHECK(!cfg_.lev_grads_.empty());

    x_min_ = -(cfg_.n_row_ / 2) * cfg_.reso_row_;
    x_max_ = -x_min_;
    y_min_ = -(cfg_.n_col_ / 2) * cfg_.reso_col_;
    y_max_ = -y_min_;

    bev_ = cv::Mat::ones(cfg_.n_row_, cfg_.n_col_, CV_32F) * (-VAL_ABS_INF_);
//    std::cout << bev_ << std::endl;
    c_height_position_ = std::vector<std::vector<V2F>>(cfg_.n_row_, std::vector<V2F>(cfg_.n_col_, V2F::Zero()));

  }

  template<typename PointType>
  ///
  /// \tparam PointType
  /// \param ptr_gapc Gravity aligned point cloud, better undistorted. Z axis should point up.
  void makeBEV(typename pcl::PointCloud<PointType>::ConstPtr &ptr_gapc) {  //
    CHECK(ptr_gapc);
    CHECK_GT(ptr_gapc->size(), 10);

    // Downsample before using?
    for (const auto &pt: ptr_gapc->points) {
      std::pair<int, int> rc = hashPointToImage<PointType>(pt);
      if (rc.first > 0) {
        float height = cfg_.lidar_height_ + pt.z;
        if (bev_.at<float>(rc.first, rc.second) < height) {
          bev_.at<float>(rc.first, rc.second) = height;
          c_height_position_[rc.first][rc.second] = V2F(pt.x, pt.y);  // TODO: use the same coordinate as row and col
        }
        max_bin_val_ = max_bin_val_ < height ? height : max_bin_val_;
        min_bin_val_ = min_bin_val_ > height ? height : min_bin_val_;
      }
    }
    printf("Max/Min bin height: %f %f\n", max_bin_val_, min_bin_val_);
    str_id = std::to_string(ptr_gapc->header.stamp);

    cv::Mat mask, view;
    inRange(bev_, cv::Scalar::all(0), cv::Scalar::all(max_bin_val_), mask);
    normalize(bev_, view, 0, 255, cv::NORM_MINMAX, -1, mask);
    cv::imwrite("cart_context-" + str_id + ".png", view);
  }

  void makeContours() {
    float h_min = -VAL_ABS_INF_;
    cv::Mat last_label_img;
    int lev = 0;
    for (const auto &cap: cfg_.lev_grads_) {
      printf("Height [%f, %f]\n", h_min, cap);
      // clamp image
      if (cont_views_.empty()) {
        cv::Mat mask, mask_u8;
        cv::threshold(bev_, mask, h_min, 255, cv::THRESH_BINARY); // mask is same type and dimension as bev_
        // 1. select points higher than a threshold
        mask.convertTo(mask_u8, CV_8U);

        cv::imwrite("cart_context-mask-" + std::to_string(lev) + "-" + str_id + ".png", mask_u8);

        // 2. calculate connected blobs
        cv::Mat labels, stats, centroids;
        cv::connectedComponentsWithStats(mask_u8, labels, stats, centroids, 8);

        // // aux: show image contour group
        cv::Mat label_img;
        cv::normalize(labels, label_img, 0, 255, cv::NORM_MINMAX);
        cv::imwrite("cart_context-labels-" + std::to_string(lev) + "-" + str_id + ".png", label_img);

        cv::imwrite("cart_context-mask-" + std::to_string(lev) + "-" + str_id + ".png", mask_u8);

        // 3. create coutours for each connected component
        for (int n = 0; n < stats.rows; n++) {
          printf("Area: %d\n", stats.at<int>(n, cv::CC_STAT_AREA));

        }
      }

      lev++;
      h_min = cap;
    }


  }
};


#endif //CONT2_CONTOUR_MNG_H
