//
// Created by lewis on 5/5/22.
//

#ifndef CONT2_CONTOUR_MNG_H
#define CONT2_CONTOUR_MNG_H

#include "cont2/contour.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <opencv2/core/mat.hpp>
#include <opencv2/core.hpp>
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
  cv::Mat1f bev_;
  std::vector<std::vector<V2F>> c_height_position_;  // downsampled but not discretized point xy position, another bev
  float max_bin_val_ = -VAL_ABS_INF_, min_bin_val_ = VAL_ABS_INF_;
  // TODO: se2
  // TODO: z axis pointing down


  // bookkeeping
  cv::Mat1b visualization;

protected:
  template<typename PointType>
  ///
  /// \tparam PointType
  /// \param pt
  /// \return (row, col) of the matched point
  std::pair<int, int> hashPointToImage(const PointType &pt) const {
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

  /// transform points from the lidar frame to continuous image frame
  /// \param p_in_l
  /// \return
  V2F pointToContRowCol(const V2F &p_in_l) const {
    V2F continuous_rc(p_in_l.x() / cfg_.reso_row_ + 0.5f, p_in_l.y() / cfg_.reso_col_ + 0.5f);
    return continuous_rc;
  }

  void makeContourRecursive(const cv::Rect &cc_roi, const cv::Mat1b &cc_mask, int level,
                            const std::shared_ptr<ContourView> &parent) {
    DCHECK(bool(level)==bool(parent));
    if (level >= cfg_.lev_grads_.size())
      return;

    float h_min = cfg_.lev_grads_[level], h_max = VAL_ABS_INF_;

    cv::Mat1f bev_roi = bev_(cc_roi), thres_roi;
    cv::threshold(bev_roi, thres_roi, h_min, 255, cv::THRESH_BINARY);

    cv::Mat1b bin_bev_roi;
    thres_roi.convertTo(bin_bev_roi, CV_8U);

    if (level)
//      cv::bitwise_and(bin_bev_roi, bin_bev_roi, bin_bev_roi, cc_mask);  // Wrong method: some pixels may be unaltered since neglected by mask
      cv::bitwise_and(bin_bev_roi, cc_mask, bin_bev_roi);

    if (level < cfg_.lev_grads_.size() - 1)
      h_max = cfg_.lev_grads_[level - 1];

    // 2. calculate connected blobs
    cv::Mat1i labels, stats;  // int (CV_32S)
    cv::Mat centroids;  // not in use
    cv::connectedComponentsWithStats(bin_bev_roi, labels, stats, centroids, 8, CV_32S);   // on local patch

    // 3. create contours for each connected component
    // https://stackoverflow.com/questions/37745274/opencv-find-perimeter-of-a-connected-component/48618464#48618464
    for (int n = 1; n < stats.rows; n++) {  // n=0: background
      printf("Area: %d\n", stats.at<int>(n, cv::CC_STAT_AREA));

      //Rectangle around the connected component
      // Rect: col0, row0, n_col, n_row
      cv::Rect rect_g(stats(n, 0) + cc_roi.x, stats(n, 1) + cc_roi.y, stats(n, 2), stats(n, 3)); // global: on bev
      cv::Rect rect_l(stats(n, 0), stats(n, 1), stats(n, 2), stats(n, 3)); // local: relative to bev_roi

      cv::Mat1b mask_n = labels(rect_l) == n;

      std::shared_ptr<ContourView> ptr_tmp_cv(new ContourView(level, h_min, h_max, rect_g, parent));

      for (int i = 0; i < rect_l.height; i++)
        for (int j = 0; j < rect_l.width; j++)
          if (mask_n(i, j))
            ptr_tmp_cv->runningStats(i + cc_roi.y, j + cc_roi.x, bev_(i, j));

      ptr_tmp_cv->calcStatVals();
      DCHECK(ptr_tmp_cv->getArea() == stats(n, 4));
      cont_views_[level].emplace_back(ptr_tmp_cv);    // add to the manager's matrix
      if (parent)
        parent->addChildren(ptr_tmp_cv);

      // recurse
      // Get the mask for the contour

      printf("contour ROI: %d, %d, level: %d\n", mask_n.rows, mask_n.cols, level);
      makeContourRecursive(rect_g, mask_n, level + 1, ptr_tmp_cv);

//      if (level == 2) {
//        cv::bitwise_or(mask_n, visualization(rect_g), visualization(rect_g));
//      }

    }

  }

public:
  explicit ContourManager(const ContourManagerConfig &config) : cfg_(config) {
    CHECK(cfg_.n_col_ % 2 == 0);
    CHECK(cfg_.n_row_ % 2 == 0);
    DCHECK(!cfg_.lev_grads_.empty());

    x_min_ = -(cfg_.n_row_ / 2) * cfg_.reso_row_;
    x_max_ = -x_min_;
    y_min_ = -(cfg_.n_col_ / 2) * cfg_.reso_col_;
    y_max_ = -y_min_;

    bev_ = cv::Mat::ones(cfg_.n_row_, cfg_.n_col_, CV_32F) * (-VAL_ABS_INF_);
//    std::cout << bev_ << std::endl;
    c_height_position_ = std::vector<std::vector<V2F>>(cfg_.n_row_, std::vector<V2F>(cfg_.n_col_, V2F::Zero()));
    cont_views_.resize(cfg_.lev_grads_.size());

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
        if (bev_(rc.first, rc.second) < height) {
          bev_(rc.first, rc.second) = height;
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

  void makeContoursRec() {
    cv::Rect full_bev(0, 0, bev_.cols, bev_.rows);
    visualization = cv::Mat::zeros(cfg_.n_row_, cfg_.n_col_, CV_8U);

    makeContourRecursive(full_bev, cv::Mat1b(1, 1), 0, nullptr);

//    cv::imwrite("cart_context-mask-" + std::to_string(3) + "-" + str_id + "rec.png", visualization);
//    for (const auto &x: cont_views_) {
//      printf("level size: %lu\n", x.size());
//    }
  }

  void makeContours() {
//    float h_min = -VAL_ABS_INF_;
    cv::Mat last_label_img;
    int lev = 0;
    for (const auto &h_min: cfg_.lev_grads_) {
      printf("Height [%f, +]\n", h_min);
      // clamp image
      if (cont_views_.empty()) {
        cv::Mat mask, mask_u8;
        cv::threshold(bev_, mask, h_min, 255, cv::THRESH_BINARY); // mask is same type and dimension as bev_
        // 1. select points higher than a threshold
        mask.convertTo(mask_u8, CV_8U);

        cv::imwrite("cart_context-mask-" + std::to_string(lev) + "-" + str_id + ".png", mask_u8);

        // 2. calculate connected blobs
        cv::Mat1i labels, stats;  // int (CV_32S)
        cv::Mat centroids;
        cv::connectedComponentsWithStats(mask_u8, labels, stats, centroids, 8, CV_32S);

        // // aux: show image contour group
        cv::Mat label_img;
        cv::normalize(labels, label_img, 0, 255, cv::NORM_MINMAX);
        cv::imwrite("cart_context-labels-" + std::to_string(lev) + "-" + str_id + ".png", label_img);
        cv::imwrite("cart_context-mask-" + std::to_string(lev) + "-" + str_id + ".png", mask_u8);

        // 3. create contours for each connected component
        // https://stackoverflow.com/questions/37745274/opencv-find-perimeter-of-a-connected-component/48618464#48618464
        std::vector<std::shared_ptr<ContourView>> level_conts;
        for (int n = 1; n < stats.rows; n++) {  // n=0: background
          printf("Area: %d\n", stats.at<int>(n, cv::CC_STAT_AREA));

          //Rectangle around the connected component
          cv::Rect rect(stats(n, 0), stats(n, 1), stats(n, 2), stats(n, 3)); // Rect: col0, row0, n_col, n_row

//          // Get the mask for the contour
//          cv::Mat1b mask_n = labels(rect) == n;
//          printf("countour ROI: %d, %d\n", mask_n.rows, mask_n.cols);

          std::shared_ptr<ContourView> ptr_tmp_cv(
              new ContourView(lev, h_min, h_min, rect, nullptr)); // TODO: dummy h_max

          for (int i = rect.y; i < rect.y + rect.height; i++)
            for (int j = rect.x; j < rect.x + rect.width; j++)
              ptr_tmp_cv->runningStats(i, j, bev_(i, j));

          ptr_tmp_cv->calcStatVals();
          DCHECK(ptr_tmp_cv->getArea() == stats(n, 4));
          level_conts.emplace_back(ptr_tmp_cv);
        }
        cont_views_.emplace_back(level_conts);
      } else {
        // create children from parents (ancestral tree)
        for (auto parent: cont_views_.back()) {

        }

      }

      lev++;
//      h_min = cap;
    }
  }

};


#endif //CONT2_CONTOUR_MNG_H
