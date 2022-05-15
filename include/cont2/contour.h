//
// Created by lewis on 5/5/22.
//

/*
 * Slice actually...
 * */

#ifndef CONT2_CONTOUR_H
#define CONT2_CONTOUR_H

#include <utility>
#include <vector>
#include <set>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <iostream>

#include <glog/logging.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>

#include <opencv2/core/types.hpp>

typedef Eigen::Matrix<float, 2, 1> V2F;
typedef Eigen::Matrix<float, 2, 2> M2F;


//struct RectRoi {
//  int r, c, nr, nc;
//
//  RectRoi(int r0, int c0, int nr0, int nc0) : r(r0), c(c0), nr(nr0), nc(nc0) {}
//};

struct ContourViewConfig {
  int min_cell_cov_ = 4;
  float point_sigma_ = 1.5; // have nothing to do with resolution: on pixel only
};

class ContourView {
  // Coordinate definition:
  //  row as x, col as y, center of pixel(0,0) as origin.
  //  Use (row, col) to access all the image data

  // config
  const ContourViewConfig cfg_;

  // property:
  int level_;
  float h_min_, h_max_;
  cv::Rect aabb_; // axis aligned bounding box of the current contour
//  int poi_[2]; // a point belonging to this contour/slice

  // data (collected on the run)
  int cell_cnt_{};
  V2F cell_pos_sum_;
  M2F cell_pos_tss_;
  float cell_vol3_{};  // or "weight" of the elevation mountain. Should we include volumns under the h_min_?
  V2F cell_vol3_torq_;

  // statistical summary
  V2F pos_mean_;
  M2F pos_cov_;
  M2F eig_vecs_; // gaussian ellipsoid axes. if ecc_feat_==false, this is meaningless
  V2F eig_vals_;
  float eccen_{};   // 0: circle
  float vol3_mean_{};
  V2F com_; // center of mass
  bool ecc_feat_ = false;   // eccentricity large enough (with enough cell count)
  bool com_feat_ = false;   // com not at fitted geometric center

  // Raw data (the pixels that belong to this Contour. Is is necessary?)
  // TODO

  // hierarchy
  std::shared_ptr<ContourView> parent_;
  std::vector<std::shared_ptr<ContourView>> children_;

public:
  // TODO: 0. build a contour from 3: pic, roi, height threshold. Called in manager.
  explicit ContourView(int level, float h_min, float h_max, const cv::Rect &aabb,
                       std::shared_ptr<ContourView> parent) : level_(level), h_min_(h_min), h_max_(h_max),
                                                              aabb_(aabb), parent_(std::move(parent)) {
    cell_pos_sum_.setZero();
    cell_pos_tss_.setZero();
    cell_vol3_torq_.setZero();
  };

  // TODO: call this function everytime encounters a pixel belonging to this connected component
  void runningStats(int curr_row, int curr_col, float height) {
    cell_cnt_ += 1;
    V2F v_rc(curr_row, curr_col);
    cell_pos_sum_ += v_rc;
    cell_pos_tss_ += v_rc * v_rc.transpose();
    cell_vol3_ += height;
    cell_vol3_torq_ += height * v_rc;
  }

  void runningStatsF(float curr_row, float curr_col, float height) {   // a more accurate one with continuous coordinate
    cell_cnt_ += 1;
    V2F v_rc(curr_row, curr_col);
    cell_pos_sum_ += v_rc;
    cell_pos_tss_ += v_rc * v_rc.transpose();
    cell_vol3_ += height;
    cell_vol3_torq_ += height * v_rc;
  }

  // TODO: 1. build children_ from a current contour

  // TODO: 2. calculate statistics from running data (including feature hypothesis)
  void calcStatVals() {
    pos_mean_ = cell_pos_sum_ / cell_cnt_;

    vol3_mean_ = cell_vol3_ / cell_cnt_;
    com_ = cell_vol3_torq_ / cell_vol3_;

    // eccentricity:
    if (cell_cnt_ < cfg_.min_cell_cov_) {
      pos_cov_ = M2F::Ones() * cfg_.point_sigma_ * cfg_.point_sigma_;
      ecc_feat_ = false;
      com_feat_ = false;
    } else {
      pos_cov_ = (cell_pos_tss_ - cell_pos_sum_ * pos_mean_.transpose() - pos_mean_ * cell_pos_sum_.transpose() +
                  pos_mean_ * pos_mean_.transpose()) / (cell_cnt_ - 1);
      Eigen::SelfAdjointEigenSolver<M2F> es(pos_cov_.template selfadjointView<Eigen::Upper>());
      eig_vals_ = es.eigenvalues();  // increasing order
      if (eig_vals_(0) < cfg_.point_sigma_)  // determine if eccentricity feat using another function
        eig_vals_(0) = cfg_.point_sigma_;
      if (eig_vals_(1) < cfg_.point_sigma_)
        eig_vals_(1) = cfg_.point_sigma_;
      eccen_ = std::sqrt(eig_vals_(1) * eig_vals_(1) - eig_vals_(0) * eig_vals_(0)) / eig_vals_(1);
      eig_vecs_ = es.eigenvectors();

      ecc_feat_ = eccentricitySalient();

      // vol/weight of mountain:
      com_feat_ = centerOfMassSalient();
    }

  }

  // TODO
  bool eccentricitySalient() {
    return false;
  }

  // TODO
  bool centerOfMassSalient() {
    return false;
  }


  // TODO: 3. check if two contours can be accepted as from the same heatmap peak, and return the transform
  // This is one of the checks for consensus (distributional), the other one is constellation
  // T_tgt = T_delta * T_src
  static std::pair<Eigen::Isometry2f, bool> checkCorresp(const ContourView &cont_src, const ContourView &cont_tgt) {
    return {};
  }

  // TODO: 4. add two contours
  static ContourView addContours(const ContourView &cont1, const ContourView &cont2) {
    CHECK_EQ(cont1.level_, cont2.level_);
  }

  // getter setter
  int getArea() const {
    return cell_cnt_;
  }

  void addChildren(std::shared_ptr<ContourView> &chd) {
    children_.push_back(chd);
  }

  // auxiliary functions
  // 1. get the position of all contour pixels
  std::vector<std::vector<int>> getContPixelPos() const {
    return {};
  }

//  // 2. visualize contour/slice
//  void displayContour(const std::string &fpath) const {
//
//  }

};


#endif //CONT2_CONTOUR_H
