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

#include "tools/algos.h"

typedef Eigen::Matrix<float, 2, 1> V2F;
typedef Eigen::Matrix<float, 2, 2> M2F;
typedef Eigen::Matrix<double, 2, 1> V2D;
typedef Eigen::Matrix<double, 2, 2> M2D;


//struct RectRoi {
//  int r, c, nr, nc;
//
//  RectRoi(int r0, int c0, int nr0, int nc0) : r(r0), c(c0), nr(nr0), nc(nc0) {}
//};

struct ContourViewConfig {
  int min_cell_cov_ = 4;
  double point_sigma_ = 1.0; // have nothing to do with resolution: on pixel only
  double com_bias_thres = 0.5;  // com dist from geometric center
};

class ContourView {
public:
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
  V2D cell_pos_sum_;
  M2D cell_pos_tss_;
  float cell_vol3_{};  // or "weight" of the elevation mountain. Should we include volumns under the h_min_?
  V2D cell_vol3_torq_;

  // statistical summary
  V2D pos_mean_;
  M2D pos_cov_;
  V2D eig_vals_;
  M2D eig_vecs_; // gaussian ellipsoid axes. if ecc_feat_==false, this is meaningless
  float eccen_{};   // 0: circle
  float vol3_mean_{};
  V2D com_; // center of mass
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
    DCHECK_GE(curr_row, -0.5f);
    DCHECK_GE(curr_col, -0.5f);
    cell_cnt_ += 1;
    V2D v_rc(curr_row, curr_col);
    cell_pos_sum_ += v_rc;
    cell_pos_tss_ += v_rc * v_rc.transpose();
    cell_vol3_ += height;
    cell_vol3_torq_ += height * v_rc;
  }

  void runningStatsF(float curr_row, float curr_col, float height) {   // a more accurate one with continuous coordinate
    DCHECK_GE(curr_row, -0.5f);
    DCHECK_GE(curr_col, -0.5f);
    cell_cnt_ += 1;
    V2D v_rc(curr_row, curr_col);
    cell_pos_sum_ += v_rc;
    cell_pos_tss_ += v_rc * v_rc.transpose();
    cell_vol3_ += height;
    cell_vol3_torq_ += height * v_rc;
  }

  // TOxDO: 1. build children_ from a current contour

  // TO-DO: 2. calculate statistics from running data (including feature hypothesis)
  void calcStatVals() {
    pos_mean_ = cell_pos_sum_ / cell_cnt_;

    vol3_mean_ = cell_vol3_ / cell_cnt_;
    com_ = cell_vol3_torq_ / cell_vol3_;

    // eccentricity:
    if (cell_cnt_ < cfg_.min_cell_cov_) {
      pos_cov_ = M2D::Identity() * cfg_.point_sigma_ * cfg_.point_sigma_;
      eig_vals_ = V2D(cfg_.point_sigma_, cfg_.point_sigma_);
      eig_vecs_.setIdentity();
      ecc_feat_ = false;
      com_feat_ = false;
    } else {
      pos_cov_ = (cell_pos_tss_ - cell_pos_sum_ * pos_mean_.transpose() - pos_mean_ * cell_pos_sum_.transpose() +
                  pos_mean_ * pos_mean_.transpose() * cell_cnt_) / (cell_cnt_ - 1);
      Eigen::SelfAdjointEigenSolver<M2D> es(pos_cov_.template selfadjointView<Eigen::Upper>());
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
  inline bool eccentricitySalient() {
    return cell_cnt_ > 5 && diff_perc(eig_vals_(0), eig_vals_(1), 0.2) && eig_vals_(1) > 2.5;
  }

  // TODO: should have sth to do with total area
  inline bool centerOfMassSalient() const {
    return (com_ - pos_mean_).norm() > cfg_.com_bias_thres;
  }

  // TODO
  bool orietSalient() {
    return false;
  }


  // TODO: 3. return true if two contours can be accepted as from the same heatmap peak
  //  use normalized L2E as similarity score?
  // This is one of the checks for consensus (distributional), the other one is constellation
  // T_tgt = T_delta * T_src
//  static std::pair<Eigen::Isometry2d, bool> checkCorresp(const ContourView &cont_src, const ContourView &cont_tgt) {
  static bool checkSim(const ContourView &cont_src, const ContourView &cont_tgt) {
    // very loose
    // TODO: more rigorous criteria (fewer branch, faster speed)
//    std::pair<Eigen::Isometry2d, bool> ret(Eigen::Isometry2d(), false);
    bool ret = false;
    // 1. area, 2.3. eig, 4. com;
    if (diff_perc(cont_src.cell_cnt_, cont_tgt.cell_cnt_, 0.2)) {
//      printf("\tCell cnt not pass.\n");
      return ret;
    }

    if (std::max(cont_src.cell_cnt_, cont_tgt.cell_cnt_) > 15 &&
        diff_delt(cont_src.vol3_mean_, cont_tgt.vol3_mean_, 0.3)) {
//      printf("\tAvg height not pass.\n");
      return ret;
    }

    if (std::max(cont_src.eig_vals_(1), cont_tgt.eig_vals_(1)) > 2.0 &&
        diff_perc(std::sqrt(cont_src.eig_vals_(1)), std::sqrt(cont_tgt.eig_vals_(1)), 0.2)) {
//      printf("\tBig eigval not pass.\n");
      return ret;
    }

    if (std::max(cont_src.eig_vals_(0), cont_tgt.eig_vals_(0)) > 1.0 &&
        diff_perc(std::sqrt(cont_src.eig_vals_(0)), std::sqrt(cont_tgt.eig_vals_(0)), 0.2)) {
//      printf("\tSmall eigval not pass.\n");
      return ret;
    }

    if (std::max((cont_src.com_ - cont_src.pos_mean_).norm(), (cont_tgt.com_ - cont_tgt.pos_mean_).norm()) > 0.5 &&
        diff_perc((cont_src.com_ - cont_src.pos_mean_).norm(), (cont_tgt.com_ - cont_tgt.pos_mean_).norm(), 0.25)) {
//      printf("\tCom radius not pass.\n");
      return ret;
    }

    ret = true;
    return ret;
  }

  // TODO: 4. add two contours. Only statistical parts are useful
  static ContourView addContourStat(const ContourView &cont1, const ContourView &cont2) {
    CHECK_EQ(cont1.level_, cont2.level_);
    ContourView res(cont1.level_, cont1.h_min_, cont1.h_max_, cont1.aabb_, nullptr);
    res.cell_cnt_ = cont1.cell_cnt_ + cont2.cell_cnt_;
    res.cell_pos_sum_ = cont1.cell_pos_sum_ + cont2.cell_pos_sum_;
    res.cell_pos_tss_ = cont1.cell_pos_tss_ + cont2.cell_pos_tss_;
    res.cell_vol3_ = cont1.cell_vol3_ + cont2.cell_vol3_;
    res.cell_vol3_torq_ = cont1.cell_vol3_torq_ + cont2.cell_vol3_torq_;
    res.calcStatVals();
    return res;
  }

  // getter setter
//  int getArea() const {
//    return cell_cnt_;
//  }

//  void addChildren(std::shared_ptr<ContourView> &chd) {
//    children_.push_back(chd);
//  }

  // auxiliary functions
  // 1. get the position of all contour pixels
  std::vector<std::vector<int>> getContPixelPos() const {
    return {};
  }

//  // 2. visualize contour/slice
//  void displayContour(const std::string &fpath) const {
//
//  }

  inline M2D getManualCov() const {
    return eig_vecs_ * eig_vals_.asDiagonal() * eig_vecs_.transpose();
  }

};


#endif //CONT2_CONTOUR_H
