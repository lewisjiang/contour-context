//
// Created by lewis on 5/5/22.
//

#ifndef CONT2_CONTOUR_MNG_H
#define CONT2_CONTOUR_MNG_H

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <bitset>
#include "cont2/contour.h"
#include "tools/algos.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <opencv2/core/mat.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

//// For SURF:
//#include "opencv2/features2d.hpp"
//#include "opencv2/xfeatures2d.hpp"

#include <utility>

#include "tools/bm_util.h"

using KeyFloatType = float; // retrieval key's float number type
//using RetrievalKey = Eigen::Matrix<KeyFloatType, 5, 1>;

template<size_t sz>
struct ArrayAsKey {
  enum {
    SizeAtCompileTime = sz
  };
//  static constexpr size_t SizeAtCompileTime = sz;  // undefined reference when linking
  KeyFloatType vals[sz]{};

  KeyFloatType *data() {
    return vals;
  }

  KeyFloatType &operator()(size_t i) { return vals[i]; }

  const KeyFloatType &operator()(size_t i) const { return vals[i]; }

  KeyFloatType &operator[](size_t i) { return vals[i]; }

  const KeyFloatType &operator[](size_t i) const { return vals[i]; }

  ArrayAsKey<sz> operator-(ArrayAsKey<sz> const &obj) const {
    ArrayAsKey<sz> res;
    for (int i = 0; i < sz; i++)
      res.vals[i] = vals[i] - obj.vals[i];
    return res;
  }

  void setZero() {
    std::fill(vals, vals + SizeAtCompileTime, KeyFloatType(0));
  }

  size_t size() const {
    return sz;
  }

  KeyFloatType sum() const {
    KeyFloatType ret(0);
    for (const auto &dat: vals)
      ret += dat;
    return ret;
  }

  KeyFloatType squaredNorm() const {
    KeyFloatType ret(0);
    for (const auto &dat: vals)
      ret += dat * dat;
    return ret;
  }
};

const int RET_KEY_DIM = 10;
using RetrievalKey = ArrayAsKey<RET_KEY_DIM>;

struct ContourManagerConfig {
  std::vector<float> lv_grads_;  // n marks, n+1 levels
  //
  float reso_row_ = 2.0f, reso_col_ = 2.0f;
  int n_row_ = 100, n_col_ = 100;
  float lidar_height_ = 2.0f;  // ground assumption
  float blind_sq_ = 9.0f;

  int cont_cnt_thres_ = 5; // the cell count threshold dividing a shaped blob from a point
  int min_cont_key_cnt_ = 9;  // minimal the cell count to calculate a valid key around an anchor contour
};

const int BITS_PER_LAYER = 64;
const int DIST_BIN_LAYERS[] = {1, 2, 3, 4};  // the layers for the dist key
const int NUM_BIN_KEY_LAYER = sizeof(DIST_BIN_LAYERS) / sizeof(int);

struct ConstellationPair {  // given a pair of ContourManager, this records the seq of 2 matched contours at certain level
  int level{};
  int seq_src{};
  int seq_tgt{};

  ConstellationPair(int l, int s, int t) : level(l), seq_src(s), seq_tgt(t) {}
};

struct BCI { //binary constellation identity
  struct RelativePoint {  // a point/star seen from an anchor contour
    int level{};
    int seq{};
    float r{};
    float theta{};

    RelativePoint(int l, int a, float b, float c) : level(l), seq(a), r(b), theta(c) {}

//    RelativePoint() = default;
  };

  struct OriePairComp {
    int level;  // the level at which the pairing occurs
    int neigh_src;
    int neigh_tgt;
    float orie_diff;

    OriePairComp(int l, int s, int t, float o) : level(l), neigh_src(s), neigh_tgt(t), orie_diff(o) {}
  };

  // Four member variable
  std::bitset<BITS_PER_LAYER * NUM_BIN_KEY_LAYER> dist_bin_;
  std::map<u_int16_t, std::vector<RelativePoint>> dist_bit_neighbors_;  // {bit position in the bit vector: [neighbours point info, ...]}
  int piv_seq_, level_;  // level and seq of the anchor

  explicit BCI(int seq, int lev) : dist_bin_(0), piv_seq_(seq), level_(lev) {}

  // check the similarity of two BCI in terms of hidden constellation
  static int checkConstellSim(const BCI &src, const BCI &tgt, std::vector<ConstellationPair> &constell) {
    DCHECK_EQ(src.level_, tgt.level_);
    std::bitset<BITS_PER_LAYER * NUM_BIN_KEY_LAYER> res1, res2, res3;
    res1 = src.dist_bin_ & tgt.dist_bin_;
    res2 = (src.dist_bin_ << 1) & tgt.dist_bin_;
    res3 = (src.dist_bin_ >> 1) & tgt.dist_bin_;
    int ovlp1 = res1.count(), ovlp2 = res2.count(), ovlp3 = res3.count();
    int ovlp_sum = ovlp1 + ovlp2 + ovlp3;
    int max_ele = std::max(ovlp1, std::max(ovlp2, ovlp3));

//    std::cout << src.dist_bin_ << std::endl << tgt.dist_bin_ << std::endl;

    // the anchors are assumed to be matched
    if (ovlp_sum >= 10 && max_ele >= 5) {
      // check the angular for constellation
      std::vector<OriePairComp> potential_pairs;

      for (u_int16_t b = 1; b < BITS_PER_LAYER * NUM_BIN_KEY_LAYER - 1; b++) {
        if (tgt.dist_bin_[b])
          for (const auto &rp2: tgt.dist_bit_neighbors_.at(b)) {
            if (res1[b])  // .test(pos) will throw out-of-range exception
              for (const auto &rp1: src.dist_bit_neighbors_.at(b)) {
                DCHECK_EQ(rp1.level, rp2.level);
                potential_pairs.emplace_back(rp1.level, rp1.seq, rp2.seq, rp2.theta - rp1.theta);
              }

            if (res2[b])
              // align after shift left, but `bitset<> x[0]` starts from right, so = origin-1
              for (const auto &rp1: src.dist_bit_neighbors_.at(b - 1)) {
                DCHECK_EQ(rp1.level, rp2.level);
                potential_pairs.emplace_back(rp1.level, rp1.seq, rp2.seq, rp2.theta - rp1.theta);
              }

            if (res3[b])
              for (const auto &rp1: src.dist_bit_neighbors_.at(b + 1)) {
                DCHECK_EQ(rp1.level, rp2.level);
                potential_pairs.emplace_back(rp1.level, rp1.seq, rp2.seq, rp2.theta - rp1.theta);
              }
          }
      }
      // potential_pairs.size() must >= ovlp_sum
      for (auto &x: potential_pairs)
        clampAng<float>(x.orie_diff);

      std::sort(potential_pairs.begin(), potential_pairs.end(), [&](const OriePairComp &a, const OriePairComp &b) {
        return a.orie_diff < b.orie_diff;
      });

      const float angular_range = M_PI / 16; // 0.2 rad, 11 deg
      const int thres_in_range = 5;  // the min number of pairs in range that assumed to be the true delta theta
      int max_in_range_beg = 0, max_in_range = 1, pot_sz = potential_pairs.size(), p1 = 0, p2 = 0;
      while (p1 < pot_sz) {
        if (potential_pairs[p2 % pot_sz].orie_diff - potential_pairs[p1].orie_diff + 2 * M_PI * int(p2 / pot_sz) >
            angular_range)
          p1++;
        else {
          if (p2 - p1 + 1 > max_in_range) {
            max_in_range = p2 - p1 + 1;
            max_in_range_beg = p1;
          }
          p2++;
        }
      }

      if (max_in_range < thres_in_range)
        return -2; // ret code -2: not enough pairs with matched dist pass the angular check

      constell.clear();
      constell.reserve(max_in_range + 1);

      // TODO: solve potential one-to-many matching ambiguity
      for (int i = max_in_range_beg; i < max_in_range + max_in_range_beg; i++) {
        constell.emplace_back(potential_pairs[i % pot_sz].level, potential_pairs[i % pot_sz].neigh_src,
                              potential_pairs[i % pot_sz].neigh_tgt);
      }
      constell.emplace_back(src.level_, src.piv_seq_, tgt.piv_seq_);  // the pivots are also a pair.

      // the sort is for human readability
      std::sort(constell.begin(), constell.end(), [&](const ConstellationPair &a, const ConstellationPair &b) {
        if (a.level == b.level)
          return a.seq_src < b.seq_src;
        return a.level < b.level;
      });

      return constell.size();


    } else {
      return -1; // ret code -1: not passing dist binary check
    }
  }
};


// manage the collection of contours in a scan
class ContourManager {
  // configuration
  const ContourManagerConfig cfg_;
  const float VAL_ABS_INF_ = 1e3;

  // property
  float x_max_, x_min_, y_max_, y_min_;
  std::string str_id_;
  int int_id_;

  // data
  std::vector<std::vector<std::shared_ptr<ContourView>>> cont_views_;  // TODO: use a parallel vec of vec for points?
  std::vector<int> layer_cell_cnt_;  // total number of cells in each layer/level
  std::vector<std::vector<RetrievalKey>> layer_keys_;  // the key of each layer
  std::vector<std::vector<BCI>> layer_key_bcis_;

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
    V2F continuous_rc(p_in_l.x() / cfg_.reso_row_ + cfg_.n_row_ / 2 - 0.5f,
                      p_in_l.y() / cfg_.reso_col_ + cfg_.n_col_ / 2 - 0.5f);
    return continuous_rc;
  }

  void makeContourRecursiveHelper(const cv::Rect &cc_roi, const cv::Mat1b &cc_mask, int level,
                                  const std::shared_ptr<ContourView> &parent);

public:
  explicit ContourManager(const ContourManagerConfig &config, int int_id) : cfg_(config), int_id_(int_id) {
    CHECK(cfg_.n_col_ % 2 == 0);
    CHECK(cfg_.n_row_ % 2 == 0);
    DCHECK(!cfg_.lv_grads_.empty());

    x_min_ = -(cfg_.n_row_ / 2) * cfg_.reso_row_;
    x_max_ = -x_min_;
    y_min_ = -(cfg_.n_col_ / 2) * cfg_.reso_col_;
    y_max_ = -y_min_;

    bev_ = cv::Mat::ones(cfg_.n_row_, cfg_.n_col_, CV_32F) * (-VAL_ABS_INF_);
//    std::cout << bev_ << std::endl;
    c_height_position_ = std::vector<std::vector<V2F>>(cfg_.n_row_, std::vector<V2F>(cfg_.n_col_, V2F::Zero()));
    cont_views_.resize(cfg_.lv_grads_.size());
    layer_cell_cnt_.resize(cfg_.lv_grads_.size());
    layer_keys_.resize(cfg_.lv_grads_.size());
    layer_key_bcis_.resize(cfg_.lv_grads_.size());
  }

  template<typename PointType>
  ///
  /// \tparam PointType
  /// \param ptr_gapc
  /// \param str_id
  void makeBEV(typename pcl::PointCloud<PointType>::ConstPtr &ptr_gapc, std::string str_id = "") {  //
    CHECK(ptr_gapc);
    CHECK_GT(ptr_gapc->size(), 10);

    // Downsample before using?
    for (const auto &pt: ptr_gapc->points) {
      std::pair<int, int> rc = hashPointToImage<PointType>(pt);
      if (rc.first > 0) {
        float height = cfg_.lidar_height_ + pt.z;
        if (bev_(rc.first, rc.second) < height) {
          bev_(rc.first, rc.second) = height;
          c_height_position_[rc.first][rc.second] = pointToContRowCol(V2F(pt.x, pt.y));  // same coord as row and col
        }
        max_bin_val_ = max_bin_val_ < height ? height : max_bin_val_;
        min_bin_val_ = min_bin_val_ > height ? height : min_bin_val_;
      }
    }
    printf("Max/Min bin height: %f %f\n", max_bin_val_, min_bin_val_);
    if (!str_id.empty())
      str_id_ = std::move(str_id);
    else
      str_id_ = std::to_string(ptr_gapc->header.stamp);

    cv::Mat mask, view;
    inRange(bev_, cv::Scalar::all(0), cv::Scalar::all(max_bin_val_), mask);
    normalize(bev_, view, 0, 255, cv::NORM_MINMAX, -1, mask);
    cv::imwrite("cart_context-" + str_id_ + ".png", view);
  }

  void makeContoursRecurs() {
    cv::Rect full_bev(0, 0, bev_.cols, bev_.rows);
    visualization = cv::Mat::zeros(cfg_.n_row_, cfg_.n_col_, CV_8U);

    TicToc clk;
    makeContourRecursiveHelper(full_bev, cv::Mat1b(1, 1), 0, nullptr);
    std::cout << "Time makecontour: " << clk.toc() << std::endl;

    for (int i = 0; i < cont_views_.size(); i++) {
      std::sort(cont_views_[i].begin(), cont_views_[i].end(),
                [&](const std::shared_ptr<ContourView> &p1, const std::shared_ptr<ContourView> &p2) -> bool {
                  return p1->cell_cnt_ > p2->cell_cnt_;
                });   // bigger contours first. Or heavier first?
      layer_cell_cnt_[i] = 0;
      for (int j = 0; j < cont_views_[i].size(); j++) {
        layer_cell_cnt_[i] += cont_views_[i][j]->cell_cnt_;
      }
    }

//    /// exp: find centers and calculate SURF at these places. Format: cv::Point (e.g. cv::x, cv::y)
//    for (int i = 0; i < std::min(10, (int) cont_views_[1].size()); i++) {
//      printf("%7.4f, %7.4f,\n", cont_views_[1][i]->pos_mean_.y(), cont_views_[1][i]->pos_mean_.x());
//    }

    /// make retrieval keys
//    // case 1: traditional key making: from top-two sized contours
//    const int id_firsts = 4; // combination of the first # will be permuated to calculate keys
//    for (int ll = 0; ll < cfg_.lv_grads_.size(); ll++) {
//      for (int id0 = 0; id0 < id_firsts; id0++) {
//        for (int id1 = id0 + 1; id1 < id_firsts; id1++) {
//          RetrievalKey key;
//          key.setZero();
//          if (cont_views_[ll].size() > id1 && cont_views_[ll][id0]->cell_cnt_ > cfg_.cont_cnt_thres_ &&
//              cont_views_[ll][id1]->cell_cnt_ > cfg_.cont_cnt_thres_) { // TODO: make multiple keys for each level
//
//            if (RET_KEY_DIM == 6) {
//              // key dim = 6
//              key(0) = std::sqrt(cont_views_[ll][id0]->cell_cnt_);
//              key(1) = std::sqrt(cont_views_[ll][id1]->cell_cnt_);
//              V2D cc_line = cont_views_[ll][id0]->pos_mean_ - cont_views_[ll][id1]->pos_mean_;
//              key(2) = cc_line.norm();
//
//              // distribution of projection perp to cc line
//              cc_line.normalize();
//              V2D cc_perp(-cc_line.y(), cc_line.x());
//
////        // case1: use cocentic distribution
////        M2D new_cov = (cont_views_[ll][id0]->getManualCov() * (cont_views_[ll][id0]->cell_cnt_ - 1) +
////                       cont_views_[ll][id1]->getManualCov() * (cont_views_[ll][id1]->cell_cnt_ - 1)) /
////                      (cont_views_[ll][id0]->cell_cnt_ + cont_views_[ll][id1]->cell_cnt_ - 1);
//              // case2: use relative translation preserving distribution
//              M2D new_cov = ContourView::addContourStat(*cont_views_[ll][id0], *cont_views_[ll][id1]).getManualCov();
//
//              key(3) = std::sqrt(cc_perp.transpose() * new_cov * cc_perp);
//
//              // distribution of projection to cc line
//              key(4) = std::sqrt(cc_line.transpose() * new_cov * cc_line);
//
//              // the max eigen value of the first ellipse
//              key(5) = std::sqrt(cont_views_[ll][id0]->eig_vals_(1));
//            } else if (RET_KEY_DIM == 11) {
//              // key dim = 11
//              V2D cc_line = cont_views_[ll][id0]->pos_mean_ - cont_views_[ll][id1]->pos_mean_;
//              key(0) = cc_line.norm();
//
//              // the max eigen value of the first ellipse
//              key(1) = std::sqrt(cont_views_[ll][id0]->eig_vals_(1));
//              key(2) = std::sqrt(cont_views_[ll][id1]->eig_vals_(1));
//
//              // the strip descriptors
//              for (int i = 0; i < 4; i++) {
//                key(3 + i * 2) = cont_views_[ll][id0]->strip_width_[i];
//                key(3 + i * 2 + 1) = cont_views_[ll][id1]->strip_width_[i];
//              }
//            } else if (RET_KEY_DIM == 9) {
//              // key dim = 9
//              V2D cc_line = cont_views_[ll][id0]->pos_mean_ - cont_views_[ll][id1]->pos_mean_;
//              key(0) = cc_line.norm();
//
//              // the max eigen value of the first ellipse
//              // the strip descriptors, area
//              for (int i = 0; i < 4; i++) {
//                key(1 + i * 2) =
//                    cont_views_[ll][id0]->strip_width_[i] * std::sqrt(cont_views_[ll][id0]->eig_vals_(1)) / 4;
//                key(1 + i * 2 + 1) =
//                    cont_views_[ll][id1]->strip_width_[i] * std::sqrt(cont_views_[ll][id1]->eig_vals_(1)) / 4;
//              }
//            }
//
//          }
//
//
//          layer_keys_[ll].emplace_back(key);
//        }
//      }
//    }

    /// case 2: new key making: from a pivot contour
    const int piv_firsts = 6;
    const int dist_firsts = 10;
    for (int ll = 0; ll < cfg_.lv_grads_.size(); ll++) {
//      cv::Mat mask;
//      cv::threshold(bev_, mask, cfg_.lv_grads_[ll], 123,
//                    cv::THRESH_TOZERO); // mask is same type and dimension as bev_
      int accumulate_cell_cnt = 0;
      for (int i = 0; i < piv_firsts; i++) {
        RetrievalKey key;
        key.setZero();

        BCI bci(i, ll);

        if (cont_views_[ll].size() > i)
          accumulate_cell_cnt += cont_views_[ll][i]->cell_cnt_;

        if (cont_views_[ll].size() > i && cont_views_[ll][i]->cell_cnt_ >= cfg_.min_cont_key_cnt_) {

          V2F v_cen = cont_views_[ll][i]->pos_mean_.cast<float>();
          int r_cen = int(v_cen.x()), c_cen = int(v_cen.y());
          int r_min = std::max(0, r_cen - 10), r_max = std::min(cfg_.n_row_ - 1, r_cen + 10);
          int c_min = std::max(0, c_cen - 10), c_max = std::min(cfg_.n_col_ - 1, c_cen + 10);

          int num_bins = 7;
          KeyFloatType bin_len = 10.0 / num_bins;
          std::vector<KeyFloatType> ring_bins(num_bins, 0);

          int div_per_bin = 5;
          std::vector<KeyFloatType> discrete_divs(div_per_bin * num_bins, 0);
          KeyFloatType div_len = 10.0 / (num_bins * div_per_bin);
          int cnt_point = 0;

          ContourView cv_tmp(ll, cfg_.lv_grads_[ll], 999.9, cv::Rect(), nullptr); // for case 3

          for (int rr = r_min; rr <= r_max; rr++) {
            for (int cc = c_min; cc <= c_max; cc++) {
              KeyFloatType dist = (c_height_position_[rr][cc] - v_cen).norm();

              // case 1: ring, height, 7
//              if (dist < 10 - 1e-2 && bev_(rr, cc) > cfg_.lv_grads_[0]) {  // add gaussian to bins
//                int bin_idx = int(dist / bin_len);
//                ring_bins[bin_idx] += bev_(rr, cc);    // no gaussian
//              }

              // case 2: gmm, normalized
              if (dist < 10 - 1e-2 && bev_(rr, cc) > cfg_.lv_grads_[ll]) {
                cnt_point++;
                for (int div_idx = 0; div_idx < num_bins * div_per_bin; div_idx++)
                  discrete_divs[div_idx] += gaussPDF<KeyFloatType>(div_idx * div_len + 0.5 * div_len, dist, 1.0);
              }

//              // case 3: using another ellipse
//              if (dist < 10 - 1e-2 && bev_(rr, cc) > cfg_.lv_grads_[ll]) {
//                cv_tmp.runningStatsF(c_height_position_[rr][cc].x(), c_height_position_[rr][cc].y(), bev_(rr, cc));
//              }

            }
          }

          // case 2: gmm, normalized
          for (int b = 0; b < num_bins; b++) {
            for (int d = 0; d < div_per_bin; d++) {
              ring_bins[b] += discrete_divs[b * div_per_bin + d];
            }
            ring_bins[b] *= bin_len / std::sqrt(cnt_point);
          }

//          // case 3: using another ellipse
//          cv_tmp.calcStatVals();



          // TODO: make the key generation from one contour more distinctive
//          key(0) = std::sqrt(cont_views_[ll][i]->eig_vals_(1));  // max eigen value
//          key(1) = std::sqrt(cont_views_[ll][i]->eig_vals_(0));  // min eigen value
//          key(2) = (cont_views_[ll][i]->pos_mean_ - cont_views_[ll][i]->com_).norm();

          key(0) =
              std::sqrt(cont_views_[ll][i]->eig_vals_(1) * cont_views_[ll][i]->cell_cnt_);  // max eigen value * cnt
          key(1) =
              std::sqrt(cont_views_[ll][i]->eig_vals_(0) * cont_views_[ll][i]->cell_cnt_);  // min eigen value * cnt
//          key(2) = (cont_views_[ll][i]->pos_mean_ - cont_views_[ll][i]->com_).norm() *
//                   std::sqrt(cont_views_[ll][i]->cell_cnt_);
//                   (cont_views_[ll][i]->cell_cnt_);
          key(2) = std::sqrt(accumulate_cell_cnt);


          // case 1,2:
          for (int nb = 0; nb < num_bins; nb++) {
//            key(3 + nb) = ring_bins[nb];
//            key(3 + nb) = ring_bins[nb] / (M_PI * (2 * nb + 1) * bin_len);  // density on the ring
            key(3 + nb) = ring_bins[nb];  // case 2.1: count on the ring
//            key(3 + nb) = ring_bins[nb] / (2 * nb + 1);  // case 2.2: kind of density on the ring
          }

//          // case 3:
//          key(3) = std::sqrt(cont_views_[ll][i]->cell_cnt_);
//
//          key(4) = std::sqrt(cv_tmp.eig_vals_(1) * cv_tmp.cell_cnt_);
//          key(5) = std::sqrt(cv_tmp.eig_vals_(0) * cv_tmp.cell_cnt_);
//          key(6) = (cv_tmp.pos_mean_ - cv_tmp.com_).norm() * cv_tmp.cell_cnt_;
//          key(7) = std::sqrt(cv_tmp.cell_cnt_);
//
//          V2D cc_line = cont_views_[ll][i]->pos_mean_ - cv_tmp.pos_mean_;
//          key(8) = cc_line.norm();
//          key(9) = std::sqrt(std::abs(cv_tmp.cell_cnt_ - cont_views_[ll][i]->cell_cnt_));



          // hash dists and angles of the neighbours around the anchor/pivot to bit keys
          // hard coded
          for (int bl = 0; bl < NUM_BIN_KEY_LAYER; bl++) {
            int bit_offset = bl * BITS_PER_LAYER;
            for (int j = 0; j < std::min(dist_firsts, (int) cont_views_[DIST_BIN_LAYERS[bl]].size()); j++) {
              if (j != i) {
                V2D vec_cc =
                    cont_views_[DIST_BIN_LAYERS[bl]][j]->pos_mean_ - cont_views_[ll][i]->pos_mean_;
                float tmp_dist = vec_cc.norm();

                if (tmp_dist > (BITS_PER_LAYER - 1) * 1.01 + 5.43 - 1e-3 // the last bit of layer sector is always 0
                    || tmp_dist <= 5.43)  // TODO: nonlinear mapping?
                  continue;

                float tmp_orie = std::atan2(vec_cc.y(), vec_cc.x());
                int dist_idx = std::min(std::floor((tmp_dist - 5.43) / 1.01), BITS_PER_LAYER - 1.0) + bit_offset;
                bci.dist_bin_.set(dist_idx, true);
                bci.dist_bit_neighbors_[dist_idx].emplace_back(DIST_BIN_LAYERS[bl], j, tmp_dist, tmp_orie);
              }
            }
          }

        }
//        if(key.sum()!=0)
        layer_key_bcis_[ll].emplace_back(bci);  // even invalid keys are recorded.
        layer_keys_[ll].emplace_back(key);  // even invalid keys are recorded.
      }
    }

    for (int ll = 0; ll < cfg_.lv_grads_.size(); ll++) {
      DCHECK_EQ(layer_keys_[ll].size(), piv_firsts);
      DCHECK_EQ(layer_key_bcis_[ll].size(), piv_firsts);
    }

    // print top 2 features in each
//    for (int i = 0; i < cfg_.lv_grads_.size(); i++) {
//      printf("\nLevel %d top 2 statistics:\n", i);
//      for (int j = 0; j < std::min(2lu, cont_views_[i].size()); j++) {
//        printf("# %d:\n", j);
//        std::cout << "Cell count " << cont_views_[i][j]->cell_cnt_ << std::endl;
//        std::cout << "Eigen Vals " << cont_views_[i][j]->eig_vals_.transpose() << std::endl;
//        std::cout << "com - cent " << (cont_views_[i][j]->com_ - cont_views_[i][j]->pos_mean_).transpose() << std::endl;
//        std::cout << "Total vol  " << cont_views_[i][j]->cell_vol3_ << std::endl;
//      }
//    }

    // save statistics of this scan:
    std::string fpath = std::string(PJSRCDIR) + "/results/contours_orig-" + str_id_ + ".txt";
    saveContours(fpath, cont_views_);

//    cv::imwrite("cart_context-mask-" + std::to_string(3) + "-" + str_id_ + "rec.png", visualization);
//    for (const auto &x: cont_views_) {
//      printf("level size: %lu\n", x.size());
//    }
  }

  // save accumulated contours to a file that is readable to the python script
  void saveAccumulatedContours(int top_n) const {
    std::vector<std::vector<std::shared_ptr<ContourView>>> new_cont_views;
    new_cont_views.resize(cont_views_.size());
    for (int ll = 0; ll < cfg_.lv_grads_.size(); ll++) {
      for (int i = 0; i < std::min(top_n, (int) cont_views_[ll].size()); i++) {
        if (i == 0)
          new_cont_views[ll].emplace_back(std::make_shared<ContourView>(*cont_views_[ll][i]));
        else {
          new_cont_views[ll].emplace_back(std::make_shared<ContourView>(
              ContourView::addContourStat(*new_cont_views[ll].back(), *cont_views_[ll][i])));
        }
      }
    }
    std::string fpath = std::string(PJSRCDIR) + "/results/contours_accu-" + str_id_ + ".txt";
    saveContours(fpath, new_cont_views);

  }

  // experimental: show dists from one contour to several others
  void expShowDists(int level, int pivot, int top_n) {
    CHECK_LT(level, cfg_.lv_grads_.size());
    CHECK_LT(pivot, cont_views_[level].size());
    printf("Level %d, pivot No.%d distances:\n", level, pivot);
    std::vector<std::pair<int, double>> dists;
    for (int i = 0; i < std::min(top_n, (int) cont_views_[level].size()); i++)
      if (i != pivot)
        dists.emplace_back(i, (cont_views_[level][i]->pos_mean_ - cont_views_[level][pivot]->pos_mean_).norm());

    std::sort(dists.begin(), dists.end(), [&](const std::pair<int, double> &a, const std::pair<int, double> &b) {
      return a.second < b.second;
    });
    for (const auto &pr: dists) {
      printf("%2d: %7.4f, ", pr.first, pr.second);
    }
    printf("\n");
  }

  // experimental: show dists from one contour to several others
  void expShowBearing(int level, int pivot, int top_n) {
    CHECK_LT(level, cfg_.lv_grads_.size());
    CHECK_LT(pivot, cont_views_[level].size());
    printf("Level %d, pivot No.%d orientations:\n", level, pivot);
    std::vector<std::pair<int, double>> bearings;
    bool first_set = false;
    V2D vec0(0, 0);
    for (int i = 0; i < std::min(top_n, (int) cont_views_[level].size()); i++) {
      if (i != pivot) {
        if (!first_set) {
          bearings.emplace_back(i, 0);
          first_set = true;
          vec0 = (cont_views_[level][i]->pos_mean_ - cont_views_[level][pivot]->pos_mean_).normalized();
        } else {
          V2D vec1 = (cont_views_[level][i]->pos_mean_ - cont_views_[level][pivot]->pos_mean_).normalized();
          double ang = std::atan2(vec0.x() * vec1.y() - vec0.y() * vec1.x(), vec0.dot(vec1));
          bearings.emplace_back(i, ang);
        }
      }
    }

    std::sort(bearings.begin(), bearings.end(), [&](const std::pair<int, double> &a, const std::pair<int, double> &b) {
      return a.second < b.second;
    });
    for (const auto &pr: bearings) {
      printf("%2d: %7.4f, ", pr.first, pr.second);
    }
    printf("\n");
  }

  void makeContours();

  // util functions
  // 1. save all contours' statistical data into a text file
  static void
  saveContours(const std::string &fpath, const std::vector<std::vector<std::shared_ptr<ContourView>>> &cont_views);

  // 2. save a layer of contours to image
  void saveContourImage(const std::string &fpath, int level) const;

  cv::Mat getContourImage(int level) const {
    cv::Mat mask;
    cv::threshold(bev_, mask, cfg_.lv_grads_[level], 123,
                  cv::THRESH_TOZERO); // mask is same type and dimension as bev_
    cv::Mat normalized_layer, mask_u8;
    cv::normalize(mask, normalized_layer, 0, 255, cv::NORM_MINMAX, CV_8U);  // dtype=-1 (default): same type as input
    return normalized_layer;
  }

  inline ContourManagerConfig getConfig() const {
    return cfg_;
  }

  // TODO: get retrieval key of a scan
  const std::vector<RetrievalKey> &getRetrievalKey(int level) const {
    DCHECK_GE(level, 0);
    DCHECK_GT(cont_views_.size(), level);
    return layer_keys_[level];
  }

  const RetrievalKey &getRetrievalKey(int level, int seq) const {
    DCHECK_GE(level, 0);
    DCHECK_GT(cont_views_.size(), level);
    DCHECK_LT(seq, layer_keys_[level].size());
    return layer_keys_[level][seq];
  }

  // get bci
  const std::vector<BCI> &getBCI(int level) const {
    DCHECK_GE(level, 0);
    DCHECK_GT(cont_views_.size(), level);
    return layer_key_bcis_[level];
  }

  const BCI &getBCI(int level, int seq) const {
    DCHECK_GE(level, 0);
    DCHECK_GT(cont_views_.size(), level);
    DCHECK_LT(seq, layer_key_bcis_[level].size());
    return layer_key_bcis_[level][seq];
  }

  inline std::string getStrID() const {
    return str_id_;
  }

  inline int getIntID() const {
    return int_id_;
  }

  // TODO: check if contours in two scans can be accepted as from the same heatmap, and return the transform
  // TODO: when retrieval key contains the combination, we should only look into that combination.
  // T_tgt = T_delta * T_src
  static std::pair<Eigen::Isometry2d, bool> calcScanCorresp(const ContourManager &src, const ContourManager &tgt);

  // T_tgt = T_delta * T_src
  ///
  /// \param src
  /// \param tgt
  /// \param levels
  /// \param ids_src The indices of the matched contours at \param{levels}, one on one correspondence with ids_tgt
  /// \param ids_tgt
  /// \param sim_idx if return true, this contains the index of ids_{src|tgt} survived pairwise similarity check. Values
  ///     taken from [0, ids_tgt.size()).
  /// \return the transform and the number of matched...
  static std::pair<Eigen::Isometry2d, int>  // int: human readability
  calcScanCorresp(const ContourManager &src, const ContourManager &tgt, const std::vector<ConstellationPair> &cstl,
                  std::vector<int> &sim_idx, const int min_pair) {
    // cross level consensus (CLC)
    // The rough constellation should have been established.
    DCHECK_EQ(src.cont_views_.size(), tgt.cont_views_.size());

    std::pair<Eigen::Isometry2d, int> ret{};

    sim_idx.clear();
    int num_sim = 0;
    // 1. check individual sim
    for (int i = 0; i < cstl.size(); i++) {
      if (ContourView::checkSim(*src.cont_views_[cstl[i].level][cstl[i].seq_src],
                                *tgt.cont_views_[cstl[i].level][cstl[i].seq_tgt]))
        sim_idx.push_back(i);
    }

    ret.second = sim_idx.size();
    if (sim_idx.size() < min_pair)
      return ret;  // TODO: use cross level consensus to find more possible matched pairs

    // 2. check orientation
    // 2.1 get major axis direction
    V2D shaft_src(0, 0), shaft_tgt(0, 0);
    for (int i = 1; i < std::min((int) sim_idx.size(), 10); i++) {
      for (int j = 0; j < i; j++) {
        V2D curr_shaft = src.cont_views_[cstl[sim_idx[i]].level][cstl[sim_idx[i]].seq_src]->pos_mean_ -
                         src.cont_views_[cstl[sim_idx[j]].level][cstl[sim_idx[j]].seq_src]->pos_mean_;
        if (curr_shaft.norm() > shaft_src.norm()) {
          shaft_src = curr_shaft.normalized();
          shaft_tgt = (tgt.cont_views_[cstl[sim_idx[i]].level][cstl[sim_idx[i]].seq_tgt]->pos_mean_ -
                       tgt.cont_views_[cstl[sim_idx[j]].level][cstl[sim_idx[j]].seq_tgt]->pos_mean_).normalized();
        }
      }
    }
    // 2.2 if both src and tgt contour are orientational salient but the orientations largely differ, remove the pair
    num_sim = sim_idx.size();
    for (int i = 0; i < num_sim;) {
      const auto &sc1 = src.cont_views_[cstl[sim_idx[i]].level][cstl[sim_idx[i]].seq_src],
          &tc1 = tgt.cont_views_[cstl[sim_idx[i]].level][cstl[sim_idx[i]].seq_tgt];
      if (sc1->ecc_feat_ && tc1->ecc_feat_) {
        double theta_s = std::acos(shaft_src.transpose() * sc1->eig_vecs_.col(1));   // acos: [0,pi)
        double theta_t = std::acos(shaft_tgt.transpose() * tc1->eig_vecs_.col(1));
        if (diff_delt(theta_s, theta_t, M_PI / 6) && diff_delt(M_PI - theta_s, theta_t, M_PI / 6)) {
          std::swap(sim_idx[i], sim_idx[num_sim - 1]);
          num_sim--;
          continue;
        }
      }
      i++;
    }
    sim_idx.resize(num_sim);
    ret.second = sim_idx.size();
    if (sim_idx.size() < min_pair)
      return ret;  // TODO: use cross level consensus to find more possible matched pairs

    std::sort(sim_idx.begin(), sim_idx.end());  // human readability

    // 3. estimate transform using the data from current level
    Eigen::Matrix<double, 2, Eigen::Dynamic> pointset1; // src
    Eigen::Matrix<double, 2, Eigen::Dynamic> pointset2; // tgt
    pointset1.resize(2, sim_idx.size());
    pointset2.resize(2, sim_idx.size());
    for (int i = 0; i < sim_idx.size(); i++) {
      pointset1.col(i) = src.cont_views_[cstl[sim_idx[i]].level][cstl[sim_idx[i]].seq_src]->pos_mean_;
      pointset2.col(i) = tgt.cont_views_[cstl[sim_idx[i]].level][cstl[sim_idx[i]].seq_tgt]->pos_mean_;
    }

    Eigen::Matrix3d T_delta = Eigen::umeyama(pointset1, pointset2, false);

    printf("Found matched pairs:\n");
    for (int i: sim_idx) {
      printf("\tlev %d, src:tgt  %d: %d\n", cstl[i].level, cstl[i].seq_src, cstl[i].seq_tgt);
    }
    std::cout << "Transform matrix:\n" << T_delta << std::endl;

//    ret.second = true;
    ret.second = sim_idx.size();
    ret.first.setIdentity();
    ret.first.rotate(std::atan2(T_delta(1, 0), T_delta(0, 0)));
    ret.first.pretranslate(T_delta.block<2, 1>(0, 2));
    return ret;
  }

  inline static bool
  checkContPairSim(const ContourManager &src, const ContourManager &tgt, const ConstellationPair &cstl) {
    return ContourView::checkSim(*src.cont_views_[cstl.level][cstl.seq_src],
                                 *tgt.cont_views_[cstl.level][cstl.seq_tgt]);
  }

  static void
  saveMatchedPairImg(const std::string &fpath, const ContourManager &cm1,
                     const ContourManager &cm2) {
    ContourManagerConfig config = cm2.getConfig();

    DCHECK_EQ(config.n_row_, cm1.getConfig().n_row_);
    DCHECK_EQ(config.n_col_, cm1.getConfig().n_col_);
    DCHECK_EQ(cm2.getConfig().n_row_, cm1.getConfig().n_row_);
    DCHECK_EQ(cm2.getConfig().n_col_, cm1.getConfig().n_col_);

    cv::Mat output((config.n_row_ + 1) * config.lv_grads_.size(), config.n_col_ * 2, CV_8U);
    output.setTo(255);

    for (int i = 0; i < config.lv_grads_.size(); i++) {
      cm1.getContourImage(i).copyTo(output(cv::Rect(0, i * config.n_row_ + i, config.n_col_, config.n_row_)));
      cm2.getContourImage(i).copyTo(
          output(cv::Rect(config.n_col_, i * config.n_row_ + i, config.n_col_, config.n_row_)));
    }
    cv::imwrite(fpath, output);
  }


};


#endif //CONT2_CONTOUR_MNG_H
