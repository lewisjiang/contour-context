//
// Created by lewis on 5/5/22.
//

#ifndef CONT2_CONTOUR_MNG_H
#define CONT2_CONTOUR_MNG_H

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <bitset>
#include <set>
#include <map>
#include <string>
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
#include "tools/algos.h"

using KeyFloatType = float; // retrieval key's float number type
//using RetrievalKey = Eigen::Matrix<KeyFloatType, 5, 1>;

template<size_t sz>
struct ArrayAsKey {
  enum {
    SizeAtCompileTime = sz
  };
//  static constexpr size_t SizeAtCompileTime = sz;  // undefined reference when linking
  KeyFloatType array[sz]{};

  KeyFloatType *data() {
    return array;
  }

  KeyFloatType &operator()(size_t i) { return array[i]; }

  const KeyFloatType &operator()(size_t i) const { return array[i]; }

  KeyFloatType &operator[](size_t i) { return array[i]; }

  const KeyFloatType &operator[](size_t i) const { return array[i]; }

  ArrayAsKey<sz> operator-(ArrayAsKey<sz> const &obj) const {
    ArrayAsKey<sz> res;
    for (int i = 0; i < sz; i++)
      res.array[i] = array[i] - obj.array[i];
    return res;
  }

  void setZero() {
    std::fill(array, array + SizeAtCompileTime, KeyFloatType(0));
  }

  size_t size() const {
    return sz;
  }

  KeyFloatType sum() const {
    KeyFloatType ret(0);
    for (const auto &dat: array)
      ret += dat;
    return ret;
  }

  KeyFloatType squaredNorm() const {
    KeyFloatType ret(0);
    for (const auto &dat: array)
      ret += dat * dat;
    return ret;
  }
};

const int RET_KEY_DIM = 10;
using RetrievalKey = ArrayAsKey<RET_KEY_DIM>;

struct ContourManagerConfig {
  std::vector<float> lv_grads_;  // n marks, n+1 levels
  //
//  float reso_row_ = 2.0f, reso_col_ = 2.0f;
//  int n_row_ = 100, n_col_ = 100;
  float reso_row_ = 1.0f, reso_col_ = 1.0f;
  int n_row_ = 150, n_col_ = 150;

  float lidar_height_ = 2.0f;  // ground assumption
  float blind_sq_ = 9.0f;

//  int cont_cnt_thres_ = 5; // the cell count threshold dividing a shaped blob from a point
  int min_cont_key_cnt_ = 9;  // minimal the cell count to calculate a valid key around an anchor contour
  int min_cont_cell_cnt_ = 3; // the minimal cell cnt to consider creating a contour
};

const int16_t BITS_PER_LAYER = 64;
const int8_t DIST_BIN_LAYERS[] = {1, 2, 3, 4};  // the layers for generating the dist key and forming the constellation
const float LAYER_AREA_WEIGHTS[] = {0.3, 0.3, 0.3,
                                    0.1};  // weights for each layer when calculating a normalized "used area percentage"
const int16_t NUM_BIN_KEY_LAYER = sizeof(DIST_BIN_LAYERS) / sizeof(int8_t);


// scores for checks
// i: int, typically for counts. f: float, just represented in int.
union ScoreConstellSim {
  enum {
    SizeAtCompileTime = 3
  };
  int data[SizeAtCompileTime]{};
  struct {
    int i_ovlp_sum;
    int i_ovlp_max_one;
    int i_in_ang_rng;
  };

  inline const int &overall() const {
    // Manually select a thres in the check as the score for the overall check to pass.
    return i_in_ang_rng;
  }

  inline int cnt() const {
    return i_in_ang_rng;
  }

  void print() const {
    printf("%d, %d, %d;", i_ovlp_sum, i_ovlp_max_one, i_in_ang_rng);
  }

  bool strictSmaller(const ScoreConstellSim &b) const {
    for (int i = 0; i < SizeAtCompileTime; i++) {
      if (data[i] >= b.data[i])
        return false;
    }
    return true;
  }
};

union ScorePairwiseSim {
  enum {
    SizeAtCompileTime = 2
  };
  int data[SizeAtCompileTime]{};
  struct {
    int i_indiv_sim;
    int i_orie_sim;
//    int f_area_perc;  // the area score is a weighted sum of used area percentage of all concerning levels, normalized to int(100)
  };

  inline const int &overall() const {
    // Manually select a thres in the check as the min requirement for the overall check to pass.
    return i_orie_sim;
//    return f_area_perc;  // which has the final say?
  }

  inline int cnt() const {
    return i_orie_sim;
  }

  void print() const {
    printf("%d, %d;", i_indiv_sim, i_orie_sim);
  }

  bool strictSmaller(const ScorePairwiseSim &b) const {
    for (int i = 0; i < SizeAtCompileTime; i++) {
      if (data[i] >= b.data[i])
        return false;
    }
    return true;
  }
};

union ScorePostProc {
  enum {
    SizeAtCompileTime = 3
  };
  float data[SizeAtCompileTime]{};

  struct {
    float correlation;
    float area_perc;
    float neg_est_dist;  // 2D trans distance. Record as the negated distance (since the larger the better)
  };

  inline const float &overall() const {
    return correlation;
  }

//  inline int cnt() const {
//    return 0;
//  }

  void print() const {
    printf("%6f, %6f%%, %6fm;", correlation, 100 * area_perc, neg_est_dist);
  }

  bool strictSmaller(const ScorePostProc &b) const {
    for (int i = 0; i < SizeAtCompileTime; i++) {
      if (data[i] >= b.data[i])
        return false;
    }
    return true;
  }
};

union ConstellationPair {  // given a pair of ContourManager, this records the seq of 2 "matched" contours at a certain level
  struct {
    int8_t level;
    int8_t seq_src;
    int8_t seq_tgt;
  };
  int data[1]{};

  ConstellationPair(int8_t l, int8_t s, int8_t t) : level(l), seq_src(s), seq_tgt(t) {}

  bool operator<(const ConstellationPair &a) const {
    return level < a.level || (level == a.level && seq_src < a.seq_src) ||
           (level == a.level && seq_src == a.seq_src && seq_tgt < a.seq_tgt);
  }

  bool operator==(const ConstellationPair &a) const {
    return level == a.level && seq_src == a.seq_src && seq_tgt == a.seq_tgt; // why not just compare data[0]?
  }

};

//! binary constellation identity
struct BCI { //binary constellation identity
  //! a point/star of the constellation seen from an anchor contour
  union RelativePoint {
    struct {
      int8_t level;
      int8_t seq;
      int16_t bit_pos;
      float r;
      float theta;
    };
    int data[3]{};

    RelativePoint(int8_t l, int8_t a, int16_t b, float f1, float f2) : level(l), seq(a), bit_pos(b), r(f1), theta(f2) {}

//    RelativePoint() = default;
  };

  //! potential pairs from 2 constellations that passed the dist check
  union DistSimPair {
    struct {
      float orie_diff;  // the diff of star-anchor orientation
      int8_t seq_src;  // source sequence of the neighbor of the anchor
      int8_t seq_tgt;
      int8_t level;  // the level at which the pairing occurs
    };
    int data[2]{};

    DistSimPair(int8_t l, int8_t s, int8_t t, float o) : level(l), seq_src(s), seq_tgt(t), orie_diff(o) {}
  };

  // Four member variable
  std::bitset<BITS_PER_LAYER * NUM_BIN_KEY_LAYER> dist_bin_;
//  std::map<u_int16_t, std::vector<RelativePoint>> dist_bit_neighbors_;  // {bit position in the bit vector: [neighbours point info, ...]}
  std::vector<RelativePoint> nei_pts_;
  std::vector<uint16_t> nei_idx_segs_;  // index in the `nei_pts_`, [seg[i], seg[i+1]) is a segment with the same dist bit set.
  int8_t piv_seq_, level_;  // level and seq of the anchor/pivot

  explicit BCI(int8_t seq, int8_t lev) : dist_bin_(0), piv_seq_(seq), level_(lev) {}

  /// Check the similarity of two BCI in terms of implicit constellation centered at the anchor
  /// \param src
  /// \param tgt
  /// \param lb low bar thresholds for the checks in this function
  /// \param constell_res
  /// \return the number of pairs of stars that pass the checks (negative numbers are for human readability)
  static ScoreConstellSim checkConstellSim(const BCI &src, const BCI &tgt, const ScoreConstellSim &lb,
                                           std::vector<ConstellationPair> &constell_res) {
    DCHECK_EQ(src.level_, tgt.level_);
    std::bitset<BITS_PER_LAYER * NUM_BIN_KEY_LAYER> and1, and2, and3;
    and1 = src.dist_bin_ & tgt.dist_bin_;
    and2 = (src.dist_bin_ << 1) & tgt.dist_bin_;
    and3 = (src.dist_bin_ >> 1) & tgt.dist_bin_;
    int ovlp1 = and1.count(), ovlp2 = and2.count(), ovlp3 = and3.count();
    int ovlp_sum = ovlp1 + ovlp2 + ovlp3;
    int max_one = std::max(ovlp1, std::max(ovlp2, ovlp3));

    ScoreConstellSim ret;  // return a score object, and check it outside with some bars to determine P/N

    ret.i_ovlp_sum = ovlp_sum;
    ret.i_ovlp_max_one = max_one;

//    std::cout << src.dist_bin_ << std::endl << tgt.dist_bin_ << std::endl;

    // the anchors are assumed to be matched
    if (ovlp_sum >= lb.i_ovlp_sum && max_one >= lb.i_ovlp_max_one) {  // TODO: use config instead of hardcoded
      // check the angular for constellation
      std::vector<DistSimPair> potential_pairs;

      int16_t p11 = 0, p12;
      for (int16_t p2 = 0; p2 < tgt.nei_idx_segs_.size() - 1; p2++) {
        while (p11 < src.nei_idx_segs_.size() - 1 &&
               src.nei_pts_[src.nei_idx_segs_[p11]].bit_pos < tgt.nei_pts_[tgt.nei_idx_segs_[p2]].bit_pos - 1) {
          p11++;
        }

        p12 = p11;

        while (p12 < src.nei_idx_segs_.size() - 1 &&
               src.nei_pts_[src.nei_idx_segs_[p12]].bit_pos <= tgt.nei_pts_[tgt.nei_idx_segs_[p2]].bit_pos + 1) {
          p12++;
        }

        for (int i = tgt.nei_idx_segs_[p2]; i < tgt.nei_idx_segs_[p2 + 1]; i++) {
          for (int j = src.nei_idx_segs_[p11]; j < src.nei_idx_segs_[p12]; j++) {
            const BCI::RelativePoint &rp1 = src.nei_pts_[j], &rp2 = tgt.nei_pts_[i];
//            printf("Adding tgt %d : src %d,   p11:p12 %d, %d\n", i, j, p11, p12);
            DCHECK_EQ(rp1.level, rp2.level);
            DCHECK_LE(std::abs(rp1.bit_pos - rp2.bit_pos), 1);
            potential_pairs.emplace_back(rp1.level, rp1.seq, rp2.seq, rp2.theta - rp1.theta);
          }
        }
      }

      // potential_pairs.size() must >= ovlp_sum
      for (auto &x: potential_pairs)
        clampAng<float>(x.orie_diff);

      std::sort(potential_pairs.begin(), potential_pairs.end(), [&](const DistSimPair &a, const DistSimPair &b) {
        return a.orie_diff < b.orie_diff;
      });

      const float angular_range = M_PI / 16; // 0.2 rad, 11 deg
      int longest_in_range_beg = 0, longest_in_range = 1, pot_sz = potential_pairs.size(), p1 = 0, p2 = 0;
      while (p1 < pot_sz) {
        if (potential_pairs[p2 % pot_sz].orie_diff - potential_pairs[p1].orie_diff + 2 * M_PI * int(p2 / pot_sz) >
            angular_range)
          p1++;
        else {
          if (p2 - p1 + 1 > longest_in_range) {
            longest_in_range = p2 - p1 + 1;
            longest_in_range_beg = p1;
          }
          p2++;
        }
      }

      ret.i_in_ang_rng = longest_in_range;

      if (longest_in_range <
          lb.i_in_ang_rng)  // the min number of pairs in range that assumed to be the true delta theta
        return ret; // ret code -2: not enough pairs with matched dist pass the angular check

      constell_res.clear();
      constell_res.reserve(longest_in_range + 1);

      // TODO: solve potential one-to-many matching ambiguity
      for (int i = longest_in_range_beg; i < longest_in_range + longest_in_range_beg; i++) {
        constell_res.emplace_back(potential_pairs[i % pot_sz].level, potential_pairs[i % pot_sz].seq_src,
                                  potential_pairs[i % pot_sz].seq_tgt);
      }
      constell_res.emplace_back(src.level_, src.piv_seq_, tgt.piv_seq_);  // the pivots are also a pair.

#if HUMAN_READABLE
      // the sort is for human readability
      std::sort(constell_res.begin(), constell_res.end(), [&](const ConstellationPair &a, const ConstellationPair &b) {
        return a.level < b.level || (a.level == b.level) && a.seq_src < b.seq_src;
      });
#endif

      return ret;


    } else {
      return ret; // ret code -1: not passing dist binary check
    }
  }
};

//! 2.5D continuous(float) pixel
union Pixelf {
  struct {
    float row_f;
    float col_f;
    float elev;
  };
  int data[3]{};

  Pixelf(float r, float c, float e) : row_f(r), col_f(c), elev(e) {}

  Pixelf() {
    row_f = -1;
    col_f = -1;
    elev = -1;
  }

  bool operator<(const Pixelf &b) const {
    return row_f < b.row_f;
  }
};

//! manage the collection of contours in a scan
class ContourManager {
  // configuration
  const ContourManagerConfig cfg_;
  const ContourViewStatConfig view_stat_cfg_;
  const float VAL_ABS_INF_ = 1e3;

  // property
  float x_max_, x_min_, y_max_, y_min_;  // for points in the sensor frame, not in the bev frame
  std::string str_id_;
  int int_id_;

  // data
  std::vector<std::vector<std::shared_ptr<ContourView>>> cont_views_;  // TODO: use a parallel vec of vec for points?
  std::vector<std::vector<float>> cont_perc_;  // the area percentage of the contour in its layer
  std::vector<int> layer_cell_cnt_;  // total number of cells in each layer/level
  std::vector<std::vector<RetrievalKey>> layer_keys_;  // the key of each layer
  std::vector<std::vector<BCI>> layer_key_bcis_;  // NOTE: No validity check on bci. Check key before using corresponding bci!

  cv::Mat1f bev_;
//  std::vector<std::vector<V2F>> c_height_position_;  // downsampled but not discretized point xy position, another bev
//  std::map<int, V2F> pillar_pos2f_;  // downsampled but not discretized point xy position,
  std::vector<std::pair<int, Pixelf>> bev_pixfs_; // float row col height, and the hash generated from discrete row column.
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
//    pillar_pos2f_.clear();
//    bev_pixfs_.reserve(int(cfg_.n_col_ * cfg_.n_row_ * 0.05));
    cont_views_.resize(cfg_.lv_grads_.size());
    cont_perc_.resize(cfg_.lv_grads_.size());

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

    TicToc clk;
    int cell_count = 0;
    std::map<int, Pixelf> tmp_pillars;
    // Downsample before using?
    for (const auto &pt: ptr_gapc->points) {
      std::pair<int, int> rc = hashPointToImage<PointType>(pt);
      if (rc.first > 0) {
        float height = cfg_.lidar_height_ + pt.z;
        if (bev_(rc.first, rc.second) < height) {
          bev_(rc.first, rc.second) = height;
          V2F coor_f = pointToContRowCol(V2F(pt.x, pt.y));  // same coord as row and col
//          pillar_pos2f_[rc.first * cfg_.n_col_ + rc.second] = coor_f;
          tmp_pillars[rc.first * cfg_.n_col_ + rc.second] = Pixelf(coor_f.x(), coor_f.y(), height);

        }
        max_bin_val_ = max_bin_val_ < height ? height : max_bin_val_;
        min_bin_val_ = min_bin_val_ > height ? height : min_bin_val_;
      }
    }
    bev_pixfs_.clear();
    bev_pixfs_.insert(bev_pixfs_.begin(), tmp_pillars.begin(), tmp_pillars.end());
    std::cout << "Time makebev: " << clk.toc() << std::endl;

//    bev_pixfs_.shrink_to_fit();
//    std::sort(bev_pixfs_.begin(), bev_pixfs_.end());  // std::map is ordered by definition

    printf("Max/Min bin height: %f %f\n", max_bin_val_, min_bin_val_);
    if (!str_id.empty())
      str_id_ = std::move(str_id);
    else
      str_id_ = std::to_string(ptr_gapc->header.stamp);

//    printf("Continuous Pos size: %lu\n", pillar_pos2f_.size());
    printf("Continuous Pos size: %lu\n", bev_pixfs_.size());

//    size_t sizeInBytes = bev_.total() * bev_.elemSize();
//    std::cout << "bev size byte: " << sizeInBytes<< std::endl;

#if SAVE_MID_FILE
    cv::Mat mask, view;
    inRange(bev_, cv::Scalar::all(0), cv::Scalar::all(max_bin_val_), mask);
//    inRange(bev_, cv::Scalar::all(0), cv::Scalar::all(5.0), mask);  // NOTE: set to a fixed max height for uniformity
//    CHECK_GT(5.0, cfg_.lv_grads_.back());
    normalize(bev_, view, 0, 255, cv::NORM_MINMAX, -1, mask);
    std::string dir = std::string(PJSRCDIR) + "/results/bev_img/";
    cv::imwrite(dir + "cart_context-" + str_id_ + ".png", view);
#endif
  }

  void clearImage() {
    bev_.release();
  }

  void resumeImage() {
    bev_ = cv::Mat::ones(cfg_.n_row_, cfg_.n_col_, CV_32F) * (-VAL_ABS_INF_);
    for (const auto &pillar: bev_pixfs_) {
      int rr = pillar.first / cfg_.n_col_;
      int cc = pillar.first % cfg_.n_col_;
      DCHECK_LT(rr, cfg_.n_row_);
      DCHECK_LT(cc, cfg_.n_col_);
      bev_(rr, cc) = pillar.second.elev;
    }
  }

  cv::Mat1f getBevImage() const {
    if (bev_.empty()) {
      cv::Mat1f tmp = cv::Mat::ones(cfg_.n_row_, cfg_.n_col_, CV_32F) * (-VAL_ABS_INF_);
      for (const auto &pillar: bev_pixfs_) {
        int rr = pillar.first / cfg_.n_col_;
        int cc = pillar.first % cfg_.n_col_;
        DCHECK_LT(rr, cfg_.n_row_);
        DCHECK_LT(cc, cfg_.n_col_);
        tmp(rr, cc) = pillar.second.elev;
      }
      return tmp;
    } else
      return bev_.clone();
  }

  void makeContoursRecurs() {
    cv::Rect full_bev_roi(0, 0, bev_.cols, bev_.rows);

    TicToc clk;
    makeContourRecursiveHelper(full_bev_roi, cv::Mat1b(1, 1), 0, nullptr);
    std::cout << "Time makecontour: " << clk.toc() << std::endl;

    for (int ll = 0; ll < cont_views_.size(); ll++) {
      std::sort(cont_views_[ll].begin(), cont_views_[ll].end(),
                [&](const std::shared_ptr<ContourView> &p1, const std::shared_ptr<ContourView> &p2) -> bool {
                  return p1->cell_cnt_ > p2->cell_cnt_;
                });   // bigger contours first. Or heavier first?
      layer_cell_cnt_[ll] = 0;
      for (int j = 0; j < cont_views_[ll].size(); j++) {
        layer_cell_cnt_[ll] += cont_views_[ll][j]->cell_cnt_;
      }

      cont_perc_[ll].reserve(cont_views_[ll].size());
      for (int j = 0; j < cont_views_[ll].size(); j++) {
        cont_perc_[ll].push_back(cont_views_[ll][j]->cell_cnt_ * 1.0f / layer_cell_cnt_[ll]);
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
    const float roi_radius = 10.0f;
    const int roi_radius_padded = std::ceil(roi_radius + 1);
    for (int ll = 0; ll < cfg_.lv_grads_.size(); ll++) {
//      cv::Mat mask;
//      cv::threshold(bev_, mask, cfg_.lv_grads_[ll], 123,
//                    cv::THRESH_TOZERO); // mask is same type and dimension as bev_
      int accumulate_cell_cnt = 0;
      for (int seq = 0; seq < piv_firsts; seq++) {
        RetrievalKey key;
        key.setZero();

        BCI bci(seq, ll);

        if (cont_views_[ll].size() > seq)
          accumulate_cell_cnt += cont_views_[ll][seq]->cell_cnt_;

        if (cont_views_[ll].size() > seq && cont_views_[ll][seq]->cell_cnt_ >= cfg_.min_cont_key_cnt_) {

          V2F v_cen = cont_views_[ll][seq]->pos_mean_.cast<float>();
          int r_cen = int(v_cen.x()), c_cen = int(v_cen.y());
          int r_min = std::max(0, r_cen - roi_radius_padded),
              r_max = std::min(cfg_.n_row_ - 1, r_cen + roi_radius_padded);
          int c_min = std::max(0, c_cen - roi_radius_padded),
              c_max = std::min(cfg_.n_col_ - 1, c_cen + roi_radius_padded);

          int num_bins = 7;
          KeyFloatType bin_len = roi_radius / num_bins;
          std::vector<KeyFloatType> ring_bins(num_bins, 0);

          int div_per_bin = 5;
          std::vector<KeyFloatType> discrete_divs(div_per_bin * num_bins, 0);
          KeyFloatType div_len = roi_radius / (num_bins * div_per_bin);
          int cnt_point = 0;

          RunningStatRecorder rec_tmp; // for case 3

//          // Get data for visualization: (for data to draw plots in paper) paper data (1/3)
//          bool vis_data = (ll == 2 && seq == 0 && int_id_ == 1648);
//          if (vis_data) { printf("\n=== vis: \n"); }
//          std::vector<KeyFloatType> dense_divs(140, 0);
//          KeyFloatType ddiv_len = roi_radius / (140);


          for (int rr = r_min; rr <= r_max; rr++) {
            for (int cc = c_min; cc <= c_max; cc++) {
//              if (bev_(rr, cc) < cfg_.lv_grads_[ll])
              if (bev_(rr, cc) < cfg_.lv_grads_[DIST_BIN_LAYERS[0]])  // NOTE: new key
                continue;

              int q_hash = rr * cfg_.n_col_ + cc;
              std::pair<int, Pixelf> sear_res = search_vec<Pixelf>(bev_pixfs_, 0, bev_pixfs_.size() - 1,
                                                                   q_hash);
              DCHECK_EQ(sear_res.first, q_hash);
              KeyFloatType dist = (V2F(sear_res.second.row_f, sear_res.second.col_f) - v_cen).norm();

//              KeyFloatType dist = (pillar_pos2f_.at(rr * cfg_.n_col_ + cc) - v_cen).norm();

              // case 1: ring, height, 7
//              if (dist < roi_radius - 1e-2 && bev_(rr, cc) > cfg_.lv_grads_[0]) {  // add gaussian to bins
//                int bin_idx = int(dist / bin_len);
//                ring_bins[bin_idx] += bev_(rr, cc);    // no gaussian
//              }

              // case 2: gmm, normalized
              if (dist < roi_radius - 1e-2 && bev_(rr, cc) > cfg_.lv_grads_[DIST_BIN_LAYERS[0]]) { // imprv key variance
//                int higher_cnt = 1; // number of levels spanned by this pixel
//                for (int ele = ll + 1; ele < cfg_.lv_grads_.size(); ele++)
                int higher_cnt = 0;
                for (int ele = DIST_BIN_LAYERS[0]; ele < cfg_.lv_grads_.size(); ele++)
                  if (bev_(rr, cc) > cfg_.lv_grads_[ele])
                    higher_cnt++;

                cnt_point++;
                for (int div_idx = 0; div_idx < num_bins * div_per_bin; div_idx++)
                  discrete_divs[div_idx] +=
                      higher_cnt * gaussPDF<KeyFloatType>(div_idx * div_len + 0.5 * div_len, dist, 1.0);

//                if (vis_data) {  // paper data (2/3)
//                  printf("discrete: %f %d %f\n", dist, higher_cnt, bev_(rr, cc));
//                  for (int ddiv_idx = 0; ddiv_idx < dense_divs.size(); ddiv_idx++)
//                    dense_divs[ddiv_idx] +=
//                        higher_cnt * gaussPDF<KeyFloatType>(ddiv_idx * ddiv_len + 0.5 * ddiv_len, dist, 1.0);
//                }
              }

//              // case 3: using another ellipse
//              if (dist < roi_radius - 1e-2 && bev_(rr, cc) > cfg_.lv_grads_[ll]) {
//                auto pos2f = pillar_pos2f_.at(rr * cfg_.n_col_ + cc);
//                rec_tmp.runningStatsF(pos2f.x(), pos2f.y(), bev_(rr, cc));
//              }

            }
          }

//          if (vis_data) {  // paper data (3/3)
//            for (auto dense_div_dat: dense_divs)
//              printf("%f,", dense_div_dat);
//          }
//          if (vis_data) { printf("\nend vis ===\n"); }


          // case 2: gmm, normalized
          for (int b = 0; b < num_bins; b++) {
            for (int d = 0; d < div_per_bin; d++) {
              ring_bins[b] += discrete_divs[b * div_per_bin + d];
            }
            ring_bins[b] *= bin_len / std::sqrt(cnt_point);
//            ring_bins[b] *= bin_len;  // almost no performance degradation compared with the above one
          }

//          // case 3: using another ellipse
//          ContourView cv_tmp(ll, 0, 0, nullptr); // for case 3
//          cv_tmp.calcStatVals(rec_tmp);



          // TODO: make the key generation from one contour more distinctive
//          key(0) = std::sqrt(cont_views_[ll][seq]->eig_vals_(1));  // max eigen value
//          key(1) = std::sqrt(cont_views_[ll][seq]->eig_vals_(0));  // min eigen value
//          key(2) = (cont_views_[ll][seq]->pos_mean_ - cont_views_[ll][seq]->com_).norm();

          key(0) =
              std::sqrt(cont_views_[ll][seq]->eig_vals_(1) * cont_views_[ll][seq]->cell_cnt_);  // max eigen value * cnt
          key(1) =
              std::sqrt(cont_views_[ll][seq]->eig_vals_(0) * cont_views_[ll][seq]->cell_cnt_);  // min eigen value * cnt
//          key(2) = (cont_views_[ll][seq]->pos_mean_ - cont_views_[ll][seq]->com_).norm() *
//                   std::sqrt(cont_views_[ll][seq]->cell_cnt_);
//                   (cont_views_[ll][seq]->cell_cnt_);
          key(2) = std::sqrt(accumulate_cell_cnt);


          // case 1,2:
          DCHECK_EQ(num_bins + 3, RET_KEY_DIM);
          for (int nb = 0; nb < num_bins; nb++) {
//            key(3 + nb) = ring_bins[nb];
//            key(3 + nb) = ring_bins[nb] / (M_PI * (2 * nb + 1) * bin_len);  // density on the ring
            key(3 + nb) = ring_bins[nb];  // case 2.1: count on the ring
//            key(3 + nb) = ring_bins[nb] / (2 * nb + 1);  // case 2.2: kind of density on the ring
          }

//          // case 3:
//          key(3) = std::sqrt(cont_views_[ll][seq]->cell_cnt_);
//
//          key(4) = std::sqrt(cv_tmp.eig_vals_(1) * cv_tmp.cell_cnt_);
//          key(5) = std::sqrt(cv_tmp.eig_vals_(0) * cv_tmp.cell_cnt_);
//          key(6) = (cv_tmp.pos_mean_ - cv_tmp.com_).norm() * cv_tmp.cell_cnt_;
//          key(7) = std::sqrt(cv_tmp.cell_cnt_);
//
//          V2D cc_line = cont_views_[ll][seq]->pos_mean_ - cv_tmp.pos_mean_;
//          key(8) = cc_line.norm();
//          key(9) = std::sqrt(std::abs(cv_tmp.cell_cnt_ - cont_views_[ll][seq]->cell_cnt_));



          // hash dists and angles of the neighbours around the anchor/pivot to bit keys
          // hard coded
          for (int bl = 0; bl < NUM_BIN_KEY_LAYER; bl++) {
            int bit_offset = bl * BITS_PER_LAYER;
            for (int j = 0; j < std::min(dist_firsts, (int) cont_views_[DIST_BIN_LAYERS[bl]].size()); j++) {
              if (ll != DIST_BIN_LAYERS[bl] || j != seq) {
                V2F vec_cc =
                    cont_views_[DIST_BIN_LAYERS[bl]][j]->pos_mean_ - cont_views_[ll][seq]->pos_mean_;
                float tmp_dist = vec_cc.norm();

                if (tmp_dist > (BITS_PER_LAYER - 1) * 1.01 + 5.43 - 1e-3 // the last bit of layer sector is always 0
                    || tmp_dist <= 5.43)  // TODO: nonlinear mapping?
                  continue;

                float tmp_orie = std::atan2(vec_cc.y(), vec_cc.x());
                int dist_idx = std::min(std::floor((tmp_dist - 5.43) / 1.01), BITS_PER_LAYER - 1.0) + bit_offset;
                DCHECK_LT(dist_idx, BITS_PER_LAYER * NUM_BIN_KEY_LAYER);
                bci.dist_bin_.set(dist_idx, true);
//                bci.dist_bit_neighbors_[dist_idx].emplace_back(DIST_BIN_LAYERS[bl], j, tmp_dist, tmp_orie);
                bci.nei_pts_.emplace_back(DIST_BIN_LAYERS[bl], j, dist_idx, tmp_dist, tmp_orie);
              }
            }
          }

          if (!bci.nei_pts_.empty()) {
            std::sort(bci.nei_pts_.begin(), bci.nei_pts_.end(),
                      [&](const BCI::RelativePoint &p1, const BCI::RelativePoint &p2) {
                        return p1.bit_pos < p2.bit_pos;
                      });

            bci.nei_idx_segs_.emplace_back(0);
            for (int p1 = 0; p1 < bci.nei_pts_.size(); p1++) {
              if (bci.nei_pts_[bci.nei_idx_segs_.back()].bit_pos != bci.nei_pts_[p1].bit_pos)
                bci.nei_idx_segs_.emplace_back(p1);
            }
            bci.nei_idx_segs_.emplace_back(bci.nei_pts_.size());
            DCHECK_EQ(bci.nei_idx_segs_.size(), bci.dist_bin_.count() + 1);
          }

        }
//        if(key.sum()!=0)
        layer_key_bcis_[ll].emplace_back(bci);  // no validity check on bci. check key before use bci!
        layer_keys_[ll].emplace_back(key);  // even invalid keys are recorded.

//        printf("Key: l%d s%d: ", ll, seq);
//        for (float &ki: key.array)
//          printf("%8.4f ", ki);
//        printf("\n");
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

#if SAVE_MID_FILE
    // save statistics of this scan:
    std::string fpath = std::string(PJSRCDIR) + "/results/contours_orig-" + str_id_ + ".txt";
    saveContours(fpath, cont_views_);
#endif

    int cnt = 0;
    printf("Manager data sizes:\n");

    for (const auto &itms: cont_views_)
      for (const auto &itm: itms)
        cnt++;
    printf("cont_views_: %d\n", cnt);

    cnt = 0;
    for (const auto &itms: layer_keys_)
      for (const auto &itm: itms)
        cnt++;
    printf("layer_keys_: %d\n", cnt);

    cnt = 0;
    for (const auto &itms: layer_key_bcis_)
      for (const auto &itm: itms)
        cnt++;
    printf("layer_key_bcis_: %d\n", cnt);

    cnt = 0;
    for (const auto &itms: bev_pixfs_)
      cnt++;
    printf("bev_pixfs_: %d\n", cnt);


    // ablation study:
//    bev_.release();
//    bev_.release();
//    bev_pixfs_.clear();
//    cont_views_.clear();
//    layer_key_bcis_.clear();
//    layer_keys_.clear();



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
              ContourView::addContourRes(*new_cont_views[ll].back(), *cont_views_[ll][i], view_stat_cfg_)));
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
    std::vector<std::pair<int, float>> dists;
    for (int i = 0; i < std::min(top_n, (int) cont_views_[level].size()); i++)
      if (i != pivot)
        dists.emplace_back(i, (cont_views_[level][i]->pos_mean_ - cont_views_[level][pivot]->pos_mean_).norm());

    std::sort(dists.begin(), dists.end(), [&](const std::pair<int, float> &a, const std::pair<int, float> &b) {
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
    std::vector<std::pair<int, float>> bearings;
    bool first_set = false;
    V2F vec0(0, 0);
    for (int i = 0; i < std::min(top_n, (int) cont_views_[level].size()); i++) {
      if (i != pivot) {
        if (!first_set) {
          bearings.emplace_back(i, 0);
          first_set = true;
          vec0 = (cont_views_[level][i]->pos_mean_ - cont_views_[level][pivot]->pos_mean_).normalized();
        } else {
          V2F vec1 = (cont_views_[level][i]->pos_mean_ - cont_views_[level][pivot]->pos_mean_).normalized();
          float ang = std::atan2(vec0.x() * vec1.y() - vec0.y() * vec1.x(), vec0.dot(vec1));
          bearings.emplace_back(i, ang);
        }
      }
    }

    std::sort(bearings.begin(), bearings.end(), [&](const std::pair<int, float> &a, const std::pair<int, float> &b) {
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
    cv::threshold(getBevImage(), mask, cfg_.lv_grads_[level], 123,
                  cv::THRESH_TOZERO); // mask is same type and dimension as bev_

    cv::Mat normalized_layer, mask_u8;
    cv::normalize(mask, normalized_layer, 0, 255, cv::NORM_MINMAX, CV_8U);  // dtype=-1 (default): same type as input
    return normalized_layer;
  }

  // TODO: get retrieval key of a scan
  const std::vector<RetrievalKey> &getLevRetrievalKey(int level) const {
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

  // get contour
  inline const std::vector<std::shared_ptr<ContourView>> &getLevContours(int level) const {
    DCHECK_GE(level, 0);
    DCHECK_GT(cont_views_.size(), level);
    return cont_views_[level];
  }

  inline int getLevTotalPix(int level) const {
    DCHECK_GE(level, 0);
    DCHECK_GT(layer_cell_cnt_.size(), level);
    return layer_cell_cnt_[level];
  }

  // get bci
  const std::vector<BCI> &getLevBCI(int level) const {
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

  inline const ContourManagerConfig &getConfig() const {
    return cfg_;
  }

  inline const float &getAreaPerc(const int8_t &lev, const int8_t &seq) const {
    return cont_perc_[lev][seq];
  }

  // TODO: check if contours in two scans can be accepted as from the same heatmap, and return the transform
  // TODO: when retrieval key contains the combination, we should only look into that combination.
  // T_tgt = T_delta * T_src
  static std::pair<Eigen::Isometry2d, bool> calcScanCorresp(const ContourManager &src, const ContourManager &tgt);

  // T_tgt = T_delta * T_src
  /// Check the similarity of each pair of stars(contours) in the constellation.
  /// Note: check the size lb outside or check the size of `area_perc` to ensure the prediction is Positive
  /// \param src
  /// \param tgt
  /// \param cstl_in
  /// \param lb
  /// \param cont_sim
  /// \param cstl_out The filtered constellation
  /// \param area_perc The accompanying area percentage of each pair (at the level specified by the constell pair)
  /// \return
  static ScorePairwiseSim checkConstellCorrespSim(const ContourManager &src, const ContourManager &tgt,
                                                  const std::vector<ConstellationPair> &cstl_in,
                                                  const ScorePairwiseSim &lb,
                                                  const ContourSimThresConfig &cont_sim,
                                                  std::vector<ConstellationPair> &cstl_out,
                                                  std::vector<float> &area_perc) {
    // cross level consensus (CLC)
    // The rough constellation should have been established.
    DCHECK_EQ(src.cont_views_.size(), tgt.cont_views_.size());

//    int matched_cnt{};
    ScorePairwiseSim ret;

    std::map<int, std::pair<float, float>> lev_frac;  // {lev:[src, tgt], }

    cstl_out.clear();
    area_perc.clear();
    // 1. check individual sim
#if HUMAN_READABLE
    printf("check individual sim of the constellation:\n");
#endif
    for (auto pr: cstl_in) {
//      if (ContourView::checkSim(*src.cont_views_[cstl_in[pr].level][cstl_in[pr].seq_src],
//                                *tgt.cont_views_[cstl_in[pr].level][cstl_in[pr].seq_tgt]))
      bool curr_success = false;
      if (checkContPairSim(src, tgt, pr, cont_sim)) {
        cstl_out.push_back(pr);
        auto &it = lev_frac[pr.level];
        it.first += src.cont_perc_[pr.level][pr.seq_src];
        it.second += tgt.cont_perc_[pr.level][pr.seq_tgt];
        curr_success = true;
      }
#if HUMAN_READABLE
      printf("%d@lev %d, %d-%d\n", int(curr_success), pr.level, pr.seq_src, pr.seq_tgt);
#endif
    }

#if HUMAN_READABLE
    for (const auto &rec: lev_frac) {
      printf("matched percent lev: %d, %.3f/%.3f\n", rec.first, rec.second.first, rec.second.second);
    }
#endif

    ret.i_indiv_sim = cstl_out.size();
    if (ret.i_indiv_sim < lb.i_indiv_sim)
      return ret;  // TODO: use cross level consensus to find more possible matched pairs

    // 2. check orientation
    // 2.1 get major axis direction
    V2F shaft_src(0, 0), shaft_tgt(0, 0);
    for (int i = 1; i < std::min((int) cstl_out.size(), 10); i++) {
      for (int j = 0; j < i; j++) {
        V2F curr_shaft = src.cont_views_[cstl_out[i].level][cstl_out[i].seq_src]->pos_mean_ -
                         src.cont_views_[cstl_out[j].level][cstl_out[j].seq_src]->pos_mean_;
        if (curr_shaft.norm() > shaft_src.norm()) {
          shaft_src = curr_shaft.normalized();
          shaft_tgt = (tgt.cont_views_[cstl_out[i].level][cstl_out[i].seq_tgt]->pos_mean_ -
                       tgt.cont_views_[cstl_out[j].level][cstl_out[j].seq_tgt]->pos_mean_).normalized();
        }
      }
    }
    // 2.2 if both src and tgt contour are orientationally salient but the orientations largely differ, remove the pair
    int num_sim = cstl_out.size();
    for (int i = 0; i < num_sim;) {
      const auto &sc1 = src.cont_views_[cstl_out[i].level][cstl_out[i].seq_src],
          &tc1 = tgt.cont_views_[cstl_out[i].level][cstl_out[i].seq_tgt];
      if (sc1->ecc_feat_ && tc1->ecc_feat_) {
        float theta_s = std::acos(shaft_src.transpose() * sc1->eig_vecs_.col(1));   // acos: [0,pi)
        float theta_t = std::acos(shaft_tgt.transpose() * tc1->eig_vecs_.col(1));
        if (diff_delt<float>(theta_s, theta_t, M_PI / 6) && diff_delt<float>(M_PI - theta_s, theta_t, M_PI / 6)) {
          std::swap(cstl_out[i], cstl_out[num_sim - 1]);
          num_sim--;
          continue;
        }
      }
      i++;
    }
    cstl_out.erase(cstl_out.begin() + num_sim, cstl_out.end()); // sure to reduce size, without default constructor
    ret.i_orie_sim = cstl_out.size();
    if (ret.i_orie_sim < lb.i_orie_sim)
      return ret;  // TODO: use cross level consensus to find more possible matched pairs

#if HUMAN_READABLE
    std::sort(cstl_out.begin(), cstl_out.end());  // human readability
    printf("Found matched pairs:\n");
    for (const auto &i: cstl_out) {
      printf("\tlev %d, src:tgt  %d: %d\n", i.level, i.seq_src, i.seq_tgt);
    }
#endif

    // get the percentage of each of the nodes in the constellation
    area_perc.reserve(cstl_out.size());
    for (const auto &i: cstl_out) {
      area_perc.push_back(0.5f * (src.cont_perc_[i.level][i.seq_src] + tgt.cont_perc_[i.level][i.seq_tgt]));
    }


    // get the percentage of points used in match v. total area of each level.
//    std::vector<float> level_perc_used_src(src.cfg_.lv_grads_.size(), 0);
//    std::vector<float> level_perc_used_tgt(tgt.cfg_.lv_grads_.size(), 0);
//    for (const auto &i: cstl_out) {
//      level_perc_used_src[i.level] += src.cont_perc_[i.level][i.seq_src];
//      level_perc_used_tgt[i.level] += tgt.cont_perc_[i.level][i.seq_tgt];
//    }
//    printf("Percentage used in the proposed constellation:\n");
//    for (int i = 0; i < src.cfg_.lv_grads_.size(); i++) {
//      printf("lev: %d, src: %4.2f, tgt: %4.2f\n", i, level_perc_used_src[i], level_perc_used_tgt[i]);
//    }

    // the percentage score in s a single float (deprecated)
//    float perc = 0;
//    for (int i = 0; i < NUM_BIN_KEY_LAYER; i++) {
//      perc += LAYER_AREA_WEIGHTS[i] *
//              (level_perc_used_src[DIST_BIN_LAYERS[i]] + level_perc_used_tgt[DIST_BIN_LAYERS[i]]) / 2;
//    }
//    ret.f_area_perc = int(perc * 100);

    return ret;
  }

  /// Calculate a transform from a list of manually/externally matched contour indices (constellation)
  /// \tparam Iter The iterator of the container class (vector, set, etc.) that holds constellation pairs
  /// \param src
  /// \param tgt
  /// \param cstl_beg
  /// \param cstl_end
  /// \return
  template<typename Iter>
  static Eigen::Isometry2d getTFFromConstell(const ContourManager &src, const ContourManager &tgt,
                                             Iter cstl_beg, Iter cstl_end) {  // no const, just don't modify in the code
    int num_elem = cstl_end - cstl_beg;
    CHECK_GT(num_elem, 2);
    Eigen::Matrix<double, 2, Eigen::Dynamic> pointset1; // src
    Eigen::Matrix<double, 2, Eigen::Dynamic> pointset2; // tgt
    pointset1.resize(2, num_elem);
    pointset2.resize(2, num_elem);
    for (int i = 0; i < num_elem; i++) {
      pointset1.col(i) = src.cont_views_[(cstl_beg + i)->level][(cstl_beg +
                                                                 i)->seq_src]->pos_mean_.template cast<double>();
      pointset2.col(i) = tgt.cont_views_[(cstl_beg + i)->level][(cstl_beg +
                                                                 i)->seq_tgt]->pos_mean_.template cast<double>();
    }

    Eigen::Matrix3d T_delta = Eigen::umeyama(pointset1, pointset2, false);
#if HUMAN_READABLE
    std::cout << "Transform matrix:\n" << T_delta << std::endl;
#endif
    Eigen::Isometry2d ret;
    ret.setIdentity();
    ret.rotate(std::atan2(T_delta(1, 0), T_delta(0, 0)));
    ret.pretranslate(T_delta.block<2, 1>(0, 2));

    return ret;
  }

  inline static bool
  checkContPairSim(const ContourManager &src, const ContourManager &tgt, const ConstellationPair &cstl,
                   const ContourSimThresConfig &cont_sim) {
    return ContourView::checkSim(*src.cont_views_[cstl.level][cstl.seq_src],
                                 *tgt.cont_views_[cstl.level][cstl.seq_tgt], cont_sim);
  }

  static void
  saveMatchedPairImg(const std::string &fpath, const ContourManager &cm1,
                     const ContourManager &cm2) {
    ContourManagerConfig config = cm2.getConfig();

    DCHECK_EQ(config.n_row_, cm1.getConfig().n_row_);
    DCHECK_EQ(config.n_col_, cm1.getConfig().n_col_);
    DCHECK_EQ(cm2.getConfig().n_row_, cm1.getConfig().n_row_);
    DCHECK_EQ(cm2.getConfig().n_col_, cm1.getConfig().n_col_);

//    cv::Mat output((config.n_row_ + 1) * config.lv_grads_.size(), config.n_col_ * 2, CV_8U);
    cv::Mat output(config.n_row_ * 2 + 1, (config.n_col_ + 1) * config.lv_grads_.size(), CV_8U);

    output.setTo(255);

    for (int i = 0; i < config.lv_grads_.size(); i++) {
//      cm1.getContourImage(i).copyTo(output(cv::Rect(0, i * config.n_row_ + i, config.n_col_, config.n_row_)));
//      cm2.getContourImage(i).copyTo(
//          output(cv::Rect(config.n_col_, i * config.n_row_ + i, config.n_col_, config.n_row_)));

      cm1.getContourImage(i).copyTo(output(cv::Rect(i * config.n_col_ + i, 0, config.n_col_, config.n_row_)));
      cm2.getContourImage(i).copyTo(
          output(cv::Rect(i * config.n_col_ + i, config.n_row_ + 1, config.n_col_, config.n_row_)));
    }
    cv::imwrite(fpath, output);
  }


};


#endif //CONT2_CONTOUR_MNG_H
