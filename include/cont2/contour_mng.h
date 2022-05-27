//
// Created by lewis on 5/5/22.
//

#ifndef CONT2_CONTOUR_MNG_H
#define CONT2_CONTOUR_MNG_H

#include <iostream>
#include <fstream>
#include <cstdlib>
#include "cont2/contour.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <opencv2/core/mat.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

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

  ArrayAsKey<sz> operator-(ArrayAsKey<sz> const &obj) {
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

using RetrievalKey = ArrayAsKey<6>;

struct ContourManagerConfig {
  std::vector<float> lv_grads_;  // n marks, n+1 levels
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
  std::string str_id_;
  int int_id_;

  // data
  std::vector<std::vector<std::shared_ptr<ContourView>>> cont_views_;  // TODO: use a parallel vec of vec for points?
  std::vector<int> layer_cell_cnt_;  // total number of cells in each layer/level
  std::vector<RetrievalKey> layer_keys_;  // the key of each layer

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
                                  const std::shared_ptr<ContourView> &parent) {
    DCHECK(bool(level)==bool(parent));
    if (level >= cfg_.lv_grads_.size())
      return;

    float h_min = cfg_.lv_grads_[level], h_max = VAL_ABS_INF_;

    cv::Mat1f bev_roi = bev_(cc_roi), thres_roi;
    cv::threshold(bev_roi, thres_roi, h_min, 255, cv::THRESH_BINARY);

    cv::Mat1b bin_bev_roi;
    thres_roi.convertTo(bin_bev_roi, CV_8U);

    if (level)
//      cv::bitwise_and(bin_bev_roi, bin_bev_roi, bin_bev_roi, cc_mask);  // Wrong method: some pixels may be unaltered since neglected by mask
      cv::bitwise_and(bin_bev_roi, cc_mask, bin_bev_roi);

    if (level < cfg_.lv_grads_.size() - 1)
      h_max = cfg_.lv_grads_[level - 1];

    // 2. calculate connected blobs
    cv::Mat1i labels, stats;  // int (CV_32S)
    cv::Mat centroids;  // not in use
    cv::connectedComponentsWithStats(bin_bev_roi, labels, stats, centroids, 8, CV_32S);   // on local patch

    // 3. create contours for each connected component
    // https://stackoverflow.com/questions/37745274/opencv-find-perimeter-of-a-connected-component/48618464#48618464
    for (int n = 1; n < stats.rows; n++) {  // n=0: background
//      printf("Area: %d\n", stats.at<int>(n, cv::CC_STAT_AREA));

      //Rectangle around the connected component
      // Rect: col0, row0, n_col, n_row
      cv::Rect rect_g(stats(n, 0) + cc_roi.x, stats(n, 1) + cc_roi.y, stats(n, 2), stats(n, 3)); // global: on bev
      cv::Rect rect_l(stats(n, 0), stats(n, 1), stats(n, 2), stats(n, 3)); // local: relative to bev_roi

      cv::Mat1b mask_n = labels(rect_l) == n;

      std::shared_ptr<ContourView> ptr_tmp_cv(new ContourView(level, h_min, h_max, rect_g, parent));

      for (int i = 0; i < rect_l.height; i++)
        for (int j = 0; j < rect_l.width; j++)
          if (mask_n(i, j)) {
//            ptr_tmp_cv->runningStats(i + rect_g.y, j + rect_g.x, bev_(i + rect_g.y, j + rect_g.x)); // discrete
            V2F c_point = c_height_position_[i + rect_g.y][j + rect_g.x];
            ptr_tmp_cv->runningStatsF(c_point.x(), c_point.y(), bev_(i + rect_g.y, j + rect_g.x)); // continuous
          }
      ptr_tmp_cv->calcStatVals();
      DCHECK(ptr_tmp_cv->cell_cnt_ == stats(n, 4));
      cont_views_[level].emplace_back(ptr_tmp_cv);    // add to the manager's matrix
      if (parent)
        parent->children_.emplace_back(ptr_tmp_cv);

      // recurse
      // Get the mask for the contour

//      printf("contour ROI: %d, %d, level: %d\n", mask_n.rows, mask_n.cols, level);
      makeContourRecursiveHelper(rect_g, mask_n, level + 1, ptr_tmp_cv);

//      if (level == 2) {
//        cv::bitwise_or(mask_n, visualization(rect_g), visualization(rect_g));
//      }

    }

  }

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
          c_height_position_[rc.first][rc.second] = pointToContRowCol(V2F(pt.x, pt.y));  // same coord as row and col
        }
        max_bin_val_ = max_bin_val_ < height ? height : max_bin_val_;
        min_bin_val_ = min_bin_val_ > height ? height : min_bin_val_;
      }
    }
    printf("Max/Min bin height: %f %f\n", max_bin_val_, min_bin_val_);
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

    // make retrieval keys
    for (int ll = 0; ll < cfg_.lv_grads_.size(); ll++) {
      RetrievalKey key;
      key.setZero();
      if (cont_views_[ll].size() > 2 && cont_views_[ll][0]->cell_cnt_ > cfg_.cont_cnt_thres_ &&
          cont_views_[ll][1]->cell_cnt_ > cfg_.cont_cnt_thres_) { // TODO: make multiple keys for each level

        key(0) = std::sqrt(cont_views_[ll][0]->cell_cnt_);
        key(1) = std::sqrt(cont_views_[ll][1]->cell_cnt_);
        V2D cc_line = cont_views_[ll][0]->pos_mean_ - cont_views_[ll][1]->pos_mean_;
        key(2) = cc_line.norm();

        // distribution of projection perp to cc line
        cc_line.normalize();
        V2D cc_perp(-cc_line.y(), cc_line.x());

//        // case1: use cocentic distribution
//        M2D new_cov = (cont_views_[ll][0]->getManualCov() * (cont_views_[ll][0]->cell_cnt_ - 1) +
//                       cont_views_[ll][1]->getManualCov() * (cont_views_[ll][1]->cell_cnt_ - 1)) /
//                      (cont_views_[ll][0]->cell_cnt_ + cont_views_[ll][1]->cell_cnt_ - 1);
        // case2: use relative translation preserving distribution
        M2D new_cov = ContourView::addContourStat(*cont_views_[ll][0], *cont_views_[ll][1]).getManualCov();

        key(3) = std::sqrt(cc_perp.transpose() * new_cov * cc_perp);

        // distribution of projection to cc line
        key(4) = std::sqrt(cc_line.transpose() * new_cov * cc_line);

        // the max eigen value of the first ellipse
        key(5) = std::sqrt(cont_views_[ll][0]->eig_vals_(1));

      }


      layer_keys_[ll] = key;
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
    std::string fpath = std::string(PJSRCDIR) + "/results/contours_orig_" + str_id_ + ".txt";
    saveContours(fpath);

//    cv::imwrite("cart_context-mask-" + std::to_string(3) + "-" + str_id_ + "rec.png", visualization);
//    for (const auto &x: cont_views_) {
//      printf("level size: %lu\n", x.size());
//    }
  }

  void makeContours();

  // util functions
  // 1. save all contours' statistical data into a text file
  void saveContours(const std::string &fpath) const;

  // 2. save a layer of contours to image
  void saveContourImage(const std::string &fpath, int level) const;

  cv::Mat getContourImage(int level) const {
    cv::Mat mask;
    cv::threshold(bev_, mask, cfg_.lv_grads_[level], 123, cv::THRESH_TOZERO); // mask is same type and dimension as bev_
    cv::Mat normalized_layer, mask_u8;
    cv::normalize(mask, normalized_layer, 0, 255, cv::NORM_MINMAX, CV_8U);  // dtype=-1 (default): same type as input
    return normalized_layer;
  }

  inline ContourManagerConfig getConfig() const {
    return cfg_;
  }

  // TODO: get retrieval key of a scan
  RetrievalKey getRetrievalKey(int level) const {
    DCHECK_GE(level, 0);
    DCHECK_GT(cont_views_.size(), level);
    return layer_keys_[level];
  }

  inline std::string getStrID() const {
    return str_id_;
  }

  inline int getIntID() const {
    return int_id_;
  }

  // TODO: check if contours in two scans can be accepted as from the same heatmap, and return the transform
  // T_tgt = T_delta * T_src
  static std::pair<Eigen::Isometry2d, bool> calcScanCorresp(const ContourManager &src, const ContourManager &tgt) {
    DCHECK_EQ(src.cont_views_.size(), tgt.cont_views_.size());
    printf("calcScanCorresp(): \n");

    // configs
    int num_tgt_top = 5;
    int num_src_top = 4;

    int num_tgt_ser = 10; // when initial result is positive, we progressively search more correspondence pairs
    int num_src_ser = 10;
    std::vector<std::pair<int, int>> src_q_comb = {{0, 1},
                                                   {0, 2},
                                                   {0, 3},
                                                   {1, 2},
                                                   {1, 3},
                                                   {2, 3}};  // in accordance with num_src_top

    std::pair<Eigen::Isometry2d, bool> ret{};
    int num_levels = (int) src.cont_views_.size() - 2;

    // TODO: check FP rate for retrieval tasks

    for (int l = 0; l < num_levels; l++) {
      if (src.cont_views_[l].size() < num_src_top || tgt.cont_views_[l].size() < 3)
        continue;
      if (ret.second)
        break;
      printf("Matching level: %d\n", l);
      for (const auto &comb: src_q_comb) {
        for (int i = 0; i < std::min((int) tgt.cont_views_[l].size(), num_tgt_top); i++)
          for (int j = 0; j < std::min((int) tgt.cont_views_[l].size(), num_tgt_top); j++) {
            if (j == i)
              continue;
            // Contour Correspondence Proposal: comb.first=i, comb.second=j
            const auto sc1 = src.cont_views_[l][comb.first], sc2 = src.cont_views_[l][comb.second],
                tc1 = tgt.cont_views_[l][i], tc2 = tgt.cont_views_[l][j];

//            printf("-- Check src: %d, %d, tgt: %d, %d\n", comb.first, comb.second, i, j);

            // 1. test if the proposal fits in terms of individual contours
            bool is_pairs_sim = ContourView::checkSim(*sc1, *tc1) && ContourView::checkSim(*sc2, *tc2);
            if (!is_pairs_sim) {
              continue;
            }

            // 2. check geometry center distance
            double dist_src = (sc1->pos_mean_ - sc2->pos_mean_).norm();
            double dist_tgt = (tc1->pos_mean_ - tc2->pos_mean_).norm();
            if (std::max(dist_tgt, dist_src) > 5.0 && diff_delt(dist_src, dist_tgt, 5.0))
              continue;

            // 3. check contour orientation
            Eigen::Vector2d cent_s = (sc1->pos_mean_ - sc2->pos_mean_).normalized();
            Eigen::Vector2d cent_t = (tc1->pos_mean_ - tc2->pos_mean_).normalized();
            if (sc1->ecc_feat_ && tc1->ecc_feat_) {
              double theta_s = std::acos(cent_s.transpose() * sc1->eig_vecs_.col(1));   // acos: [0,pi)
              double theta_t = std::acos(cent_t.transpose() * tc1->eig_vecs_.col(1));
              if (diff_delt(theta_s, theta_t, M_PI / 12) && diff_delt(M_PI - theta_s, theta_t, M_PI / 12))
                continue;
            }
            if (sc2->ecc_feat_ && tc2->ecc_feat_) {
              double theta_s = std::acos(cent_s.transpose() * sc2->eig_vecs_.col(1));   // acos: [0,pi)
              double theta_t = std::acos(cent_t.transpose() * tc2->eig_vecs_.col(1));
              if (diff_delt(theta_s, theta_t, M_PI / 6) && diff_delt(M_PI - theta_s, theta_t, M_PI / 6))
                continue;
            }

            // 4. PROSAC
            // 4.1 get the rough transform to facilitate the similarity check (relatively large acceptance range)
            // can come from a naive 2 point transform estimation or a gmm2gmm
            Eigen::Matrix3d T_delta = estimateTF(sc1->pos_mean_, sc2->pos_mean_, tc1->pos_mean_,
                                                 tc2->pos_mean_).matrix(); // naive 2 point estimation

            // for pointset transform estimation
            Eigen::Matrix<double, 2, Eigen::Dynamic> pointset1; // src
            Eigen::Matrix<double, 2, Eigen::Dynamic> pointset2; // tgt
            pointset1.resize(2, 2);
            pointset2.resize(2, 2);
            pointset1.col(0) = sc1->pos_mean_;
            pointset1.col(1) = sc2->pos_mean_;
            pointset2.col(0) = tc1->pos_mean_;
            pointset2.col(1) = tc2->pos_mean_;

            // 4.2 create adjacency matrix (binary weight bipartite graph) or calculate on the go?
            std::vector<std::pair<int, int>> match_list = {{comb.first,  i},
                                                           {comb.second, j}};
            std::set<int> used_src{comb.first, comb.second}, used_tgt{i, j};
            // 4.3 check if new pairs exit
            double tf_dist_max = 5.0;
            for (int ii = 0; ii < std::min((int) src.cont_views_[l].size(), num_src_ser); ii++) {
              if (used_src.find(ii) != used_src.end())
                continue;
              for (int jj = 0; jj < std::min((int) tgt.cont_views_[l].size(), num_tgt_ser); jj++) {
                if (used_tgt.find(jj) != used_tgt.end())
                  continue;
                V2D pos_mean_src_tf = T_delta.block<2, 2>(0, 0) * src.cont_views_[l][ii]->pos_mean_
                                      + T_delta.block<2, 1>(0, 2);
                if ((pos_mean_src_tf - tgt.cont_views_[l][jj]->pos_mean_).norm() > tf_dist_max ||
                    !ContourView::checkSim(*src.cont_views_[l][ii], *tgt.cont_views_[l][jj])
                    )
                  continue;
                // handle candidate pairs
                // TODO: check consensus before adding:
                match_list.emplace_back(ii, jj);
                used_src.insert(ii);  // greedy
                used_tgt.insert(jj);

                // TODO: update transform
                // pure point method: umeyama

                pointset1.conservativeResize(Eigen::NoChange_t(), match_list.size());
                pointset2.conservativeResize(Eigen::NoChange_t(), match_list.size());
                pointset1.rightCols(1) = src.cont_views_[l][ii]->pos_mean_;
                pointset2.rightCols(1) = tgt.cont_views_[l][jj]->pos_mean_;
                T_delta = Eigen::umeyama(pointset1, pointset2, false);  // also need to check consensus

              }
              // TODO: termination criteria
            }


            // TODO: metric results, filter out some outlier
            if (match_list.size() > 4) {
              printf("Found matched pairs in level %d:\n", l);
              for (const auto &pr: match_list) {
                printf("\tsrc:tgt  %d: %d\n", pr.first, pr.second);
              }
              std::cout << "Transform matrix:\n" << T_delta << std::endl;
              // TODO: move ret to later
              ret.second = true;
              ret.first.setIdentity();
              ret.first.rotate(std::atan2(T_delta(1, 0), T_delta(0, 0)));
              ret.first.pretranslate(T_delta.block<2, 1>(0, 2));
            }
          }
      }

      // TODO: cross level consensus

      // TODO: full metric estimation

    }

    return ret;
  }


};


#endif //CONT2_CONTOUR_MNG_H
