//
// Created by lewis on 6/21/22.
//

#ifndef CONT2_CORRELATION_H
#define CONT2_CORRELATION_H

#include "cont2/contour_mng.h"
#include <ceres/ceres.h>

#include <memory>
#include <utility>


struct GMMOptConfig {
  double max_corr_dist_ = 10.0; // in bev pixels
  double min_area_perc_ = 0.95; // minimal percentage of area involved for each layer
  std::vector<int> levels_ = {1, 2, 3, 4}; // the layers to be considered in ellipse gmm.
  double cov_dilate_scale_ = 2.0;
};


struct GMMPair {
  struct GMMEllipse {
    Eigen::Matrix<double, 2, 2> cov_;
    Eigen::Matrix<double, 2, 1> mu_;
    double w_;

    GMMEllipse(Eigen::Matrix<double, 2, 2> cov, Eigen::Matrix<double, 2, 1> mu, double w) : cov_(std::move(cov)),
                                                                                            mu_(std::move(mu)),
                                                                                            w_(w) {}
  };

  // layers of useful data set at the time of init
  std::vector<std::vector<GMMEllipse>> ellipses_src, ellipses_tgt;
  std::vector<std::vector<std::pair<int, int>>> selected_pair_idx_;  // selected {src: tgt} pairs for f.g L2 distance calculation
  std::vector<int> src_cell_cnts_, tgt_cell_cnts_;
  int total_cells_src_ = 0, total_cells_tgt_ = 0;
  double auto_corr_src_{}, auto_corr_tgt_{};  // without normalization by cell count
  const double scale_;

  GMMPair(const ContourManager &cm_src, const ContourManager &cm_tgt, const GMMOptConfig &config,
          const Eigen::Isometry2d &T_init) : scale_(config.cov_dilate_scale_) {
    DCHECK_LE(config.levels_.size(), cm_src.getConfig().lv_grads_.size());

    // collect eigen values to isolate insignificant correlations
    std::vector<std::vector<float>> max_majax_src, max_majax_tgt;

    for (const auto lev: config.levels_) {
      int cnt_src_run = 0, cnt_src_full = cm_src.getLevTotalPix(lev);
      int cnt_tgt_run = 0, cnt_tgt_full = cm_tgt.getLevTotalPix(lev);

      ellipses_src.emplace_back();
      ellipses_tgt.emplace_back();
      max_majax_src.emplace_back();
      max_majax_tgt.emplace_back();
      selected_pair_idx_.emplace_back();

      const auto &src_layer = cm_src.getLevContours(lev);
      const auto &tgt_layer = cm_tgt.getLevContours(lev);

      for (const auto &view_ptr: src_layer) {
        if (cnt_src_run * 1.0 / cnt_src_full >= config.min_area_perc_)
          break;
        ellipses_src.back().emplace_back(view_ptr->getManualCov().cast<double>(), view_ptr->pos_mean_.cast<double>(),
                                         double(view_ptr->cell_cnt_));
        max_majax_src.back().emplace_back(std::sqrt(view_ptr->eig_vals_.y()));
        cnt_src_run += view_ptr->cell_cnt_;
      }
      for (const auto &view_ptr: tgt_layer) {
        if (cnt_tgt_run * 1.0 / cnt_tgt_full >= config.min_area_perc_)
          break;
        ellipses_tgt.back().emplace_back(view_ptr->getManualCov().cast<double>(), view_ptr->pos_mean_.cast<double>(),
                                         double(view_ptr->cell_cnt_));
        max_majax_tgt.back().emplace_back(std::sqrt(view_ptr->eig_vals_.y()));
        cnt_tgt_run += view_ptr->cell_cnt_;
      }
      src_cell_cnts_.emplace_back(cnt_src_full);
      tgt_cell_cnts_.emplace_back(cnt_tgt_full);
      total_cells_src_ += src_cell_cnts_.back();
      total_cells_tgt_ += tgt_cell_cnts_.back();
    }

    // pre-select (need initial guess)
    int total_pairs = 0;
    for (int li = 0; li < ellipses_src.size(); li++) {
      for (int si = 0; si < ellipses_src[li].size(); si++) {
        for (int ti = 0; ti < ellipses_tgt[li].size(); ti++) {
          Eigen::Matrix<double, 2, 1> delta_mu = T_init * ellipses_src[li][si].mu_ - ellipses_tgt[li][ti].mu_;
          if (delta_mu.norm() < 3.0 * (max_majax_src[li][si] + max_majax_tgt[li][ti])) {  // close enough to correlate
            selected_pair_idx_[li].emplace_back(si, ti);
            total_pairs++;
          }
        }
      }
    }
#if HUMAN_READABLE
    printf("Total pairs of gmm ellipses: %d\n", total_pairs);
#endif

    // calc auto-correlation
    for (int li = 0; li < config.levels_.size(); li++) {
      for (int i = 0; i < ellipses_src[li].size(); i++) {
        for (int j = 0; j < ellipses_src[li].size(); j++) {
          Eigen::Matrix2d new_cov = scale_ * (ellipses_src[li][i].cov_ + ellipses_src[li][j].cov_);
          Eigen::Vector2d new_mu = ellipses_src[li][i].mu_ - ellipses_src[li][j].mu_;
          auto_corr_src_ += ellipses_src[li][i].w_ * ellipses_src[li][j].w_ / std::sqrt(new_cov.determinant()) *
                            std::exp(-0.5 * new_mu.transpose() * new_cov.inverse() * new_mu);
        }
      }
      for (int i = 0; i < ellipses_tgt[li].size(); i++) {
        for (int j = 0; j < ellipses_tgt[li].size(); j++) {
          Eigen::Matrix2d new_cov = scale_ * (ellipses_tgt[li][i].cov_ + ellipses_tgt[li][j].cov_);
          Eigen::Vector2d new_mu = ellipses_tgt[li][i].mu_ - ellipses_tgt[li][j].mu_;
          auto_corr_tgt_ += ellipses_tgt[li][i].w_ * ellipses_tgt[li][j].w_ / std::sqrt(new_cov.determinant()) *
                            std::exp(-0.5 * new_mu.transpose() * new_cov.inverse() * new_mu);
        }
      }
    }
//    printf("Auto corr: src: %f, tgt: %f\n", auto_corr_src_, auto_corr_tgt_);
//    printf("Tot cells: src: %d, tgt: %d\n", total_cells_src_, total_cells_tgt_);
  }

  // evaluate
  template<typename T>
  bool operator()(const T *parameters, T *cost) const {

    const T x = parameters[0];
    const T y = parameters[1];
    const T theta = parameters[2];

    Eigen::Matrix<T, 2, 2> R;
    R << cos(theta), -sin(theta), sin(theta), cos(theta);
    Eigen::Matrix<T, 2, 1> t(x, y);

    cost[0] = T(0);

    for (int li = 0; li < selected_pair_idx_.size(); li++) {
      for (const auto &pr: selected_pair_idx_[li]) {
        // TODO: fine tuning: different weights for different levels
        Eigen::Matrix<T, 2, 2> new_cov =
            scale_ * (R * ellipses_src[li][pr.first].cov_ * R.transpose() + ellipses_tgt[li][pr.second].cov_);
        Eigen::Matrix<T, 2, 1> new_mu = R * ellipses_src[li][pr.first].mu_ + t - ellipses_tgt[li][pr.second].mu_;

        T qua = -0.5 * new_mu.transpose() * new_cov.inverse() * new_mu;
        cost[0] += -ellipses_tgt[li][pr.second].w_ * ellipses_src[li][pr.first].w_ * 1.0 / sqrt(new_cov.determinant()) *
                   exp(qua);
      }
    }

    return true;
  }

};

//! Constellation correlation
class ConstellCorrelation {
  GMMOptConfig cfg_;
  std::unique_ptr<ceres::GradientProblem> problem_ptr = nullptr;
  double auto_corr_src{}, auto_corr_tgt{};
  Eigen::Isometry2d T_best_;

public:
  ConstellCorrelation() = default;

  explicit ConstellCorrelation(GMMOptConfig cfg) : cfg_(std::move(cfg)) {
    T_best_.setIdentity();
  };

  /// Split init cost calc from full optimization, in case of too many candidates
  /// \param cm_src
  /// \param cm_tgt
  /// \param T_init should be the best possible, because we use it to simplify weak correlation pairs.
  /// \return
  double initProblem(const ContourManager &cm_src, const ContourManager &cm_tgt, const Eigen::Isometry2d &T_init) {
//    printf("Param before opt:\n");
//    for (auto dat: parameters) {
//      std::cout << dat << std::endl;
//    }
//    std::cout << T_delta.matrix() << std::endl;

    T_best_ = T_init;
    std::unique_ptr<GMMPair> ptr_gmm_pair(new GMMPair(cm_src, cm_tgt, cfg_, T_init));

    auto_corr_src = ptr_gmm_pair->auto_corr_src_;
    auto_corr_tgt = ptr_gmm_pair->auto_corr_tgt_;
    problem_ptr = std::make_unique<ceres::GradientProblem>(
        new ceres::AutoDiffFirstOrderFunction<GMMPair, 3>(ptr_gmm_pair.release()));

    return tryProblem(T_init);
  }

  /// Get the correlation under a certain transform
  /// \param T_try The TF under which to get the correlation. Big diff between T_init and T_try will cause problems.
  /// \return
  double tryProblem(const Eigen::Isometry2d &T_try) const {
    DCHECK(problem_ptr);
    double parameters[3] = {T_try(0, 2), T_try(1, 2), std::atan2(T_try(1, 0), T_try(0, 0))}; //
    double cost[1] = {0};
    problem_ptr->Evaluate(parameters, cost, nullptr);
    return -cost[0] / std::sqrt(auto_corr_src * auto_corr_tgt);
  }


  // T_tgt (should)= T_delta * T_src
  std::pair<double, Eigen::Isometry2d> calcCorrelation() {
    DCHECK(nullptr != problem_ptr);
    // gmmreg, rigid, 2D.
    double parameters[3] = {T_best_(0, 2), T_best_(1, 2),
                            std::atan2(T_best_(1, 0),
                                       T_best_(0, 0))}; // set according to the constellation output.

    ceres::GradientProblemSolver::Options options;
    options.minimizer_progress_to_stdout = HUMAN_READABLE;
    options.max_num_iterations = 10;
    ceres::GradientProblemSolver::Summary summary;
    ceres::Solve(options, *problem_ptr, parameters, &summary);

#if HUMAN_READABLE
    std::cout << summary.FullReport() << "\n";

    printf("Param after opt:\n");
    for (auto dat: parameters) {
      std::cout << dat << std::endl;
    }
#endif

    // normalize the score according to cell counts, and return the optimized parameter
//    Eigen::Isometry2d T_res;
    T_best_.setIdentity();
    T_best_.rotate(parameters[2]);
    T_best_.pretranslate(V2D(parameters[0], parameters[1]));

    double correlation = -summary.final_cost / std::sqrt(auto_corr_src * auto_corr_tgt);
//    printf("Correlation: %f\n", correlation);
    return {correlation, T_best_};

  }

  // TODO: evaluate metric estimation performance given the 3D gt poses
  static Eigen::Isometry2d evalMetricEst(const Eigen::Isometry2d &T_delta, const Eigen::Isometry3d &gt_src_3d,
                                         const Eigen::Isometry3d &gt_tgt_3d, const ContourManagerConfig &bev_config) {
    // ignore non-square resolution for now:
    CHECK_EQ(bev_config.reso_row_, bev_config.reso_col_);

    Eigen::Isometry2d T_so_ssen = Eigen::Isometry2d::Identity(), T_to_tsen;  // {}_sensor in {}_bev_origin frame
    T_so_ssen.translate(V2D(bev_config.n_row_ / 2 - 0.5, bev_config.n_col_ / 2 - 0.5));
    T_to_tsen = T_so_ssen;
    Eigen::Isometry2d T_tsen_ssen2_est = T_to_tsen.inverse() * T_delta * T_so_ssen;
    T_tsen_ssen2_est.translation() *= bev_config.reso_row_;
//    std::cout << "Estimated src in tgt sensor frame:\n" << T_tsen_ssen2_est.matrix() << std::endl;

    // Lidar sensor src in the lidar tgt frame, T_wc like.
    Eigen::Isometry3d T_tsen_ssen3 = gt_tgt_3d.inverse() * gt_src_3d;
//    std::cout << "gt src in tgt sensor frame 3d:\n" << T_tsen_ssen3.matrix() << std::endl;

    // TODO: project gt 3d into some gt 2d, and use
    Eigen::Isometry2d T_tsen_ssen2_gt;
    T_tsen_ssen2_gt.setIdentity();
    // for translation: just the xy difference
    // for rotation: rotate so that the two z axis align
    Eigen::Vector3d z0(0, 0, 1);
    Eigen::Vector3d z1 = T_tsen_ssen3.matrix().block<3, 1>(0, 2);
    Eigen::Vector3d ax = z0.cross(z1).normalized();
    double ang = acos(z0.dot(z1));
    Eigen::AngleAxisd d_rot(-ang, ax);

    Eigen::Matrix3d R_rectified = d_rot.matrix() * T_tsen_ssen3.matrix().topLeftCorner<3, 3>();  // only top 2x2 useful
//    std::cout << "R_rect:\n" << R_rectified << std::endl;
    CHECK_LT(R_rectified.row(2).norm(), 1 + 1e-3);
    CHECK_LT(R_rectified.col(2).norm(), 1 + 1e-3);

    T_tsen_ssen2_gt.rotate(std::atan2(R_rectified(1, 0), R_rectified(0, 0)));
    T_tsen_ssen2_gt.pretranslate(Eigen::Vector2d(T_tsen_ssen3.translation().segment(0, 2)));  // only xy

    std::cout << "T delta gt 2d:\n" << T_tsen_ssen2_gt.matrix() << std::endl;  // Note T_delta is not comparable to this

    Eigen::Isometry2d T_gt_est = T_tsen_ssen2_gt.inverse() * T_tsen_ssen2_est;
    return T_gt_est;
  }

  /// Get estimated transform between sensors.
  ///  T_tgt = T_delta * T_src, image orig frame, while this one is in sensor frame
  /// \param T_delta
  /// \param bev_config
  /// \return
  static Eigen::Isometry2d getEstSensTF(const Eigen::Isometry2d &T_delta, const ContourManagerConfig &bev_config) {
    // ignore non-square resolution for now:
    CHECK_EQ(bev_config.reso_row_, bev_config.reso_col_);

    Eigen::Isometry2d T_so_ssen = Eigen::Isometry2d::Identity(), T_to_tsen;  // {}_sensor in {}_bev_origin frame
    T_so_ssen.translate(V2D(bev_config.n_row_ / 2 - 0.5, bev_config.n_col_ / 2 - 0.5));
    T_to_tsen = T_so_ssen;
    Eigen::Isometry2d T_tsen_ssen2_est = T_to_tsen.inverse() * T_delta * T_so_ssen; // sensor in sensor frame: dist
    return T_tsen_ssen2_est;
  }

};

//! Full bev correlation
class BEVCorrelation {

};

#endif //CONT2_CORRELATION_H
