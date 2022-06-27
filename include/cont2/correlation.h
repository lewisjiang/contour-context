//
// Created by lewis on 6/21/22.
//

#ifndef CONT2_CORRELATION_H
#define CONT2_CORRELATION_H

#include "cont2/contour_mng.h"
#include <ceres/ceres.h>

#include <utility>


struct ConstellCorrelationConfig {
  float max_corr_dist_ = 10.0; // in bev pixels
  float min_area_perc_ = 0.80; // minimal percentage of area involved for each layer
  std::vector<int> levels_ = {1, 2, 3, 4}; // the layers to be considered in ellipse gmm.
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

  GMMPair(const ContourManager &cm_src, const ContourManager &cm_tgt, const ConstellCorrelationConfig &config,
          const Eigen::Isometry2d &T_init) {
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
      src_cell_cnts_.emplace_back(cm_src.getLevCnt(lev));
      tgt_cell_cnts_.emplace_back(cm_tgt.getLevCnt(lev));
    }

    // pre select (need initial guess)
    for (int li = 0; li < ellipses_src.size(); li++) {
      for (int si = 0; si < ellipses_src[li].size(); si++) {
        for (int ti = 0; ti < ellipses_tgt[li].size(); ti++) {
          Eigen::Matrix<double, 2, 1> delta_mu = T_init * ellipses_src[li][si].mu_ - ellipses_tgt[li][ti].mu_;
          if (delta_mu.norm() < 3.0 * (max_majax_src[li][si] + max_majax_tgt[li][ti])) {  // close enough to correlate
            selected_pair_idx_[li].emplace_back(si, ti);
          }
        }
      }
    }
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
        Eigen::Matrix<T, 2, 2> new_cov =
            R * ellipses_src[li][pr.first].cov_ * R.transpose() + ellipses_tgt[li][pr.second].cov_;
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
  ConstellCorrelationConfig cfg_;

public:
  ConstellCorrelation() = default;

  ConstellCorrelation(ConstellCorrelationConfig cfg) : cfg_(std::move(cfg)) {};

  // T_tgt = T_delta * T_src
  float calcCorrelation(const ContourManager &cm_src, const ContourManager &cm_tgt, const Eigen::Isometry2d &T_init) {
    // gmmreg, rigid, 2D.
    double parameters[3] = {T_init.translation().x(), T_init.translation().y(),
                            atan2(T_init.rotation()(1, 0),
                                  T_init.rotation()(0, 0))}; // set according to the constellation output.
    printf("Param before opt:\n");
    for (auto dat: parameters) {
      std::cout << dat << std::endl;
    }
    ceres::GradientProblem problem(
        new ceres::AutoDiffFirstOrderFunction<GMMPair, 3>(new GMMPair(cm_src, cm_tgt, cfg_, T_init)));

    ceres::GradientProblemSolver::Options options;
    options.minimizer_progress_to_stdout = true;
//  options.max_num_iterations = 8;
    ceres::GradientProblemSolver::Summary summary;
    ceres::Solve(options, problem, parameters, &summary);

    std::cout << summary.FullReport() << "\n";

    printf("Param after opt:\n");
    for (auto dat: parameters) {
      std::cout << dat << std::endl;
    }

    // TODO: normalize the score according to cell counts, and return the optimized parameter
  }

  // TODO: evaluate metric estimation performance given the 3D gt poses
  void evalMetricEst(const Eigen::Isometry2d &T_delta, const Eigen::Isometry3d &gt_src_3d,
                     const Eigen::Isometry3d &gt_tgt_3d) {

  }

};

//! Full bev correlation
class BEVCorrelation {

};

#endif //CONT2_CORRELATION_H
