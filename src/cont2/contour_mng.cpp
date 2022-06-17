//
// Created by lewis on 5/5/22.
//

#include "cont2/contour_mng.h"

void ContourManager::saveContours(const std::string &fpath,
                                  const std::vector<std::vector<std::shared_ptr<ContourView>>> &cont_views) {
  // 0:level, 1:cell_cnt, 2:pos_mean, 4:pos_cov, 8:eig_vals, eig_vecs(10), 14:eccen, 15:vol3_mean, 16:com, 18,19:..
  // Note that recording data as strings has accuracy loss
//    std::string fpath = sav_dir + "/contours_" + str_id_ + ".txt";
  std::fstream res_file(fpath, std::ios::out);

  if (res_file.rdstate() != std::ifstream::goodbit) {
    std::cerr << "Error opening " << fpath << std::endl;
    return;
  }
  printf("Writing results to file \"%s\" ...", fpath.c_str());
  res_file << "\nDATA_START\n";
  for (const auto &layer: cont_views) {
    for (const auto &cont: layer) {
      res_file << cont->level_ << '\t';
      res_file << cont->cell_cnt_ << '\t';

      res_file << cont->pos_mean_.x() << '\t' << cont->pos_mean_.y() << '\t';
      for (int i = 0; i < 4; i++)
        res_file << cont->pos_cov_.data()[i] << '\t';

      res_file << cont->eig_vals_.x() << '\t' << cont->eig_vals_.y() << '\t';
      for (int i = 0; i < 4; i++)
        res_file << cont->eig_vecs_.data()[i] << '\t';

      res_file << cont->eccen_ << '\t';
      res_file << cont->vol3_mean_ << '\t';
      res_file << cont->com_.x() << '\t' << cont->com_.y() << '\t';

      res_file << int(cont->ecc_feat_) << '\t';
      res_file << int(cont->com_feat_) << '\t';

      res_file << '\n';
    }
  }
  res_file << "DATA_END\n";
  res_file.close();
  printf("Writing results finished.\n");

}

void ContourManager::makeContours() {
//    float h_min = -VAL_ABS_INF_;
  cv::Mat last_label_img;
  int16_t lev = 0;
  for (const auto &h_min: cfg_.lv_grads_) {
    printf("Height [%f, +]\n", h_min);
    // clamp image
    if (cont_views_.empty()) {
      cv::Mat mask, mask_u8;
      cv::threshold(bev_, mask, h_min, 255, cv::THRESH_BINARY); // mask is same type and dimension as bev_
      // 1. select points higher than a threshold
      mask.convertTo(mask_u8, CV_8U);

      cv::imwrite("cart_context-mask-" + std::to_string(lev) + "-" + str_id_ + ".png", mask_u8);

      // 2. calculate connected blobs
      cv::Mat1i labels, stats;  // int (CV_32S)
      cv::Mat centroids;
      cv::connectedComponentsWithStats(mask_u8, labels, stats, centroids, 8, CV_32S);

      // // aux: show image contour group
      cv::Mat label_img;
      cv::normalize(labels, label_img, 0, 255, cv::NORM_MINMAX);
      cv::imwrite("cart_context-labels-" + std::to_string(lev) + "-" + str_id_ + ".png", label_img);
      cv::imwrite("cart_context-mask-" + std::to_string(lev) + "-" + str_id_ + ".png", mask_u8);

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

        RunningStatRecorder tmp_rec;
        int poi_r = -1, poi_c = -1;

        for (int i = rect.y; i < rect.y + rect.height; i++)
          for (int j = rect.x; j < rect.x + rect.width; j++)
            if (bev_(i, j) > h_min) {  // consistent with opencv threshold: if src(x,y)>thresh, ...
              tmp_rec.runningStats(i, j, bev_(i, j));
              poi_r = i;
              poi_c = j;
            }

//        std::shared_ptr<ContourView> ptr_tmp_cv(new ContourView(lev, poi_r, poi_c, nullptr));
        std::shared_ptr<ContourView> ptr_tmp_cv(new ContourView(lev, poi_r, poi_c));
        ptr_tmp_cv->calcStatVals(tmp_rec, view_stat_cfg_);
        DCHECK(ptr_tmp_cv->cell_cnt_ == stats(n, 4));
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

void ContourManager::saveContourImage(const std::string &fpath, int level) const {
  CHECK(!bev_.empty());
  cv::imwrite(fpath, getContourImage(level));
}

std::pair<Eigen::Isometry2d, bool>
ContourManager::calcScanCorresp(const ContourManager &src, const ContourManager &tgt) {
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
          V2F cent_s = (sc1->pos_mean_ - sc2->pos_mean_).normalized();
          V2F cent_t = (tc1->pos_mean_ - tc2->pos_mean_).normalized();
          if (sc1->ecc_feat_ && tc1->ecc_feat_) {
            float theta_s = std::acos(cent_s.transpose() * sc1->eig_vecs_.col(1));   // acos: [0,pi)
            float theta_t = std::acos(cent_t.transpose() * tc1->eig_vecs_.col(1));
            if (diff_delt<float>(theta_s, theta_t, M_PI / 12) && diff_delt<float>(M_PI - theta_s, theta_t, M_PI / 12))
              continue;
          }
          if (sc2->ecc_feat_ && tc2->ecc_feat_) {
            float theta_s = std::acos(cent_s.transpose() * sc2->eig_vecs_.col(1));   // acos: [0,pi)
            float theta_t = std::acos(cent_t.transpose() * tc2->eig_vecs_.col(1));
            if (diff_delt<float>(theta_s, theta_t, M_PI / 6) && diff_delt<float>(M_PI - theta_s, theta_t, M_PI / 6))
              continue;
          }

          // 4. PROSAC
          // 4.1 get the rough transform to facilitate the similarity check (relatively large acceptance range)
          // can come from a naive 2 point transform estimation or a gmm2gmm
          Eigen::Matrix3d T_delta = estimateTF<double>(sc1->pos_mean_.cast<double>(), sc2->pos_mean_.cast<double>(),
                                                       tc1->pos_mean_.cast<double>(),
                                                       tc2->pos_mean_.cast<double>()).matrix(); // naive 2 point estimation

          // for pointset transform estimation
          Eigen::Matrix<double, 2, Eigen::Dynamic> pointset1; // src
          Eigen::Matrix<double, 2, Eigen::Dynamic> pointset2; // tgt
          pointset1.resize(2, 2);
          pointset2.resize(2, 2);
          pointset1.col(0) = sc1->pos_mean_.cast<double>();
          pointset1.col(1) = sc2->pos_mean_.cast<double>();
          pointset2.col(0) = tc1->pos_mean_.cast<double>();
          pointset2.col(1) = tc2->pos_mean_.cast<double>();

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
              V2F pos_mean_src_tf = T_delta.block<2, 2>(0, 0).cast<float>() * src.cont_views_[l][ii]->pos_mean_
                                    + T_delta.block<2, 1>(0, 2).cast<float>();
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
              pointset1.rightCols(1) = src.cont_views_[l][ii]->pos_mean_.cast<double>();
              pointset2.rightCols(1) = tgt.cont_views_[l][jj]->pos_mean_.cast<double>();
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

void ContourManager::makeContourRecursiveHelper(const cv::Rect &cc_roi, const cv::Mat1b &cc_mask, int level,
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
    if (stats(n, 4) < cfg_.min_cont_cell_cnt_)  // ignore contours that are too small
      continue;

    //Rectangle around the connected component
    // Rect: col0, row0, n_col, n_row
    cv::Rect rect_g(stats(n, 0) + cc_roi.x, stats(n, 1) + cc_roi.y, stats(n, 2), stats(n, 3)); // global: on bev
    cv::Rect rect_l(stats(n, 0), stats(n, 1), stats(n, 2), stats(n, 3)); // local: relative to bev_roi

    cv::Mat1b mask_n = labels(rect_l) == n;

    RunningStatRecorder tmp_rec;
    int poi_r = -1, poi_c = -1;  // the point(r,c) coordinate on the global bev for the contour

    for (int i = 0; i < rect_l.height; i++)
      for (int j = 0; j < rect_l.width; j++)
        if (mask_n(i, j)) {
          poi_r = i + rect_g.y;
          poi_c = j + rect_g.x;
//            tmp_rec.runningStats(i + rect_g.y, j + rect_g.x, bev_(i + rect_g.y, j + rect_g.x)); // discrete
//          V2F c_point = pillar_pos2f_.at(poi_r * cfg_.n_col_ + poi_c);

          int q_hash = poi_r * cfg_.n_col_ + poi_c;
          std::pair<int, Pixelf> sear_res = search_vec<Pixelf>(bev_pixfs_, 0,
                                                               (int) bev_pixfs_.size() - 1, q_hash);
          DCHECK_EQ(sear_res.first, q_hash);
          tmp_rec.runningStatsF(sear_res.second.row_f, sear_res.second.col_f, bev_(poi_r, poi_c)); // continuous
//          tmp_rec.runningStatsF(c_point.x(), c_point.y(), bev_(poi_r, poi_c)); // continuous
        }

//    std::shared_ptr<ContourView> ptr_tmp_cv(new ContourView(level, poi_r, poi_c, parent));
    std::shared_ptr<ContourView> ptr_tmp_cv(new ContourView(level, poi_r, poi_c));
    ptr_tmp_cv->calcStatVals(tmp_rec, view_stat_cfg_);
    DCHECK(ptr_tmp_cv->cell_cnt_ == stats(n, 4));
    cont_views_[level].emplace_back(ptr_tmp_cv);    // add to the manager's matrix
//    if (parent)
//      parent->children_.emplace_back(ptr_tmp_cv);

    // recurse
    // Get the mask for the contour

//      printf("contour ROI: %d, %d, level: %d\n", mask_n.rows, mask_n.cols, level);
    makeContourRecursiveHelper(rect_g, mask_n, level + 1, ptr_tmp_cv);

//      if (level == 2) {
//        cv::bitwise_or(mask_n, visualization(rect_g), visualization(rect_g));
//      }

  }

}
