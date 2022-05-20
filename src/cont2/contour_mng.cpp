//
// Created by lewis on 5/5/22.
//

#include "cont2/contour_mng.h"

void ContourManager::saveContours(const std::string &fpath) const {
  // 0:level, 1:cell_cnt, 2:pos_mean, 4:pos_cov, 8:eig_vals, eig_vecs(10), 14:eccen, 15:vol3_mean, 16:com, 18,19:..
  // Note that recording data as strings has accuracy loss
//    std::string fpath = sav_dir + "/contours_" + str_id + ".txt";
  std::fstream res_file(fpath, std::ios::out);

  if (res_file.rdstate() != std::ifstream::goodbit) {
    std::cerr << "Error opening " << fpath << std::endl;
    return;
  }
  printf("Writing results to file \"%s\" \n", fpath.c_str());
  res_file << "\nDATA_START\n";
  for (const auto &layer: cont_views_) {
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
  int lev = 0;
  for (const auto &h_min: cfg_.lv_grads_) {
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
