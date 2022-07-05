//
// Created by lewis on 5/6/22.
//

#include "cont2/contour_db.h"

void ContourDB::queryKNN(const ContourManager &q_cont, std::vector<std::shared_ptr<const ContourManager>> &cand_ptrs,
                         std::vector<KeyFloatType> &dist_sq) const {
  cand_ptrs.clear();
  dist_sq.clear();

  // for each layer
  std::vector<std::pair<IndexOfKey, KeyFloatType>> res_container;
  for (int ll = 0; ll < q_levels_.size(); ll++) {
    size_t size_beg = res_container.size();
    for (const auto &permu_key: q_cont.getLevRetrievalKey(q_levels_[ll])) {
      if (permu_key.sum() != 0) {
        std::vector<std::pair<IndexOfKey, KeyFloatType>> tmp_res;

        layer_db_[ll].layerKNNSearch(permu_key, cfg_.max_candi_per_layer_, cfg_.max_dist_sq_, tmp_res);
        res_container.insert(res_container.end(), tmp_res.begin(),
                             tmp_res.end());  // Q:different thres for different levels?
      }
    }
    // limit number of returned values in layer
    std::sort(res_container.begin() + size_beg, res_container.end(),
              [&](const std::pair<IndexOfKey, KeyFloatType> &a, const std::pair<IndexOfKey, KeyFloatType> &b) {
                return a.second < b.second;
              });
    if (res_container.size() > cfg_.max_candi_per_layer_ + size_beg)
      res_container.resize(cfg_.max_candi_per_layer_ + size_beg, res_container[0]);

  }

//    // limit number of returned values as whole
//    std::sort(res_container.begin(), res_container.end(),
//              [&](const std::pair<size_t, KeyFloatType> &a, const std::pair<size_t, KeyFloatType> &b) {
//                return a.second < b.second;
//              });
//    if (res_container.size() > cfg_.max_total_candi_)
//      res_container.resize(cfg_.max_total_candi_);

  std::set<size_t> used_gidx;
  printf("Query dists_sq: ");
  for (const auto &res: res_container) {
    if (used_gidx.find(res.first.gidx) == used_gidx.end()) {
      used_gidx.insert(res.first.gidx);
      cand_ptrs.emplace_back(all_bevs_[res.first.gidx]);
      dist_sq.emplace_back(res.second);
      printf("%7.4f", res.second);
    }
  }
  printf("\n");


  // find which bucket ranges does the query fall in

  // query and accumulate results

}
