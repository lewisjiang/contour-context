//
// Created by lewis on 5/6/22.
//

#ifndef CONT2_CONTOUR_DB_H
#define CONT2_CONTOUR_DB_H

#include "cont2/contour_mng.h"

#include <memory>
#include <algorithm>
#include <set>
#include <unordered_set>

#include <nanoflann.hpp>
#include "KDTreeVectorOfVectorsAdaptor.h"

//typedef Eigen::Matrix<KeyFloatType, 4, 1> tree_key_t;
typedef std::vector<RetrievalKey> my_vector_of_vectors_t;
typedef KDTreeVectorOfVectorsAdaptor<my_vector_of_vectors_t, KeyFloatType> my_kd_tree_t;

const KeyFloatType MAX_BACKET_VAL = 1000.0f;
const KeyFloatType MAX_DIST_SQ = 1e6;

struct RetrievalMetric {

};

struct TreeBucketConfig {
  double max_elapse_ = 10.0;  // the max spatial/temporal delay before adding to the trees
  double min_elapse_ = 5.0;  // the min spatial/temporal delay to wait before adding to the trees
};

struct TreeBucket {
  struct BufferTriplet {
    RetrievalKey pt;
    double ts{};
    size_t gidx{};

    BufferTriplet() = default;

    BufferTriplet(const RetrievalKey &_a, double _b, size_t _c) : pt(_a), ts(_b), gidx(_c) {}
  };

  const TreeBucketConfig cfg_;

  KeyFloatType buc_beg_{}, buc_end_{};  // [beg, end)
  my_vector_of_vectors_t data_tree_;
  std::shared_ptr<my_kd_tree_t> tree_ptr = nullptr;
  std::vector<BufferTriplet> buffer_;    // ordered, ascending
  std::vector<size_t> gidx_tree_;  // global index of ContourManager in the whole database

  TreeBucket(const TreeBucketConfig &config, KeyFloatType beg, KeyFloatType end) : cfg_(config), buc_beg_(beg),
                                                                                   buc_end_(end) {}

  size_t getTreeSize() const {
    return data_tree_.size();
  }

  void pushBuffer(const RetrievalKey &tree_key, double ts, size_t gidx) {
    buffer_.emplace_back(tree_key, ts, gidx);
  }

  inline bool needPopBuffer(double curr_ts) const {
    double ts_overflow = curr_ts - cfg_.max_elapse_;
    if (buffer_.empty() || buffer_[0].ts > ts_overflow)  // rebuild every (max-min) sec, ignore newest min.
      return false;
    return true;
  }

  inline void rebuildTree() {
    tree_ptr = std::make_shared<my_kd_tree_t>(RetrievalKey::SizeAtCompileTime /*dim*/, data_tree_,
                                              10 /* max leaf */ );
  }

  /// Pop max possible from the buffer into the tree, and rebuild the tree
  /// \param curr_ts
  void popBufferMax(double curr_ts) {
    double ts_cutoff = curr_ts - cfg_.min_elapse_;
    int gap = 0;
    for (; gap < buffer_.size(); gap++) {
      if (buffer_[gap].ts >= ts_cutoff) {
        break;
      }
    }

    if (gap > 0) {
      size_t sz0 = data_tree_.size();
      data_tree_.resize(sz0 + gap);
      gidx_tree_.resize(sz0 + gap);
      for (size_t i = 0; i < gap; i++) {
        data_tree_[i + sz0] = buffer_[i].pt;
        gidx_tree_[i + sz0] = buffer_[i].gidx;
      }
      buffer_.erase(buffer_.begin(), buffer_.begin() + gap);

      rebuildTree();
    }
  }

  void
  knnSearch(const int num_res, std::vector<size_t> &ret_idx, std::vector<KeyFloatType> &out_dist_sq,
            RetrievalKey q_key) const {
    ret_idx.resize(num_res);
    out_dist_sq.resize(num_res);
    std::fill(out_dist_sq.begin(), out_dist_sq.end(), MAX_DIST_SQ);   // in case of fewer points in the tree than k

    if (!tree_ptr) { return; }

    nanoflann::KNNResultSet<KeyFloatType> resultSet(num_res);
    resultSet.init(&ret_idx[0], &out_dist_sq[0]);

    tree_ptr->index->findNeighbors(resultSet, q_key.data(), nanoflann::SearchParams(10));
    for (size_t i = 0; i < num_res; i++) {
      if (out_dist_sq[i] >= 0)
        ret_idx[i] = gidx_tree_[i];
    }
  }

};


struct LayerDB {
  const int min_elem_split_ = 100;
  const double imba_diff_ratio_ = 0.2; // if abs(size(i)-size(i+1))>ratio * max(,), we need to balance the two trees.
  const int max_num_backets_ = 6;
  const int bucket_chann_ = 0; // the #th dimension of the retrieval key that is used as buckets.

  std::vector<TreeBucket> buckets_;

  std::vector<KeyFloatType> bucket_ranges_;  // [min, max) pairs for buckets' range
  LayerDB() {
    bucket_ranges_.resize(max_num_backets_ + 1);
    bucket_ranges_.front() = -MAX_BACKET_VAL;
    bucket_ranges_.back() = MAX_BACKET_VAL;
    buckets_.emplace_back(TreeBucketConfig(), -MAX_BACKET_VAL, MAX_BACKET_VAL);

    // empty buckets
    for (int i = 1; i < max_num_backets_; i++) {
      bucket_ranges_[i] = MAX_BACKET_VAL;
      buckets_.emplace_back(TreeBucketConfig(), MAX_BACKET_VAL, MAX_BACKET_VAL);
    }
  }

  // add buffer
  void pushBuffer(const RetrievalKey &layer_key, double ts, size_t scan_gidx) {
    for (int i = 0; i < max_num_backets_; i++) {
      if (bucket_ranges_[i] <= layer_key(bucket_chann_) && layer_key(bucket_chann_) < bucket_ranges_[i + 1]) {
        if (layer_key.sum() != 0) // if an all zero key, we do not add it.
          buckets_[i].pushBuffer(layer_key, ts, scan_gidx);
        return;
      }
    }
  }

  // TODO: rebalance and add buffer to the tree
  // Assumption: rebuild in turn instead of rebuild all at once. tr1 and tr2 are adjacent, tr1 has a lower bucket range.
//  void rebuild(int seed, double curr_ts) {
  void rebuild(int idx_t1, double curr_ts) {
//    int idx_t1 = std::abs(seed) % (2 * (max_num_backets_ - 2));
//    if (idx_t1 > (max_num_backets_ - 2))
//      idx_t1 = 2 * (max_num_backets_ - 2) - idx_t1;

    DCHECK_LT(idx_t1, buckets_.size() - 1);
    TreeBucket &tr1 = buckets_[idx_t1], &tr2 = buckets_[idx_t1 + 1];

    DCHECK_EQ(tr1.buc_end_, tr2.buc_beg_);
    bool pb1 = tr1.needPopBuffer(curr_ts), pb2 = tr2.needPopBuffer(curr_ts); // if we need to pop buffer
    if (!pb1 && !pb2)
      return;  // rebuild is scheduled at the time when we pop the buffer

    int sz1 = tr1.getTreeSize(), sz2 = tr2.getTreeSize();
    double diff_ratio = 1.0 * std::abs(sz1 - sz2) / std::max(sz1, sz2);
    if (pb1 && !pb2 && (diff_ratio < imba_diff_ratio_ || std::max(sz1, sz2) < min_elem_split_)) {
      tr1.popBufferMax(curr_ts);
      return;
    }

    if (!pb1 && pb2 && (diff_ratio < imba_diff_ratio_ || std::max(sz1, sz2) < min_elem_split_)) {
      tr2.popBufferMax(curr_ts);
      return;
    }
    // else, balance two trees
    // 1. find new boundary/pivot.
    // q: How to find the elements to move in O(n)
    //  a1: flatten and rebuild?
    //  a2: sort the larger one, and copy parts to the smaller one.

    if (diff_ratio < 0.5 * imba_diff_ratio_) {
      if (pb1) tr1.popBufferMax(curr_ts);
      if (pb2) tr2.popBufferMax(curr_ts);
      return;
    }

    printf(" (m->)"); // merging here
    if (sz1 > sz2) {
//      DCHECK_GE(sz1, min_elem_split_);
      int to_move_max = int((sz1 - sz2 + imba_diff_ratio_ * sz2) / (2 - imba_diff_ratio_));
      int to_move_mid = int((sz1 - sz2) / 2.0);
      int to_move_min = std::max(0, int((sz1 - sz2 - imba_diff_ratio_ * sz1) / (2 - imba_diff_ratio_)));
      DCHECK_GE(to_move_max, to_move_mid);
      DCHECK_GE(to_move_mid, to_move_min);

      std::vector<int> sort_permu(sz1);
      std::iota(sort_permu.begin(), sort_permu.end(), 0);
      std::sort(sort_permu.begin(), sort_permu.end(), [&](const int &a, const int &b) {
        return tr1.data_tree_[a](bucket_chann_) < tr1.data_tree_[b](bucket_chann_);
      }); // not actually moving data positions

      int num_to_move = 0;
      KeyFloatType split_val = tr1.buc_end_; // tree1: ( , split val], tree2: [split val, )
      if (tr1.data_tree_[sort_permu[sz1 - to_move_mid]](bucket_chann_) !=
          tr1.data_tree_[sort_permu[sz1 - to_move_mid - 1]](bucket_chann_)) { // if no contagious value across split
        num_to_move = to_move_mid;
        split_val = tr1.data_tree_[sort_permu[sz1 - to_move_mid]](bucket_chann_);
      } else {
        KeyFloatType contagious_val = tr1.data_tree_[sort_permu[sz1 - to_move_mid]](bucket_chann_);
        int i = to_move_mid - 1;
        for (; i > to_move_min; i--) {  // look into larger size
          if (tr1.data_tree_[sort_permu[sz1 - i]](bucket_chann_) != contagious_val) {  // found the split position
            num_to_move = i;
            split_val = tr1.data_tree_[sort_permu[sz1 - i]](bucket_chann_);
            break;
          }
        }
        if (num_to_move == 0) {
          i = to_move_mid + 1;
          for (; i < to_move_max; i++) {  // look into smaller side
            if (tr1.data_tree_[sort_permu[sz1 - i]](bucket_chann_) != contagious_val) {  // found the split position
              num_to_move = i - 1;
              split_val = contagious_val;
              break;
            }
          }
        }
      }

      if (num_to_move == 0) { // should rebalance but cannot, due to a strip of contagious bucket values
        printf("Cannot split due to contagious values A");
        tr1.popBufferMax(curr_ts);
        if (pb2) tr2.popBufferMax(curr_ts);
        return;
      }

      for (int i = 0; i < num_to_move; i++) {  // add to shorter tree
        tr2.data_tree_.emplace_back(tr1.data_tree_[sort_permu[sz1 - i - 1]]);
        tr2.gidx_tree_.emplace_back(tr1.gidx_tree_[sort_permu[sz1 - i - 1]]);
      }

      int p_dat = sz1 - 1, p_perm = sz1 - 1;
      for (; p_perm >= sz1 - num_to_move; p_perm--) {  // rotate to the back;
        while (tr1.data_tree_[p_dat](bucket_chann_) >= split_val)
          p_dat--;
        if (sort_permu[p_perm] < p_dat) {
          std::swap(tr1.data_tree_[p_dat], tr1.data_tree_[sort_permu[p_perm]]);
          std::swap(tr1.gidx_tree_[p_dat], tr1.gidx_tree_[sort_permu[p_perm]]);
          p_dat--;
        }
      }
      DCHECK_EQ(p_dat + num_to_move, sz1 - 1);
      tr1.data_tree_.resize(p_dat + 1);

      // move buffer: merge and redistribute
      int p1 = 0, p2 = tr1.buffer_.size() - 1, sz_rem;
      while (p1 <= p2) {
        if (tr1.buffer_[p1].pt(bucket_chann_) >= split_val && tr1.buffer_[p2].pt(bucket_chann_) < split_val) {
          std::swap(tr1.buffer_[p1], tr1.buffer_[p2]);
          p1++;
          p2--;
        } else {
          if (tr1.buffer_[p2].pt(bucket_chann_) >= split_val)
            p2--;
          if (tr1.buffer_[p1].pt(bucket_chann_) < split_val)
            p1++;
        }
      }
      sz_rem = p2 + 1;
      tr2.buffer_.insert(tr2.buffer_.end(), tr1.buffer_.begin() + sz_rem, tr1.buffer_.end());
      tr1.buffer_.resize(sz_rem);
      tr1.buc_end_ = tr2.buc_beg_ = split_val;
      bucket_ranges_[idx_t1 + 1] = split_val;


    } else { // if tree 1 is shorter and move elements from tree 2 to tree 1:
//      DCHECK_GE(sz2, min_elem_split_);
      int to_move_max = int((sz2 - sz1 + imba_diff_ratio_ * sz1) / (2 - imba_diff_ratio_));
      int to_move_mid = int((sz2 - sz1) / 2.0);
      int to_move_min = std::max(0, int((sz2 - sz1 - imba_diff_ratio_ * sz2) / (2 - imba_diff_ratio_)));
      DCHECK_GE(to_move_max, to_move_mid);
      DCHECK_GE(to_move_mid, to_move_min);

      std::vector<int> sort_permu(sz2);
      std::iota(sort_permu.begin(), sort_permu.end(), 0);
      std::sort(sort_permu.begin(), sort_permu.end(), [&](const int &a, const int &b) {
        return tr2.data_tree_[a](bucket_chann_) > tr2.data_tree_[b](bucket_chann_);
      }); // smaller at the back

      int num_to_move = 0;
      KeyFloatType split_val = tr1.buc_end_; // tree1: ( , split val], tree2: [split val, )
      if (tr2.data_tree_[sort_permu[sz2 - to_move_mid]](bucket_chann_) !=
          tr2.data_tree_[sort_permu[sz2 - to_move_mid - 1]](bucket_chann_)) { // if no contagious value across split
        num_to_move = to_move_mid;
        split_val = tr2.data_tree_[sort_permu[sz2 - to_move_mid - 1]](bucket_chann_);
      } else {
        KeyFloatType contagious_val = tr2.data_tree_[sort_permu[sz2 - to_move_mid]](bucket_chann_);
        int i = to_move_mid - 1;
        for (; i > to_move_min; i--) {  // look into smaller size
          if (tr2.data_tree_[sort_permu[sz2 - i]](bucket_chann_) != contagious_val) {  // found the split position
            num_to_move = i;
            split_val = contagious_val;
            break;
          }
        }
        if (num_to_move == 0) {
          i = to_move_mid + 1;
          for (; i < to_move_max; i++) {  // look into larger buc side
            if (tr2.data_tree_[sort_permu[sz2 - i]](bucket_chann_) != contagious_val) {  // found the split position
              num_to_move = i - 1;
              split_val = tr2.data_tree_[sort_permu[sz2 - i]](bucket_chann_);
              break;
            }
          }
        }
      }

      if (num_to_move == 0) { // should rebalance but cannot, due to a strip of contagious bucket values
        printf("Cannot split due to contagious values B");
        if (pb1) tr1.popBufferMax(curr_ts);
        tr2.popBufferMax(curr_ts);
        return;
      }

      for (int i = 0; i < num_to_move; i++) {  // add to shorter tree
        tr1.data_tree_.emplace_back(tr2.data_tree_[sort_permu[sz2 - i - 1]]);
        tr1.gidx_tree_.emplace_back(tr2.gidx_tree_[sort_permu[sz2 - i - 1]]);
      }

      int p_dat = sz2 - 1, p_perm = sz2 - 1;
      for (; p_perm >= sz2 - num_to_move; p_perm--) {  // rotate to the back;
        while (tr2.data_tree_[p_dat](bucket_chann_) < split_val)
          p_dat--;
        if (sort_permu[p_perm] < p_dat) {
          std::swap(tr2.data_tree_[p_dat], tr2.data_tree_[sort_permu[p_perm]]);
          std::swap(tr2.gidx_tree_[p_dat], tr2.gidx_tree_[sort_permu[p_perm]]);
          p_dat--;
        }
      }
      DCHECK_EQ(p_dat + num_to_move, sz2 - 1);
      tr2.data_tree_.resize(p_dat + 1);

      // move buffer: merge and redistribute
      int p1 = 0, p2 = tr2.buffer_.size() - 1, sz_rem;
      while (p1 <= p2) {
        if (tr2.buffer_[p1].pt(bucket_chann_) < split_val && tr2.buffer_[p2].pt(bucket_chann_) >= split_val) {
          std::swap(tr2.buffer_[p1], tr2.buffer_[p2]);
          p1++;
          p2--;
        } else {
          if (tr2.buffer_[p2].pt(bucket_chann_) < split_val)
            p2--;
          if (tr2.buffer_[p1].pt(bucket_chann_) >= split_val)
            p1++;
        }
      }
      sz_rem = p2 + 1;
      tr1.buffer_.insert(tr1.buffer_.end(), tr2.buffer_.begin() + sz_rem, tr2.buffer_.end());
      tr2.buffer_.resize(sz_rem);
      tr1.buc_end_ = tr2.buc_beg_ = split_val;
      bucket_ranges_[idx_t1 + 1] = split_val;

    }

    for (const auto &dat: tr1.data_tree_) {
      DCHECK_LT(dat(bucket_chann_), tr1.buc_end_);
      DCHECK_GE(dat(bucket_chann_), tr1.buc_beg_);
    }
    for (const auto &dat: tr1.buffer_) {
      DCHECK_LT(dat.pt(bucket_chann_), tr1.buc_end_);
      DCHECK_GE(dat.pt(bucket_chann_), tr1.buc_beg_);
    }

    for (const auto &dat: tr2.data_tree_) {
      DCHECK_LT(dat(bucket_chann_), tr2.buc_end_);
      DCHECK_GE(dat(bucket_chann_), tr2.buc_beg_);
    }
    for (const auto &dat: tr2.buffer_) {
      DCHECK_LT(dat.pt(bucket_chann_), tr2.buc_end_);
      DCHECK_GE(dat.pt(bucket_chann_), tr2.buc_beg_);
    }

    // q: what if all the keys have the same bucket element?
    //  a: calc the number allowed for removal in the larger one, if moving the block will cross the line (another
    //  imba), then do not move.

    // 2. move across boundary

    // 3. rebuild trees
    std::sort(tr1.buffer_.begin(), tr1.buffer_.end(), [&](const auto &a, const auto &b) {
      return a.ts < b.ts;
    });

    std::sort(tr2.buffer_.begin(), tr2.buffer_.end(), [&](const auto &a, const auto &b) {
      return a.ts < b.ts;
    });

    tr1.popBufferMax(curr_ts);
    tr2.popBufferMax(curr_ts);

  }

  // TODO: query
  void
  layerKNNSearch(const RetrievalKey &q_key, const int k_top, const KeyFloatType max_dist_sq,
                 std::vector<std::pair<size_t, KeyFloatType>> &res_pairs) const {
    // the result vector can have fewer than k result pairs.

    int mid_bucket = 0;
    for (int i = 0; i < max_num_backets_; i++) {
      if (bucket_ranges_[i] <= q_key(bucket_chann_) && bucket_ranges_[i + 1] > q_key(bucket_chann_)) {
        mid_bucket = i;
        break;
      }
    }

    KeyFloatType max_dist_sq_run = max_dist_sq;


//    std::vector<std::pair<size_t, KeyFloatType>> res_pairs;
    res_pairs.clear();  // all pairs are meaningful, which may fall short of k_top.

    for (int i = 0; i < max_num_backets_; i++) {
      std::vector<size_t> tmp_gidx;
      std::vector<KeyFloatType> tmp_dists_sq;

      if (i == 0) {
        buckets_[mid_bucket].knnSearch(k_top, tmp_gidx, tmp_dists_sq, q_key);
        for (int j = 0; j < k_top; j++)
          if (tmp_dists_sq[j] < max_dist_sq_run)
            res_pairs.emplace_back(tmp_gidx[j], tmp_dists_sq[j]);
          else
            break;

      } else if (mid_bucket - i >= 0) {
        if ((q_key(bucket_chann_) - bucket_ranges_[mid_bucket - i + 1]) *
            (q_key(bucket_chann_) - bucket_ranges_[mid_bucket - i + 1]) > max_dist_sq_run) { continue; }
        buckets_[mid_bucket - i].knnSearch(k_top, tmp_gidx, tmp_dists_sq, q_key);
        for (int j = 0; j < k_top; j++)
          if (tmp_dists_sq[j] < max_dist_sq_run)
            res_pairs.emplace_back(tmp_gidx[j], tmp_dists_sq[j]);
          else
            break;

      } else if (mid_bucket + i < max_num_backets_) {  // query a
        if ((q_key(bucket_chann_) - bucket_ranges_[mid_bucket + i]) *
            (q_key(bucket_chann_) - bucket_ranges_[mid_bucket + i]) > max_dist_sq_run) { continue; }
        buckets_[mid_bucket + i].knnSearch(k_top, tmp_gidx, tmp_dists_sq, q_key);
        for (int j = 0; j < k_top; j++)
          if (tmp_dists_sq[j] < max_dist_sq_run)
            res_pairs.emplace_back(tmp_gidx[j], tmp_dists_sq[j]);
          else
            break;

      }
      std::sort(res_pairs.begin(), res_pairs.end(),
                [&](const std::pair<size_t, KeyFloatType> &a, const std::pair<size_t, KeyFloatType> &b) {
                  return a.second < b.second;
                });
      if (res_pairs.size() >= k_top) {
        res_pairs.resize(k_top);
        max_dist_sq_run = res_pairs.back().second;
      }
    }

//    ret_gidx.resize(k_top);
//    dists_sq.resize(k_top);
//    std::fill(dists_sq.begin(), dists_sq.end(), MAX_DIST_SQ);
//    for (int i = 0; i < res_pairs.size(); i++) {
//      ret_gidx[i] = res_pairs[i].first;
//      dists_sq[i] = res_pairs[i].second;
//    }

  }

};

struct ContourDBConfig {
  int num_trees_ = 6;  // max number of trees per layer
  int max_candi_per_layer_ = 40;  // should we use different values for different layers?
  int max_total_candi_ = 80;  // should we use different values for different layers?
  KeyFloatType max_dist_sq_ = 200.0;
};

// manages the whole database of contours for place re-identification
// top level database
class ContourDB {
  const ContourDBConfig cfg_;
  const std::vector<int> q_levels_;

  std::vector<LayerDB> layer_db_;
  std::vector<std::shared_ptr<ContourManager>> all_bevs_;

public:
  ContourDB(const ContourDBConfig &config, const std::vector<int> &q_levels) : cfg_(config), q_levels_(q_levels) {
    layer_db_.resize(q_levels_.size());
  }

  // TODO: 1. query database
  void queryCandidates(const ContourManager &q_cont,
                       std::vector<std::shared_ptr<ContourManager>> &cand_ptrs,
                       std::vector<KeyFloatType> &dist_sq) const {
    cand_ptrs.clear();
    dist_sq.clear();

    // for each layer
    std::vector<std::pair<size_t, KeyFloatType>> q_container;
    for (int ll = 0; ll < q_levels_.size(); ll++) {
      size_t size_beg = q_container.size();
      for (const auto &permu_key: q_cont.getRetrievalKey(q_levels_[ll])) {
        if (permu_key.sum() != 0) {
          std::vector<std::pair<size_t, KeyFloatType>> tmp_res;

          layer_db_[ll].layerKNNSearch(permu_key, cfg_.max_candi_per_layer_, cfg_.max_dist_sq_, tmp_res);
          q_container.insert(q_container.end(), tmp_res.begin(),
                             tmp_res.end());  // Q:different thres for different levels?
        }
      }
      // limit number of returned values in layer
      std::sort(q_container.begin() + size_beg, q_container.end(),
                [&](const std::pair<size_t, KeyFloatType> &a, const std::pair<size_t, KeyFloatType> &b) {
                  return a.second < b.second;
                });
      if (q_container.size() > cfg_.max_candi_per_layer_ + size_beg)
        q_container.resize(cfg_.max_candi_per_layer_ + size_beg);

    }

//    // limit number of returned values as whole
//    std::sort(q_container.begin(), q_container.end(),
//              [&](const std::pair<size_t, KeyFloatType> &a, const std::pair<size_t, KeyFloatType> &b) {
//                return a.second < b.second;
//              });
//    if (q_container.size() > cfg_.max_total_candi_)
//      q_container.resize(cfg_.max_total_candi_);

    std::set<size_t> used_gidx;
    printf("Query dists_sq: ");
    for (const auto &res: q_container) {
      if (used_gidx.find(res.first) == used_gidx.end()) {
        used_gidx.insert(res.first);
        cand_ptrs.emplace_back(all_bevs_[res.first]);
        dist_sq.emplace_back(res.second);
        printf("%7.4f", res.second);
      }
    }
    printf("\n");


    // find which bucket ranges does the query fall in

    // query and accumulate results

  }

  // TODO: 2. add a scan, and retrieval data to buffer
  void addScan(const std::shared_ptr<ContourManager> &added, double curr_timestamp) {
    for (int ll = 0; ll < q_levels_.size(); ll++) {
      for (const auto &permu_key: added->getRetrievalKey(q_levels_[ll]))
        if (permu_key.sum() != 0)
          layer_db_[ll].pushBuffer(permu_key, curr_timestamp, all_bevs_.size());
    }
    all_bevs_.emplace_back(added);
  }

  // TODO: 3. push data popped from buffer, and maintain balance (at most 2 buckets at a time)
  void pushAndBalance(int seed, double curr_timestamp) {
    int idx_t1 = std::abs(seed) % (2 * (layer_db_[0].max_num_backets_ - 2));
    if (idx_t1 > (layer_db_[0].max_num_backets_ - 2))
      idx_t1 = 2 * (layer_db_[0].max_num_backets_ - 2) - idx_t1;

    printf("Balancing bucket %d and %d\n", idx_t1, idx_t1 + 1);

    printf("Tree size of each bucket: \n");
    for (int ll = 0; ll < q_levels_.size(); ll++) {
      printf("q_levels_[%d]: ", ll);
      layer_db_[ll].rebuild(idx_t1, curr_timestamp);
      for (int i = 0; i < layer_db_[ll].max_num_backets_; i++) {
        printf("%5lu", layer_db_[ll].buckets_[i].getTreeSize());
      }
      printf("\n");
    }
  }

};


#endif //CONT2_CONTOUR_DB_H
