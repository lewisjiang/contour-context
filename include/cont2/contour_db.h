//
// Created by lewis on 5/6/22.
//

#ifndef CONT2_CONTOUR_DB_H
#define CONT2_CONTOUR_DB_H

#include "cont2/contour_mng.h"
#include "cont2/correlation.h"

#include <memory>
#include <algorithm>
#include <set>
#include <unordered_set>

#include <nanoflann.hpp>
#include <utility>
#include "KDTreeVectorOfVectorsAdaptor.h"

//typedef Eigen::Matrix<KeyFloatType, 4, 1> tree_key_t;
typedef std::vector<RetrievalKey> my_vector_of_vectors_t;
typedef KDTreeVectorOfVectorsAdaptor<my_vector_of_vectors_t, KeyFloatType> my_kd_tree_t;

const KeyFloatType MAX_BACKET_VAL = 1000.0f;
const KeyFloatType MAX_DIST_SQ = 1e6;

template<
    typename _DistanceType, typename _IndexType = size_t,
    typename _CountType = size_t>
class MyKNNResSet : public nanoflann::KNNResultSet<_DistanceType, _IndexType, _CountType> {
public:
  using DistanceType = _DistanceType;
  using IndexType = _IndexType;
  using CountType = _CountType;

  inline explicit MyKNNResSet(CountType capacity_)
      : nanoflann::KNNResultSet<_DistanceType, _IndexType, _CountType>(capacity_) {
  }

  inline void init(IndexType *indices_, DistanceType *dists_, DistanceType max_dist_metric) {
    this->indices = indices_;
    this->dists = dists_;
    this->count = 0;
    if (this->capacity)
      this->dists[this->capacity - 1] = max_dist_metric;
  }
};

struct TreeBucketConfig {
  double max_elapse_ = 20.0;  // the max spatial/temporal delay before adding to the trees
  double min_elapse_ = 10.0;  // the min spatial/temporal delay to wait before adding to the trees
};

struct IndexOfKey {  // where does the key come from? 1) global idx, 2) level, 3) ith/seq at that level
  size_t gidx{};
  int level{};
  int seq{};

  IndexOfKey(size_t g, int l, int s) : gidx(g), level(l), seq(s) {}
};

struct TreeBucket {


  struct RetrTriplet {  // retrieval triplet
    RetrievalKey pt;
    double ts{};
    IndexOfKey iok;

//    RetrTriplet() = default;

    RetrTriplet(const RetrievalKey &_a, double _b, size_t g, int l, int s) : pt(_a), ts(_b), iok(g, l, s) {}

    RetrTriplet(const RetrievalKey &_a, double _b, IndexOfKey i) : pt(_a), ts(_b), iok(i) {}
  };

  const TreeBucketConfig cfg_;

  KeyFloatType buc_beg_{}, buc_end_{};  // [beg, end)
  my_vector_of_vectors_t data_tree_;
  std::shared_ptr<my_kd_tree_t> tree_ptr = nullptr;
  std::vector<RetrTriplet> buffer_;    // ordered, ascending
  std::vector<IndexOfKey> gkidx_tree_;  // global index of ContourManager in the whole database

  TreeBucket(const TreeBucketConfig &config, KeyFloatType beg, KeyFloatType end) : cfg_(config), buc_beg_(beg),
                                                                                   buc_end_(end) {}

  size_t getTreeSize() const {
    DCHECK_EQ(data_tree_.size(), gkidx_tree_.size());
    return data_tree_.size();
  }

  void pushBuffer(const RetrievalKey &tree_key, double ts, IndexOfKey iok) {
    buffer_.emplace_back(tree_key, ts, iok);
  }

  inline bool needPopBuffer(double curr_ts) const {
    double ts_overflow = curr_ts - cfg_.max_elapse_;
    if (buffer_.empty() || buffer_[0].ts > ts_overflow)  // rebuild every (max-min) sec, ignore newest min.
      return false;
    return true;
  }

  inline void rebuildTree() {
    if (tree_ptr)
      // is this an efficient rebuild when number of elements change?
      // should be OK, since the data array is stored in flann as const alias, will not be affected by reassigning.
      tree_ptr->index->buildIndex();
    else
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
      DCHECK_EQ(sz0, gkidx_tree_.size());
      data_tree_.reserve(sz0 + gap);
      gkidx_tree_.reserve(sz0 + gap);
      for (size_t i = 0; i < gap; i++) {
        data_tree_.emplace_back(buffer_[i].pt);
        gkidx_tree_.emplace_back(buffer_[i].iok);
      }
      buffer_.erase(buffer_.begin(), buffer_.begin() + gap);

      rebuildTree();
    }
  }

  ///
  /// \param num_res
  /// \param ret_idx
  /// \param out_dist_sq The size must be num_res
  /// \param q_key
  void
  knnSearch(const int num_res, std::vector<IndexOfKey> &ret_idx, std::vector<KeyFloatType> &out_dist_sq,
            RetrievalKey q_key, const KeyFloatType max_dist_sq) const {
    ret_idx.clear();
    out_dist_sq.resize(num_res);
    std::fill(out_dist_sq.begin(), out_dist_sq.end(), MAX_DIST_SQ);   // in case of fewer points in the tree than k

    if (!tree_ptr) { return; }

    ret_idx.reserve(num_res);

    std::vector<size_t> idx(num_res, 0);

//    nanoflann::KNNResultSet<KeyFloatType> resultSet(num_res);  // official knn search
//    resultSet.init(&idx[0], &out_dist_sq[0]);

    MyKNNResSet<KeyFloatType> resultSet(num_res);   // knn with max dist
    resultSet.init(&idx[0], &out_dist_sq[0], max_dist_sq);

    tree_ptr->index->findNeighbors(resultSet, q_key.data(), nanoflann::SearchParams(10));
    for (size_t i = 0; i < num_res; i++) {
      ret_idx.emplace_back(gkidx_tree_[idx[i]]);
    }
  }

  void rangeSearch(KeyFloatType worst_dist_sq, std::vector<IndexOfKey> &ret_idx, std::vector<KeyFloatType> &out_dist_sq,
                   RetrievalKey q_key) const {
    std::vector<std::pair<size_t, KeyFloatType>> ret_matches;
    nanoflann::SearchParams params;
    // params.sorted = false;

    ret_idx.clear();
    out_dist_sq.clear();
    if (!tree_ptr) { return; }

    const size_t nMatches = tree_ptr->index->radiusSearch(q_key.data(), worst_dist_sq, ret_matches, params);

    ret_idx.reserve(nMatches);
    out_dist_sq.reserve(nMatches);

    for (size_t i = 0; i < ret_matches.size(); i++) {
      ret_idx.emplace_back(gkidx_tree_[ret_matches[i].first]);
      out_dist_sq.emplace_back(ret_matches[i].second);
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
  void pushBuffer(const RetrievalKey &layer_key, double ts, IndexOfKey scan_key_gidx) {
    for (int i = 0; i < max_num_backets_; i++) {
      if (bucket_ranges_[i] <= layer_key(bucket_chann_) && layer_key(bucket_chann_) < bucket_ranges_[i + 1]) {
        if (layer_key.sum() != 0) // if an all zero key, we do not add it.
          buckets_[i].pushBuffer(layer_key, ts, scan_key_gidx);
        return;
      }
    }
  }

  // TO-DO: rebalance and add buffer to the tree
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
        tr2.gkidx_tree_.emplace_back(tr1.gkidx_tree_[sort_permu[sz1 - i - 1]]);
      }

      int p_dat = sz1 - 1, p_perm = sz1 - 1;
      for (; p_perm >= sz1 - num_to_move; p_perm--) {  // rotate to the back;
        while (tr1.data_tree_[p_dat](bucket_chann_) >= split_val)
          p_dat--;
        if (sort_permu[p_perm] < p_dat) {
          std::swap(tr1.data_tree_[p_dat], tr1.data_tree_[sort_permu[p_perm]]);
          std::swap(tr1.gkidx_tree_[p_dat], tr1.gkidx_tree_[sort_permu[p_perm]]);
          p_dat--;
        }
      }
      DCHECK_EQ(p_dat + num_to_move, sz1 - 1);
      tr1.data_tree_.resize(p_dat + 1);
      tr1.gkidx_tree_.resize(p_dat + 1, tr1.gkidx_tree_[0]);

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
      DCHECK_LE(sz_rem, tr1.buffer_.size());
      tr1.buffer_.resize(sz_rem, tr1.buffer_[0]);  // ugly
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
        tr1.gkidx_tree_.emplace_back(tr2.gkidx_tree_[sort_permu[sz2 - i - 1]]);
      }

      int p_dat = sz2 - 1, p_perm = sz2 - 1;
      for (; p_perm >= sz2 - num_to_move; p_perm--) {  // rotate to the back;
        while (tr2.data_tree_[p_dat](bucket_chann_) < split_val)
          p_dat--;
        if (sort_permu[p_perm] < p_dat) {
          std::swap(tr2.data_tree_[p_dat], tr2.data_tree_[sort_permu[p_perm]]);
          std::swap(tr2.gkidx_tree_[p_dat], tr2.gkidx_tree_[sort_permu[p_perm]]);
          p_dat--;
        }
      }
      DCHECK_EQ(p_dat + num_to_move, sz2 - 1);
      tr2.data_tree_.resize(p_dat + 1);
      tr2.gkidx_tree_.resize(p_dat + 1, tr2.gkidx_tree_[0]);

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
      DCHECK_LE(sz_rem, tr2.buffer_.size());
      tr2.buffer_.resize(sz_rem, tr2.buffer_[0]);
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

  // TO-DO: query
  void layerKNNSearch(const RetrievalKey &q_key, const int k_top, const KeyFloatType max_dist_sq,
                      std::vector<std::pair<IndexOfKey, KeyFloatType>> &res_pairs) const {
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
      std::vector<IndexOfKey> tmp_gidx;
      std::vector<KeyFloatType> tmp_dists_sq;

      if (i == 0) {
        buckets_[mid_bucket].knnSearch(k_top, tmp_gidx, tmp_dists_sq, q_key, max_dist_sq_run);
        for (int j = 0; j < k_top; j++)
          if (tmp_dists_sq[j] < max_dist_sq_run)
            res_pairs.emplace_back(tmp_gidx[j], tmp_dists_sq[j]);
          else
            break;

      } else if (mid_bucket - i >= 0) {
        if ((q_key(bucket_chann_) - bucket_ranges_[mid_bucket - i + 1]) *
            (q_key(bucket_chann_) - bucket_ranges_[mid_bucket - i + 1]) > max_dist_sq_run) { continue; }
        buckets_[mid_bucket - i].knnSearch(k_top, tmp_gidx, tmp_dists_sq, q_key, max_dist_sq_run);
        for (int j = 0; j < k_top; j++)
          if (tmp_dists_sq[j] < max_dist_sq_run)
            res_pairs.emplace_back(tmp_gidx[j], tmp_dists_sq[j]);
          else
            break;

      } else if (mid_bucket + i < max_num_backets_) {  // query a
        if ((q_key(bucket_chann_) - bucket_ranges_[mid_bucket + i]) *
            (q_key(bucket_chann_) - bucket_ranges_[mid_bucket + i]) > max_dist_sq_run) { continue; }
        buckets_[mid_bucket + i].knnSearch(k_top, tmp_gidx, tmp_dists_sq, q_key, max_dist_sq_run);
        for (int j = 0; j < k_top; j++)
          if (tmp_dists_sq[j] < max_dist_sq_run)
            res_pairs.emplace_back(tmp_gidx[j], tmp_dists_sq[j]);
          else
            break;

      }
      std::sort(res_pairs.begin(), res_pairs.end(),
                [&](const std::pair<IndexOfKey, KeyFloatType> &a, const std::pair<IndexOfKey, KeyFloatType> &b) {
                  return a.second < b.second;
                });
      if (res_pairs.size() >= k_top) {
        res_pairs.resize(k_top, res_pairs[0]);
        max_dist_sq_run = res_pairs.back().second;
      }
    }
  }

  // TODO:
  void layerRangeSearch(const RetrievalKey &q_key, const KeyFloatType max_dist_sq,
                        std::vector<std::pair<IndexOfKey, KeyFloatType>> &res_pairs) const {
    res_pairs.clear();
    for (int i = 0; i < max_num_backets_; i++) {
      std::vector<IndexOfKey> tmp_gkidx;
      std::vector<KeyFloatType> tmp_dists_sq;
      buckets_[i].rangeSearch(max_dist_sq, tmp_gkidx, tmp_dists_sq, q_key);
      for (int j = 0; j < tmp_gkidx.size(); j++) {
        res_pairs.emplace_back(tmp_gkidx[j], tmp_dists_sq[j]);
      }
    }
  }

};

struct CandSimScore {
  // Our similarity score is multi-dimensional
  int cnt_constell_ = 0;
  int cnt_pairwise_sim_ = 0;
  double correlation_ = 0;

  CandSimScore() = default;

  CandSimScore(int cnt_chk1, int cnt_chk2, double init_corr) : cnt_constell_(cnt_chk1), cnt_pairwise_sim_(cnt_chk2),
                                                               correlation_(init_corr) {}

  bool operator<(const CandSimScore &a) const {
    if (cnt_pairwise_sim_ == a.cnt_pairwise_sim_)
      return correlation_ < a.correlation_;
    return cnt_pairwise_sim_ < a.cnt_pairwise_sim_;
  }

  bool operator>(const CandSimScore &a) const {
    if (cnt_pairwise_sim_ == a.cnt_pairwise_sim_)
      return correlation_ > a.correlation_;
    return cnt_pairwise_sim_ > a.cnt_pairwise_sim_;
  }
};


// definition:
// 1. given a target/new cm and a multi-dimension similarity threshold, we first feed in, consecutively, all the
// src/history/old cms that are considered as its loop closure candidates, each with a matching hint in the form of a
// certain retrieval key (since a cm has many keys) and the key's BCI.
// 2. When feeding in a given candidate, we progressively calculate and check the similarity score and discard it once
// any dimension falls below threshold. For a candidate with multiple hints, we choose the "most similar" one before the
// calculation of any continuous correlation score???.
// 3. After getting the best of each candidate for all the candidates, we sort them according to ??? and select the top
// several to further optimize and calculate the correlation score (If all the candidates are required, we optimize them
// all). All the remaining candidates are predicted as positive, and the user can request "top-1" for protocol 1 and
// "all" for protocol 2.
struct CandidateManager {
  struct CandidateData {
    std::shared_ptr<const ContourManager> cm_cand_;
    std::unique_ptr<ConstellCorrelation> corr_est_;
    Eigen::Isometry2d T_delta_;  // transform candidate into tgt/curr/query/new frame
    CandSimScore sim_score_;
  };

  CandSimScore score_lb_;  // score lower bound
  std::shared_ptr<const ContourManager> cm_tgt_;

  // data structure
  std::map<int, int> cand_id_pos_pair_;
  std::vector<CandidateData> candidates_;

  // bookkeeping:
  bool adding_finished = false;
  int num_aft_check1 = 0;
  int num_aft_check2 = 0;
  int num_aft_check3 = 0;
  int num_aft_check4 = 0;

  CandidateManager(std::shared_ptr<const ContourManager> cm_q,
                   const CandSimScore &score_lb) : cm_tgt_(std::move(cm_q)), score_lb_(score_lb) {}

  // "hint": the key pairing and the bci pairing
  bool checkCandWithHint(const std::shared_ptr<const ContourManager> &cm_cand, const ConstellationPair &anchor_pair) {
    CHECK(!adding_finished);
    int cand_id = cm_cand->getIntID();
    CandSimScore curr_score;

    // check: (1/3) anchor similarity
    bool anchor_sim = ContourManager::checkContPairSim(*cm_cand, *cm_tgt_, anchor_pair);
    if (!anchor_sim)
      return false;
    num_aft_check1++;

    // check (2/3): pure constellation check
    std::vector<ConstellationPair> tmp_pairs;
    int num_constell_pairs = BCI::checkConstellSim(cm_cand->getBCI(anchor_pair.level, anchor_pair.seq_src),
                                                   cm_tgt_->getBCI(anchor_pair.level, anchor_pair.seq_tgt), tmp_pairs);
    if (num_constell_pairs < score_lb_.cnt_constell_)
      return false;
    curr_score.cnt_constell_ = num_constell_pairs;
    num_aft_check2++;

    // check (3/3): individual similarity check
    std::vector<int> tmp_sim_idx;  // the index of contours in tmp_pairs after individual check
    std::pair<Eigen::Isometry2d, int> mat_res;
    mat_res = ContourManager::calcScanCorresp(*cm_cand, *cm_tgt_, tmp_pairs, tmp_sim_idx,
                                              score_lb_.cnt_pairwise_sim_);  // 5: minimal remaining pairs required
    if (mat_res.second < score_lb_.cnt_pairwise_sim_)
      return false;
    curr_score.cnt_pairwise_sim_ = mat_res.second;
    num_aft_check3++;

    // correlation calculation
    auto cand_it = cand_id_pos_pair_.find(cand_id);
    if (cand_it != cand_id_pos_pair_.end() &&
        curr_score < candidates_[cand_it->second].sim_score_)  // if a better candidate rep exists
      return false;

    GMMOptConfig gmm_config;
    std::unique_ptr<ConstellCorrelation> corr_est(new ConstellCorrelation(gmm_config));
    double corr_score_init = corr_est->initProblem(*cm_cand, *cm_tgt_, mat_res.first);
    if (corr_score_init < score_lb_.correlation_)
      return false;
    num_aft_check4++;

    curr_score.correlation_ = corr_score_init;

    // add to database
    if (cand_it != cand_id_pos_pair_.end()) {
      auto &existing_cand = candidates_[cand_it->second];
      if (curr_score < existing_cand.sim_score_)
        return false;
      else {
        existing_cand.sim_score_ = curr_score;
        existing_cand.T_delta_ = mat_res.first;
        existing_cand.cm_cand_ = cm_cand;
        existing_cand.corr_est_ = std::move(corr_est);
      }
    } else {
      CandidateData new_cand;
      new_cand.sim_score_ = curr_score;
      new_cand.T_delta_ = mat_res.first;
      new_cand.cm_cand_ = cm_cand;
      new_cand.corr_est_ = std::move(corr_est);
      cand_id_pos_pair_.insert({cand_id, candidates_.size()});
      candidates_.emplace_back(std::move(new_cand));
    }

    return true;

    // TODO: check results of each level when fine tuning for recall:
//    if (num_aft_check1) {
//      printf("L:%d S:%d. After check 1: %d\n", q_levels_[ll], seq, num_aft_check1);
//      printf("L:%d S:%d. After check 2: %d\n", q_levels_[ll], seq, num_aft_check2);
//      printf("L:%d S:%d. After check 3: %d\n", q_levels_[ll], seq, num_aft_check3);
//    }

  }

  int
  fineOptimize(int top_n, std::vector<std::shared_ptr<const ContourManager>> &res_cand, std::vector<double> &res_corr,
               std::vector<Eigen::Isometry2d> &res_T) {
    adding_finished = true;

    res_cand.clear();
    res_corr.clear();
    res_T.clear();

    if (candidates_.empty())
      return 0;

    std::sort(candidates_.begin(), candidates_.end(), [&](const CandidateData &d1, const CandidateData &d2) {
//      return d1.sim_score_ > d2.sim_score_;
      return d1.sim_score_.correlation_ > d2.sim_score_.correlation_;
    });

    int ret_size = std::min(top_n, (int) candidates_.size());
    for (int i = 0; i < ret_size; i++) {
      auto tmp_res = candidates_[i].corr_est_->calcCorrelation();
      candidates_[i].sim_score_.correlation_ = tmp_res.first;
      candidates_[i].T_delta_ = tmp_res.second;
    }

    std::sort(candidates_.begin(), candidates_.begin() + ret_size,
              [&](const CandidateData &d1, const CandidateData &d2) {
//                return d1.sim_score_ > d2.sim_score_;
                return d1.sim_score_.correlation_ > d2.sim_score_.correlation_;
              });

    printf("Fine optim corr:\n");
    for (int i = 0; i < ret_size; i++) {
      res_cand.emplace_back(candidates_[i].cm_cand_);
      res_corr.emplace_back(candidates_[i].sim_score_.correlation_);
      res_T.emplace_back(candidates_[i].T_delta_);
      printf("correlation: %f\n", candidates_[i].sim_score_.correlation_);
    }

    return ret_size;
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
  std::vector<std::shared_ptr<const ContourManager>> all_bevs_;

public:
  ContourDB(const ContourDBConfig &config, std::vector<int> q_levels) : cfg_(config), q_levels_(std::move(q_levels)) {
    layer_db_.resize(q_levels_.size());
  }

  // TODO: 1. query database
  void queryKNN(const ContourManager &q_cont,
                std::vector<std::shared_ptr<const ContourManager>> &cand_ptrs,
                std::vector<KeyFloatType> &dist_sq) const;

  // TODO: unlike queryKNN, this one directly calculates relative transform and requires no post processing
  //  outside the function. The returned cmng are the matched ones.
  ///
  /// \param q_cont
  /// \param cand_ptrs
  /// \param cand_corr
  /// \param cand_tf candidates are src/old, T_tgt = T_delta * T_src
  void queryRangedKNN(const std::shared_ptr<const ContourManager> &q_ptr,
                      std::vector<std::shared_ptr<const ContourManager>> &cand_ptrs,
                      std::vector<double> &cand_corr,
                      std::vector<Eigen::Isometry2d> &cand_tf) const {
    cand_ptrs.clear();
    cand_corr.clear();

    double t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0;
    TicToc clk;

    CandSimScore score_lb(10, 5, 0.65);
    CandidateManager cand_mng(q_ptr, score_lb);

    // for each layer
//    std::set<size_t> matched_gidx;
    for (int ll = 0; ll < q_levels_.size(); ll++) {
      std::vector<BCI> q_bcis = q_ptr->getLevBCI(q_levels_[ll]);
      std::vector<RetrievalKey> q_keys = q_ptr->getLevRetrievalKey(q_levels_[ll]);
      DCHECK_EQ(q_bcis.size(), q_keys.size());
      for (int seq = 0; seq < q_bcis.size(); seq++) {
        if (q_keys[seq].sum() != 0) {
          // 1. query
          clk.tic();
          std::vector<std::pair<IndexOfKey, KeyFloatType>> tmp_res;
//          layer_db_[ll].layerRangeSearch(q_keys[seq], 3.0, tmp_res);  // 5.0: squared norm
          // calculate max query distance from key bits:
          KeyFloatType key_bounds[3][2];
          key_bounds[0][0] = q_keys[seq][0] * 0.8;  // sqrt(max_eig*cnt)
          key_bounds[0][1] = q_keys[seq][0] / 0.8;

          key_bounds[1][0] = q_keys[seq][1] * 0.8;  // sqrt(min_eig*cnt)
          key_bounds[1][1] = q_keys[seq][1] / 0.8;

          key_bounds[2][0] = q_keys[seq][2] * 0.8 * 0.75;  // com*cnt
          key_bounds[2][1] = q_keys[seq][2] / (0.8 * 0.75);

          KeyFloatType dist_ub = 1e6;
          dist_ub = std::max((q_keys[seq][0] - key_bounds[0][0]) * (q_keys[seq][0] - key_bounds[0][0]),
                             (q_keys[seq][0] - key_bounds[0][1]) * (q_keys[seq][0] - key_bounds[0][1]))
                    + std::max((q_keys[seq][1] - key_bounds[1][0]) * (q_keys[seq][1] - key_bounds[1][0]),
                               (q_keys[seq][1] - key_bounds[1][1]) * (q_keys[seq][1] - key_bounds[1][1]))
                    + std::max((q_keys[seq][2] - key_bounds[2][0]) * (q_keys[seq][2] - key_bounds[2][0]),
                               (q_keys[seq][2] - key_bounds[2][1]) * (q_keys[seq][2] - key_bounds[2][1]));
          printf("Dist ub: %f\n", dist_ub);

//          layer_db_[ll].layerKNNSearch(q_keys[seq], 100, dist_ub, tmp_res);
          layer_db_[ll].layerKNNSearch(q_keys[seq], 50, dist_ub, tmp_res);
//          layer_db_[ll].layerKNNSearch(q_keys[seq], 200, 2000.0, tmp_res);
          t1 += clk.toc();

          printf("L:%d S:%d. Found in range: %lu\n", q_levels_[ll], seq, tmp_res.size());

          // 2. check
          for (const auto &sear_res: tmp_res) {
            clk.tic();
            bool add_one = cand_mng.checkCandWithHint(all_bevs_[sear_res.first.gidx],
                                                      ConstellationPair(q_levels_[ll], sear_res.first.seq, seq));
            t2 += clk.toc();
          }

        }
      }
    }

    // find the best ones with fine tuning:
    const int top_n = 1;
    std::vector<std::shared_ptr<const ContourManager>> res_cand_ptr;
    std::vector<double> res_corr;
    std::vector<Eigen::Isometry2d> res_T;

    clk.tic();
    int num_best_cands = cand_mng.fineOptimize(top_n, res_cand_ptr, res_corr, res_T);
    t5 += clk.toc();

    if (num_best_cands) {
      printf("After check 1: %d\n", cand_mng.num_aft_check1);
      printf("After check 2: %d\n", cand_mng.num_aft_check2);
      printf("After check 3: %d\n", cand_mng.num_aft_check3);
      printf("After check 4: %d\n", cand_mng.num_aft_check4);
    } else {
      printf("No candidates are valid after checks.\n");
    }

    for (int i = 0; i < num_best_cands; i++) {
      cand_ptrs.emplace_back(res_cand_ptr[i]);
      cand_corr.emplace_back(res_corr[i]);
      cand_tf.emplace_back(res_T[i]);
    }

    printf("T knn search : %7.5f\n", t1);
    printf("T running chk: %7.5f\n", t2);
//    printf("T conste check: %7.5f\n", t3);
//    printf("T calc corresp: %7.5f\n", t4);
    printf("T fine optim : %7.5f\n", t5);

    // TODO: separate pose refinement into another protected function
  }

  // TO-DO: 2. add a scan, and retrieval data to buffer
  void addScan(const std::shared_ptr<ContourManager> &added, double curr_timestamp) {
    for (int ll = 0; ll < q_levels_.size(); ll++) {
      int seq = 0; // key seq in a layer for a given scan.
      for (const auto &permu_key: added->getLevRetrievalKey(q_levels_[ll])) {
        if (permu_key.sum() != 0)
          layer_db_[ll].pushBuffer(permu_key, curr_timestamp, IndexOfKey(all_bevs_.size(), q_levels_[ll], seq));
        seq++;
      }
    }
    all_bevs_.emplace_back(added);
  }

  // TO-DO: 3. push data popped from buffer, and maintain balance (at most 2 buckets at a time)
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
