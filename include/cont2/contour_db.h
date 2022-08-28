//
// Created by lewis on 5/6/22.
//

#ifndef CONT2_CONTOUR_DB_H
#define CONT2_CONTOUR_DB_H

#include "cont2/contour_mng.h"
#include "cont2/correlation.h"
#include "tools/algos.h"

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

//! The minimal unit of a tree/wrapper of a kd-tree
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
  void knnSearch(const int num_res, std::vector<IndexOfKey> &ret_idx, std::vector<KeyFloatType> &out_dist_sq,
                 RetrievalKey q_key, const KeyFloatType max_dist_sq) const;

  void rangeSearch(KeyFloatType worst_dist_sq, std::vector<IndexOfKey> &ret_idx, std::vector<KeyFloatType> &out_dist_sq,
                   RetrievalKey q_key) const;

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
  void rebuild(int idx_t1, double curr_ts);

  // TO-DO: query
  void layerKNNSearch(const RetrievalKey &q_key, const int k_top, const KeyFloatType max_dist_sq,
                      std::vector<std::pair<IndexOfKey, KeyFloatType>> &res_pairs) const;

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

//struct CandSimScore {
//  // Our similarity score is multi-dimensional
//  int cnt_constell_ = 0;
//  int cnt_pairwise_sim_ = 0;
//  double correlation_ = 0;
//
//  CandSimScore() = default;
//
//  CandSimScore(int cnt_chk1, int cnt_chk2, double init_corr) : cnt_constell_(cnt_chk1), cnt_pairwise_sim_(cnt_chk2),
//                                                               correlation_(init_corr) {}
//
//  bool operator<(const CandSimScore &a) const {
//    if (cnt_pairwise_sim_ == a.cnt_pairwise_sim_)
//      return correlation_ < a.correlation_;
//    return cnt_pairwise_sim_ < a.cnt_pairwise_sim_;
//  }
//
//  bool operator>(const CandSimScore &a) const {
//    if (cnt_pairwise_sim_ == a.cnt_pairwise_sim_)
//      return correlation_ > a.correlation_;
//    return cnt_pairwise_sim_ > a.cnt_pairwise_sim_;
//  }
//};

// A set of all the check parameters combined as a whole
struct CandidateScoreEnsemble {
  ScoreConstellSim sim_constell;
  ScorePairwiseSim sim_pair;
  ScorePostProc sim_post;
//  float correlation = 0;
//  float area_perc = 0;
};


// definition:
// 1. given a target/new cm and a multi-dimension similarity threshold, we first feed in, consecutively, all the
// src/history/old cms that are considered as its loop closure candidates, each with a matching hint in the form of a
// certain retrieval key (since a cm has many keys) and the key's BCI.
// 2. When feeding in a given candidate, we progressively calculate and check the similarity score and discard it once
// any dimension falls below threshold. For a candidate with multiple hints, we combine results from them and remove
// outliers before the calculation of any continuous correlation score.
// 3. After getting the best est of each candidate for all the candidates, we sort them according to some partial order
// and select the top several to further optimize and calculate the correlation score (If all the candidates are
// required, we optimize them all). All the remaining candidates are predicted as positive, and the user can request
// "top-1" for protocol 1 and "all" for protocol 2.
struct CandidateManager {
  // It is possible that multiple key/bci matches (pass the checks with different TF) correspond to the same pose, so
  // we record the potential proposals and find the most likely one after checking all the candidates.
  struct CandidateAnchorProp {
    std::map<ConstellationPair, float> constell_;  // map of {constellation matches: percentage score}
    Eigen::Isometry2d T_delta_;  // the key feature that distinguishes different proposals
    float correlation_ = 0;
    int vote_cnt_ = 0;  // cnt of matched contours voting for this TF (not cnt of unique pairs)
    float area_perc_ = 0;  // a weighted sum of the area % of used contours in all the contours at different levels.
    // TODO: should we record area percentage as the metric for "votes"?
  };

  ///
  struct CandidatePoseData {
    std::shared_ptr<const ContourManager> cm_cand_;
    std::unique_ptr<ConstellCorrelation> corr_est_;  // generate the correlation estimator after polling all the cands
    std::vector<CandidateAnchorProp> anch_props_;

    /// add a anchor proposal, either merge or create new in `anch_props_`
    /// \param T_prop
    /// \param sim_pairs
    /// \param sim_area_perc The level percentage score of a corresponding constellation
    void addProposal(const Eigen::Isometry2d &T_prop, const std::vector<ConstellationPair> &sim_pairs,
                     const std::vector<float> &sim_area_perc) {
      DCHECK_GT(sim_pairs.size(), 3);  // hard bottom line
      DCHECK_EQ(sim_pairs.size(), sim_area_perc.size());

      for (int i = 0; i < anch_props_.size(); i++) {
        const Eigen::Isometry2d delta_T = T_prop.inverse() * anch_props_[i].T_delta_;
        // hardcoded threshold: 2.0m, 0.3.rad
        if (delta_T.translation().norm() < 2.0 && std::abs(std::atan2(delta_T(1, 0), delta_T(0, 0))) < 0.3) {
          for (int j = 0; j < sim_pairs.size(); j++) {
            anch_props_[i].constell_.insert({sim_pairs[j], sim_area_perc[j]});  // unique
          }

          anch_props_[i].vote_cnt_ += sim_pairs.size();  // not unique
          // TODO: Do we need the `CandSimScore` object?
          // A: seems no.
//          anch_props_[i].sim_score_.cnt_pairwise_sim_ = std::max(anch_props_[i].sim_score_.cnt_pairwise_sim_,
//                                                                 (int) sim_pairs.size());
          // TODO: Do we need to re-estimate the TF? Or just blend (manipulate with TF param and weights)?
          // current method: naively blend parameters
//            anch_props_[i].T_delta_ = ContourManager::getTFFromConstell();
          int w1 = anch_props_[i].vote_cnt_, w2 = sim_pairs.size();
          Eigen::Vector2d trans_bl =
              (anch_props_[i].T_delta_.translation() * w1 + T_prop.translation() * w2) / (w1 + w2);
          double ang1 = std::atan2(anch_props_[i].T_delta_(1, 0), anch_props_[i].T_delta_(0, 0));
          double ang2 = std::atan2(T_prop(1, 0), T_prop(0, 0));

          double diff = ang2 - ang1;
          if (diff < 0) diff += 2 * M_PI;
          if (diff > M_PI) diff -= 2 * M_PI;
          double ang_bl = diff * w2 / (w1 + w2) + ang1;

          anch_props_[i].T_delta_.setIdentity();
          anch_props_[i].T_delta_.rotate(ang_bl);  // no need to clamp
          anch_props_[i].T_delta_.pretranslate(trans_bl);

          return;  // greedy
        }
      }

      if (anch_props_.size() > 3)
        return; // limit the number of different proposals w.r.t. a pose

      // empty set or no similar proposal
      anch_props_.emplace_back();
      anch_props_.back().T_delta_ = T_prop;
      for (int j = 0; j < sim_pairs.size(); j++) {
        anch_props_.back().constell_.insert({sim_pairs[j], sim_area_perc[j]});  // unique
      }
      anch_props_.back().vote_cnt_ = sim_pairs.size();
//      anch_props_.back().sim_score_.cnt_pairwise_sim_ = sim_pairs.size();

    }
  };

  //=============================================================

//  const CandSimScore score_lb_;  // score lower/upper bound
  std::shared_ptr<const ContourManager> cm_tgt_;

  // dynamic thresholds param and var for different checks. Used to replace `score_lb_`
  const CandidateScoreEnsemble sim_ub_;  // the upper bound of check thres
  CandidateScoreEnsemble sim_var_;   // the (dynamic) lower bound of check thres, increase with positive predictions

//  // check 1: `checkConstellSim`
//  const ScoreConstellSim sim_constell_ub_;  // the upper bound of check thres
//  ScoreConstellSim sim_constell_var_;  // the (dynamic) lower bound of check thres, increase with positive predictions
//  // check 2: `checkPairwiseSim`
//  const ScorePairwiseSim sim_pair_ub_;  // the upper bound of check thres
//  ScorePairwiseSim sim_pair_var_;

  // data structure
  std::map<int, int> cand_id_pos_pair_;
  std::vector<CandidatePoseData> candidates_;

  // bookkeeping:
//  bool adding_finished = false, tidy_finished=false;
  int flow_valve = 0; // avoid to reverse the work flow
  int cand_aft_check1 = 0;  // number of candidate occurrence (not unique places) after each check
  int cand_aft_check2 = 0;
  int cand_aft_check3 = 0;
  int cand_aft_check4 = 0;

  CandidateManager(std::shared_ptr<const ContourManager> cm_q,
                   const CandidateScoreEnsemble sim_lb, const CandidateScoreEnsemble sim_ub) :
      cm_tgt_(std::move(cm_q)), sim_var_(sim_lb), sim_ub_(sim_ub) {
    // TO-DO: pass in sim score lb and ub as params
    CHECK(sim_lb.sim_constell.strictSmaller(sim_ub.sim_constell));
    CHECK(sim_lb.sim_pair.strictSmaller(sim_ub.sim_pair));
    CHECK(sim_lb.sim_post.strictSmaller(sim_ub.sim_post));
  }

  /// Main func 1/3: check possible anchor pairing and add to the database
  /// \param cm_cand contour manager for the candidate
  /// \param anchor_pair "hint": the anchor for key and bci pairing
  /// \return
  CandidateScoreEnsemble checkCandWithHint(const std::shared_ptr<const ContourManager> &cm_cand,
                                           const ConstellationPair &anchor_pair) {
    DCHECK(flow_valve == 0);
    int cand_id = cm_cand->getIntID();
//    CandSimScore curr_score;

    // count the number of passed contour pairs in each check
    // TODO: optimize this cnt_pass return variable
//    std::array<int, 4> cnt_pass = {0, 0, 0, 0};  // 0: anchor sim; 1: constell sim; 2: constell corresp sim; 3:
    // Q: is it the same as `curr_score`?
    CandidateScoreEnsemble ret_score;

    // check: (1/4) anchor similarity
    bool anchor_sim = ContourManager::checkContPairSim(*cm_cand, *cm_tgt_, anchor_pair);
    if (!anchor_sim)
      return ret_score;
    cand_aft_check1++;

    // human readability
    printf("Before check, curr bar:");
    sim_var_.sim_constell.print();
    printf("\t");
    sim_var_.sim_pair.print();
    printf("\n");

    // check (2/4): pure constellation check
    std::vector<ConstellationPair> tmp_pairs1;
    ScoreConstellSim ret_constell_sim = BCI::checkConstellSim(cm_cand->getBCI(anchor_pair.level, anchor_pair.seq_src),
                                                              cm_tgt_->getBCI(anchor_pair.level, anchor_pair.seq_tgt),
                                                              sim_var_.sim_constell, tmp_pairs1);
    ret_score.sim_constell = ret_constell_sim;
    if (ret_constell_sim.overall() < sim_var_.sim_constell.overall())
      return ret_score;
//    curr_score.cnt_constell_ = ret_constell_sim;
    cand_aft_check2++;

    // check (3/4): individual similarity check
    std::vector<ConstellationPair> tmp_pairs2;
    std::vector<float> tmp_area_perc;
    ScorePairwiseSim ret_pairwise_sim = ContourManager::checkConstellCorrespSim(*cm_cand, *cm_tgt_, tmp_pairs1,
                                                                                sim_var_.sim_pair, tmp_pairs2,
                                                                                tmp_area_perc);
    ret_score.sim_pair = ret_pairwise_sim;
    if (ret_pairwise_sim.overall() < sim_var_.sim_pair.overall())
      return ret_score;
//    curr_score.cnt_pairwise_sim_ = ret_pairwise_sim;
    cand_aft_check3++;


    // 2. Get the transform between the two scans
    Eigen::Isometry2d T_pass = ContourManager::getTFFromConstell(*cm_cand, *cm_tgt_, tmp_pairs2.begin(),
                                                                 tmp_pairs2.end());


//    // additional check (4/4) self censor (need transform T) NOTE: switch on to use
//    double est_trans_norm2d = ConstellCorrelation::getEstSensTF(T_pass, cm_tgt_->getConfig()).translation().norm();
//    if (est_trans_norm2d > 4.0) {
//      printf("Long dist censored: %6f > %6f\n", est_trans_norm2d, 4.0);
//      return ret_score;
//    }

    // Now we assume the pair has passed all the tests. We will add the results to the candidate data structure
    // 2. Update the dynamic score thresholds for different
    const int cnt_curr_valid = ret_pairwise_sim.cnt();  // the count of pairs for this positive match
    // 2.1 constell sim
    auto new_const_lb = sim_var_.sim_constell;
    new_const_lb.i_ovlp_sum = cnt_curr_valid;
    new_const_lb.i_ovlp_max_one = cnt_curr_valid;
    new_const_lb.i_in_ang_rng = cnt_curr_valid;
    alignLB<ScoreConstellSim>(new_const_lb, sim_var_.sim_constell);
    alignUB<ScoreConstellSim>(sim_ub_.sim_constell, sim_var_.sim_constell);

    // 2.2 pairwise sim
    auto new_pair_lb = sim_var_.sim_pair;
    new_pair_lb.i_indiv_sim = cnt_curr_valid;
    new_pair_lb.i_orie_sim = cnt_curr_valid;
//    new_pair_lb.f_area_perc = ret_pairwise_sim.f_area_perc;
    alignLB<ScorePairwiseSim>(new_pair_lb, sim_var_.sim_pair);
    alignUB<ScorePairwiseSim>(sim_ub_.sim_pair, sim_var_.sim_pair);

    // 2.3 human readability
    printf("Cand passed. New dynamic bar:");
    sim_var_.sim_constell.print();
    printf("\t");
    sim_var_.sim_pair.print();
    printf("\n");


    // 3. add to/update candidates_
    auto cand_it = cand_id_pos_pair_.find(cand_id);
    if (cand_it != cand_id_pos_pair_.end()) {
      // the candidate pose exists
      candidates_[cand_it->second].addProposal(T_pass, tmp_pairs2, tmp_area_perc);
    } else {
      // add new
      CandidatePoseData new_cand;
      new_cand.cm_cand_ = cm_cand;
      new_cand.addProposal(T_pass, tmp_pairs2, tmp_area_perc);
      cand_id_pos_pair_.insert({cand_id, candidates_.size()});
      candidates_.emplace_back(std::move(new_cand));
    }

    // correlation calculation
    // TODO: merge results for the same candidate pose

    return ret_score;
  }

  // here "anchor" is no longer meaningful, since we've established constellations beyond any single anchor BCI can
  // offer
  // pre-calculate the correlation scores for each candidate set, and check the correlation scores.
  /// Main func 2/3:
  void tidyUpCandidates() {
    DCHECK(flow_valve < 1);
    flow_valve++;
    GMMOptConfig gmm_config;
    printf("Tidy up pose %lu candidates.\n", candidates_.size());

    int cnt_to_rm = 0;

    // analyze the anchor pairs for each pose
    for (auto &candidate: candidates_) {
      DCHECK(!candidate.anch_props_.empty());
      // find the best T_init for setting up correlation problem estimation (based on vote)
      int idx_sel = 0;
      for (int i = 0; i < candidate.anch_props_.size(); i++) {  // TODO: should we use vote count or area?

        // get the percentage of points used in match v. total area of each level.
        std::vector<float> lev_perc(cm_tgt_->getConfig().lv_grads_.size(), 0);
        for (const auto &pr: candidate.anch_props_[i].constell_) {
          lev_perc[pr.first.level] += pr.second;
//          level_perc_used_src[pr.level] += cm_src->getAreaPerc(pr.level, pr.seq_src);
//          level_perc_used_tgt[pr.level] += cm_tgt_->getAreaPerc(pr.level, pr.seq_tgt);
        }

        float perc = 0;
        for (int j = 0; j < NUM_BIN_KEY_LAYER; j++)
//          perc += LAYER_AREA_WEIGHTS[j] * (0 + 2 * level_perc_used_tgt[DIST_BIN_LAYERS[j]]) / 2;
          perc += LAYER_AREA_WEIGHTS[j] * lev_perc[DIST_BIN_LAYERS[j]];

        candidate.anch_props_[i].area_perc_ = perc;

        if (candidate.anch_props_[i].vote_cnt_ > candidate.anch_props_[idx_sel].vote_cnt_)
          idx_sel = i;
      }
      // put the best prop to the first and leave the rest as they are
      std::swap(candidate.anch_props_[0], candidate.anch_props_[idx_sel]);

      // Overall test 1: area percentage score
      printf("Cand id:%d, area perc: %f @max# %d votes\n", candidate.cm_cand_->getIntID(),
             candidate.anch_props_[0].area_perc_, candidate.anch_props_[0].vote_cnt_);
      if (candidate.anch_props_[0].area_perc_ < sim_var_.sim_post.area_perc) { // check (1/3): area score.
        printf("Low area skipped: %6f < %6f\n", candidate.anch_props_[0].area_perc_, sim_var_.sim_post.area_perc);
        cnt_to_rm++;
        continue;
      }

      // Overall test 2: Censor distance. NOTE: The negate!! Larger is better
      double neg_est_trans_norm2d = -ConstellCorrelation::getEstSensTF(candidate.anch_props_[0].T_delta_,
                                                                       cm_tgt_->getConfig()).translation().norm();
      if (neg_est_trans_norm2d < sim_var_.sim_post.neg_est_dist) { // check (2/3): area score.
        printf("Low dist skipped: %6f < %6f\n", neg_est_trans_norm2d, sim_var_.sim_post.neg_est_dist);
        cnt_to_rm++;
        continue;
      }


      // Overall test 3: correlation score
      // set up the correlation optimization problem for each candidate pose
      std::unique_ptr<ConstellCorrelation> corr_est(new ConstellCorrelation(gmm_config));
      auto corr_score_init = (float) corr_est->initProblem(*(candidate.cm_cand_), *cm_tgt_,
                                                           candidate.anch_props_[0].T_delta_);

      printf("Cand id:%d, init corr: %f @max# %d votes\n", candidate.cm_cand_->getIntID(), corr_score_init,
             candidate.anch_props_[0].vote_cnt_);

      // TODO: find the best T_best for optimization init guess (based on problem->Evaluate())
      // Is it necessary?

      if (corr_score_init < sim_var_.sim_post.correlation) { // check (3/3): correlation score.
        printf("Low corr skipped: %6f < %6f\n", corr_score_init, sim_var_.sim_post.correlation);
        cnt_to_rm++;
        continue;
      }

      // passes the test, update the thres variable, and update data structure info
      auto new_post_lb = sim_var_.sim_post;
      new_post_lb.correlation = corr_score_init;
      new_post_lb.area_perc = candidate.anch_props_[0].area_perc_;
      new_post_lb.neg_est_dist = neg_est_trans_norm2d;
      alignLB<ScorePostProc>(new_post_lb, sim_var_.sim_post);
      alignUB<ScorePostProc>(sim_ub_.sim_post, sim_var_.sim_post);

//      sim_var_.area_perc = candidate.anch_props_[0].area_perc_;  // right must ge left
//      sim_var_.area_perc = sim_var_.area_perc > sim_ub_.area_perc ? sim_ub_.area_perc : sim_var_.area_perc;
//
//      sim_var_.correlation = corr_score_init;
//      sim_var_.correlation = sim_var_.correlation > sim_ub_.correlation ? sim_ub_.correlation : sim_var_.correlation;

//      candidate.anch_props_[0].sim_score_.correlation_ = corr_score_init;
      candidate.anch_props_[0].correlation_ = corr_score_init;
      candidate.corr_est_ = std::move(corr_est);
    }

    // remove poses failing the corr check
    int p1 = 0, p2 = candidates_.size() - 1;
    while (p1 <= p2) {
      if (!candidates_[p1].corr_est_ && candidates_[p2].corr_est_) {
        std::swap(candidates_[p1], candidates_[p2]);
        p1++;
        p2--;
      } else {
        if (candidates_[p1].corr_est_) p1++;
        if (!candidates_[p2].corr_est_) p2--;
      }
    }
    CHECK_EQ(p2 + 1 + cnt_to_rm, candidates_.size());
    candidates_.erase(candidates_.begin() + p2 + 1, candidates_.end());

    printf("Tidy up pose remaining: %lu.\n", candidates_.size());

  }

  /// Main func 3/3: pre select hopeful pose candidates, and optimize for finer pose estimations.
  /// \param max_fine_opt
  /// \param res_cand
  /// \param res_corr
  /// \param res_T
  /// \return
  int fineOptimize(int max_fine_opt, std::vector<std::shared_ptr<const ContourManager>> &res_cand,
                   std::vector<double> &res_corr, std::vector<Eigen::Isometry2d> &res_T) {
    DCHECK(flow_valve < 2);
    flow_valve++;

    res_cand.clear();
    res_corr.clear();
    res_T.clear();

    if (candidates_.empty())
      return 0;

    std::sort(candidates_.begin(), candidates_.end(), [&](const CandidatePoseData &d1, const CandidatePoseData &d2) {
      return d1.anch_props_[0].vote_cnt_ > d2.anch_props_[0].vote_cnt_;  // anch_props_ is guaranteed to be non-empty
//      return d1.sim_score_ > d2.sim_score_;
//      return d1.sim_score_.correlation_ > d2.sim_score_.correlation_;
    });

    int pre_sel_size = std::min(max_fine_opt, (int) candidates_.size());
    for (int i = 0; i < pre_sel_size; i++) {
      auto tmp_res = candidates_[i].corr_est_->calcCorrelation();  // fine optimize
//      candidates_[i].anch_props_[0].sim_score_.correlation_ = tmp_res.first;
      candidates_[i].anch_props_[0].correlation_ = tmp_res.first;
      candidates_[i].anch_props_[0].T_delta_ = tmp_res.second;
    }

    std::sort(candidates_.begin(), candidates_.begin() + pre_sel_size,
              [&](const CandidatePoseData &d1, const CandidatePoseData &d2) {
//                return d1.sim_score_ > d2.sim_score_;
//                return d1.sim_score_.correlation_ > d2.sim_score_.correlation_;
//            x    return d1.anch_props_[0].sim_score_.correlation_ > d2.anch_props_[0].sim_score_.correlation_;
                return d1.anch_props_[0].correlation_ > d2.anch_props_[0].correlation_;
              });

    printf("Fine optim corr:\n");
    int ret_size = 1;  // the needed
    for (int i = 0; i < ret_size; i++) {
      res_cand.emplace_back(candidates_[i].cm_cand_);
      res_corr.emplace_back(candidates_[i].anch_props_[0].correlation_);
      res_T.emplace_back(candidates_[i].anch_props_[0].T_delta_);
      printf("correlation: %f\n", candidates_[i].anch_props_[0].correlation_);
    }

    return ret_size;
  }

  // TODO: We hate censorship but this makes output data look pretty.
  // We remove candidates with a MPE trans norm greater than the TP threshold.
  void selfCensor() {

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
  const std::vector<int> q_levels_;  // the layers to generate anchors (Note the difference between `DIST_BIN_LAYERS`)

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
                      const CandidateScoreEnsemble &thres_lb,
                      const CandidateScoreEnsemble &thres_ub,
                      std::vector<std::shared_ptr<const ContourManager>> &cand_ptrs,
                      std::vector<double> &cand_corr,
                      std::vector<Eigen::Isometry2d> &cand_tf) const {
    cand_ptrs.clear();
    cand_corr.clear();
    cand_tf.clear();

    double t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0;
    TicToc clk;

//    CandSimScore score_lb(10, 5, 0.65);  // TODO: use new thres init
//    CandidateManager cand_mng(q_ptr, score_lb);



//    CandidateManager cand_mng(q_ptr, s_const_lb, s_const_ub, s_pair_lb, s_pair_ub);
    CandidateManager cand_mng(q_ptr, thres_lb, thres_ub);

    // for each layer
//    std::set<size_t> matched_gidx;
    for (int ll = 0; ll < q_levels_.size(); ll++) {
      const std::vector<BCI> &q_bcis = q_ptr->getLevBCI(q_levels_[ll]);
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
            auto cnt_chk_pass = cand_mng.checkCandWithHint(all_bevs_[sear_res.first.gidx],
                                                           ConstellationPair(q_levels_[ll], sear_res.first.seq, seq));
            t2 += clk.toc();
          }

        }
      }
    }

    // find the best ones with fine-tuning:
    const int max_fine_opt = 5;
    std::vector<std::shared_ptr<const ContourManager>> res_cand_ptr;
    std::vector<double> res_corr;
    std::vector<Eigen::Isometry2d> res_T;

    clk.tic();
    cand_mng.tidyUpCandidates();
    int num_best_cands = cand_mng.fineOptimize(max_fine_opt, res_cand_ptr, res_corr, res_T);
    t5 += clk.toc();

    if (num_best_cands) {
      printf("After check 1: %d\n", cand_mng.cand_aft_check1);
      printf("After check 2: %d\n", cand_mng.cand_aft_check2);
      printf("After check 3: %d\n", cand_mng.cand_aft_check3);
      printf("After check 4: %d\n", cand_mng.cand_aft_check4);
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
