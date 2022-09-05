//
// Created by lewis on 8/27/22.
//

#include <utility>

#include "cont2/contour_db.h"
#include "eval/evaluator.h"
#include "cont2_ros/spinner_ros.h"

const std::string PROJ_DIR = std::string(PJSRCDIR);

class BatchBinSpinner : public BaseROSSpinner {
  // --- Added members for evaluation and LC module running ---
  ContourDB contour_db;

  ContLCDEvaluator evaluator;

  CandidateScoreEnsemble thres_lb_, thres_ub_;  // check thresholds

  // bookkeeping
  int cnt_tp = 0, cnt_fn = 0, cnt_fp = 0;
  double ts_beg = -1;

public:
  explicit BatchBinSpinner(ros::NodeHandle &nh_, const ContourDBConfig &db_config, std::vector<int> q_levels,
                           const std::string &fpath_pose,
                           const std::string &fpath_laser) : BaseROSSpinner(nh_),
                                                             contour_db(db_config, std::move(q_levels)),
                                                             evaluator(fpath_pose, fpath_laser) {

  }

  // before start: 1/1: load thres
  void loadThres(const std::string &thres_fpath) {
    ContLCDEvaluator::loadCheckThres(thres_fpath, thres_lb_, thres_ub_);
  }

  ///
  /// \param outer_cnt
  /// \return 0: normal. <0: external signal. 1: load failed
  int spinOnce(int &outer_cnt) {
    mtx_status.lock();
    if (stat_terminated) {
      printf("Spin terminated by external signal.\n");
      mtx_status.unlock();
      return -1;
    }
    if (stat_paused) {
      printf("Spin paused by external signal.\n");
      mtx_status.unlock();
      return -2;
    }
    mtx_status.unlock();

    bool loaded = evaluator.loadNewScan();
    if (!loaded) {
      printf("Load new scan failed.\n");
      return 1;
    }
    TicToc clk;
    ros::Time wall_time_ros = ros::Time::now();
    outer_cnt++;

    // 1. Init current scan
    ContourManagerConfig cm_config;
//    cm_config.lv_grads_ = {1.5, 2, 2.5, 3, 3.5, 4};  // KITTI
//    cm_config.lv_grads_ = {1.5, 2.5, 3.5, 4.5, 5.5, 6.5};  // mulran test 1
//    cm_config.lv_grads_ = {1.0, 2.5, 4.0, 5.5, 7.0, 8.5};  // mulran
    cm_config.lv_grads_ = {1.5, 2.0, 3.0, 4.5, 6.0, 7.0};  // mulran

    std::shared_ptr<ContourManager> ptr_cm_tgt = evaluator.getCurrContourManager(cm_config);
    const auto laser_info_tgt = evaluator.getCurrScanInfo();
    printf("\n===\nLoaded: assigned seq: %d, bin path: %s\n", laser_info_tgt.seq, laser_info_tgt.fpath.c_str());

    // 1.1 Prepare and display info: gt/shifted pose, tf
    double ts_curr = laser_info_tgt.ts;
    if (ts_beg < 0) ts_beg = ts_curr;

    Eigen::Isometry3d T_gt_curr = laser_info_tgt.sens_pose;
    Eigen::Vector3d time_translate(0, 0, 1);
    time_translate = time_translate * (ts_curr - ts_beg) / 60;  // 1m per min
    g_poses.insert(std::make_pair(laser_info_tgt.seq, GlobalPoseInfo(T_gt_curr, time_translate.z())));

    geometry_msgs::TransformStamped tf_gt_curr = tf2::eigenToTransform(T_gt_curr);
    broadcastCurrPose(tf_gt_curr);  // the stamp is now

    tf_gt_curr.header.seq = laser_info_tgt.seq;
    tf_gt_curr.transform.translation.z += time_translate.z();
    publishPath(wall_time_ros, tf_gt_curr);
    publishScanSeqText(wall_time_ros, tf_gt_curr, laser_info_tgt.seq);


    // 1.2. save images of layers
    clk.tic();
    for (int i = 0; i < cm_config.lv_grads_.size(); i++) {
      std::string f_name = PROJ_DIR + "/results/layer_img/contour_" + "lv" + std::to_string(i) + "_" +
                           ptr_cm_tgt->getStrID() + ".png";   // TODO: what should be the str name of scans?
      ptr_cm_tgt->saveContourImage(f_name, i);
    }
    std::cout << "Time save layers: " << clk.toctic() << std::endl;

    ptr_cm_tgt->clearImage();  // a must to save memory

    // 2. query
    std::vector<std::pair<int, int>> new_lc_pairs;
    std::vector<std::shared_ptr<const ContourManager>> ptr_cands;
    std::vector<double> cand_corr;
    std::vector<Eigen::Isometry2d> bev_tfs;

    clk.tic();
    contour_db.queryRangedKNN(ptr_cm_tgt, thres_lb_, thres_ub_, ptr_cands, cand_corr, bev_tfs);
    printf("%lu Candidates in %7.5fs: \n", ptr_cands.size(), clk.toc());

//    if(laser_info_tgt.seq == 894){
//      printf("Manual break point here.\n");
//    }

    // 2.1 process query results
    CHECK(ptr_cands.size() < 2);
    PredictionOutcome pred_res;
    if (ptr_cands.empty())
      pred_res = evaluator.addPrediction(ptr_cm_tgt);
    else {
      pred_res = evaluator.addPrediction(ptr_cm_tgt, ptr_cands[0], bev_tfs[0]);
      new_lc_pairs.emplace_back(ptr_cm_tgt->getIntID(), ptr_cands[0]->getIntID());
      // save images of pairs
      std::string f_name =
          PROJ_DIR + "/results/match_comp_img/lc_" + ptr_cm_tgt->getStrID() + "-" + ptr_cands[0]->getStrID() +
          ".png";
      ContourManager::saveMatchedPairImg(f_name, *ptr_cm_tgt, *ptr_cands[0]);
      printf("Image saved: %s-%s\n", ptr_cm_tgt->getStrID().c_str(), ptr_cands[0]->getStrID().c_str());

    }

    switch (pred_res.tfpn) {
      case PredictionOutcome::TP:
        printf("Prediction outcome: TP\n");
        cnt_tp++;
        break;
      case PredictionOutcome::FP:
        printf("Prediction outcome: FP\n");
        cnt_fp++;
        break;
      case PredictionOutcome::TN:
        printf("Prediction outcome: TN\n");
        break;
      case PredictionOutcome::FN:
        printf("Prediction outcome: FN\n");
        cnt_fn++;
        break;
    }

    printf("TP Error mean: t:%7.4f m, r:%7.4f deg\n", evaluator.getTPMeanTrans(), evaluator.getTPMeanRot());
    printf("TP Error rmse: t:%7.4f m, r:%7.4f deg\n", evaluator.getTPRMSETrans(), evaluator.getTPRMSERot());
    printf("Accumulated tp poses: %d\n", cnt_tp);
    printf("Accumulated fn poses: %d\n", cnt_fn);
    printf("Accumulated fp poses: %d\n", cnt_fp);

    // 3. update database
    // add scan
    contour_db.addScan(ptr_cm_tgt, laser_info_tgt.ts);
    // balance
    clk.tic();
    contour_db.pushAndBalance(laser_info_tgt.seq, laser_info_tgt.ts);
    printf("Rebalance tree cost: %7.5f\n", clk.toc());

    // 4. publish new vis
    publishLCConnections(new_lc_pairs, wall_time_ros);

    return 0;
  }

  void savePredictionResults(const std::string &sav_path) const {
    evaluator.savePredictionResults(sav_path);
  }
};


int main(int argc, char **argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);

  ros::init(argc, argv, "batch_bin_test");
  ros::NodeHandle nh;

  printf("batch bin test start\n");

  const int max_spin_cnt = 407000;
  const int max_spin_att = 123450;

//  // Two file path
//  std::string fpath_sens_gt_pose = PROJ_DIR + "/sample_data/ts-sens_pose-kitti00.txt";
//  std::string fpath_lidar_bins = PROJ_DIR + "/sample_data/ts-lidar_bins-kitti00.txt";
//  // Check thres path
//  std::string cand_score_config = PROJ_DIR + "/config/score_thres_kitti_bag_play.cfg";
//  // Sav path
//  std::string fpath_outcome_sav = PROJ_DIR + "/results/outcome_txt/outcome-kitti00.txt";

//  // KITTI 08
//  std::string fpath_sens_gt_pose = PROJ_DIR + "/sample_data/ts-sens_pose-kitti08.txt";
//  std::string fpath_lidar_bins = PROJ_DIR + "/sample_data/ts-lidar_bins-kitti08.txt";
//  std::string fpath_outcome_sav = PROJ_DIR + "/results/outcome_txt/outcome-kitti08.txt";
//  // Check thres path
//  std::string cand_score_config = PROJ_DIR + "/config/score_thres_kitti_bag_play.cfg";


//  // Mulran KAIST 01
//  std::string fpath_sens_gt_pose = PROJ_DIR + "/sample_data/ts-sens_pose-mulran-kaist01.txt";
//  std::string fpath_lidar_bins = PROJ_DIR + "/sample_data/ts-lidar_bins-mulran-kaist01.txt";
//  std::string fpath_outcome_sav = PROJ_DIR + "/results/outcome_txt/outcome-mulran-kaist01.txt";

  // Mulran KAIST 01
  std::string fpath_sens_gt_pose = PROJ_DIR + "/sample_data/ts-sens_pose-mulran-kaist01.txt";
  std::string fpath_lidar_bins = PROJ_DIR + "/sample_data/ts-lidar_bins-mulran-kaist01.txt";
  std::string fpath_outcome_sav = PROJ_DIR + "/results/outcome_txt/outcome-mulran-kaist01.txt";

//  // Mulran Riverside 02
//  std::string fpath_sens_gt_pose = PROJ_DIR + "/sample_data/ts-sens_pose-mulran-rs02.txt";
//  std::string fpath_lidar_bins = PROJ_DIR + "/sample_data/ts-lidar_bins-mulran-rs02.txt";
//  std::string fpath_outcome_sav = PROJ_DIR + "/results/outcome_txt/outcome-mulran-rs02.txt";


  // Check thres path
  std::string cand_score_config = PROJ_DIR + "/config/score_thres_kitti_bag_play.cfg";

  // Main process:
  ContourDBConfig db_config;
  std::vector<int> db_q_levels = {1, 2, 3};

  BatchBinSpinner o(nh, db_config, db_q_levels, fpath_sens_gt_pose, fpath_lidar_bins);
  o.loadThres(cand_score_config);

  ros::Rate rate(30);
  int cnt = 0;
  int cnt_attempted = 0;

  printf("\nHold for 3 seconds...\n");
  std::this_thread::sleep_for(std::chrono::duration<double>(3.0));  // human readability: have time to see init output

  while (ros::ok()) {
    ros::spinOnce();
    if (cnt_attempted > max_spin_att) {
      printf("Max spin attempt times %d reached. Exiting the loop...\n", max_spin_att);
      break;
    }

    cnt_attempted++;
    int ret_code = o.spinOnce(cnt);
    if (ret_code == -2 || ret_code == 1)
      ros::Duration(1.0).sleep();
    else if (ret_code == -1)
      break;

    rate.sleep();
  }

  o.savePredictionResults(fpath_outcome_sav);


  return 0;
}