//
// Created by lewis on 8/27/22.
//

#include <memory>
#include <utility>

#include "cont2/contour_db.h"
#include "eval/evaluator.h"
#include "cont2_ros/spinner_ros.h"
#include "tools/bm_util.h"
#include "tools/config_handler.h"

const std::string PROJ_DIR = std::string(PJSRCDIR);

SequentialTimeProfiler stp;

class BatchBinSpinner : public BaseROSSpinner {
  // --- Added members for evaluation and LC module running ---
  std::unique_ptr<ContourDB> ptr_contour_db;
  std::unique_ptr<ContLCDEvaluator> ptr_evaluator;

  ContourManagerConfig cm_config;
  ContourDBConfig db_config;

  CandidateScoreEnsemble thres_lb_, thres_ub_;  // check thresholds

  // bookkeeping
  int cnt_tp = 0, cnt_fn = 0, cnt_fp = 0;
  double ts_beg = -1;

public:
  explicit BatchBinSpinner(ros::NodeHandle &nh_) : BaseROSSpinner(nh_) {  // mf1 k02

  }

  // before start: 1/1: load thres
  void loadConfig(const std::string &config_fpath, std::string &sav_path) {
    cv::FileStorage fs;

    printf("Loading parameters...\n");
    fs.open(config_fpath, cv::FileStorage::READ);

    std::string fpath_sens_gt_pose, fpath_lidar_bins;
    double corr_thres;

    loadOneConfig(fs, {"fpath_sens_gt_pose"}, fpath_sens_gt_pose);
    loadOneConfig(fs, {"fpath_lidar_bins"}, fpath_lidar_bins);
    loadOneConfig(fs, {"correlation_thres"}, corr_thres);
    ptr_evaluator = std::make_unique<ContLCDEvaluator>(fpath_sens_gt_pose, fpath_lidar_bins, corr_thres);

    loadOneConfig(fs, {"ContourDBConfig", "nnk_"}, db_config.nnk_);
    loadOneConfig(fs, {"ContourDBConfig", "max_fine_opt_"}, db_config.max_fine_opt_);
    loadSeqConfig(fs, {"ContourDBConfig", "q_levels_"}, db_config.q_levels_);

    loadOneConfig(fs, {"ContourDBConfig", "ContourSimThresConfig", "ta_cell_cnt"}, db_config.cont_sim_cfg_.ta_cell_cnt);
    loadOneConfig(fs, {"ContourDBConfig", "ContourSimThresConfig", "tp_cell_cnt"}, db_config.cont_sim_cfg_.tp_cell_cnt);
    loadOneConfig(fs, {"ContourDBConfig", "ContourSimThresConfig", "tp_eigval"}, db_config.cont_sim_cfg_.tp_eigval);
    loadOneConfig(fs, {"ContourDBConfig", "ContourSimThresConfig", "ta_h_bar"}, db_config.cont_sim_cfg_.ta_h_bar);
    loadOneConfig(fs, {"ContourDBConfig", "ContourSimThresConfig", "ta_rcom"}, db_config.cont_sim_cfg_.ta_rcom);
    loadOneConfig(fs, {"ContourDBConfig", "ContourSimThresConfig", "tp_rcom"}, db_config.cont_sim_cfg_.tp_rcom);
    ptr_contour_db = std::make_unique<ContourDB>(db_config);

    loadOneConfig(fs, {"thres_lb_", "i_ovlp_sum"}, thres_lb_.sim_constell.i_ovlp_sum);
    loadOneConfig(fs, {"thres_lb_", "i_ovlp_max_one"}, thres_lb_.sim_constell.i_ovlp_max_one);
    loadOneConfig(fs, {"thres_lb_", "i_in_ang_rng"}, thres_lb_.sim_constell.i_in_ang_rng);
    loadOneConfig(fs, {"thres_lb_", "i_indiv_sim"}, thres_lb_.sim_pair.i_indiv_sim);
    loadOneConfig(fs, {"thres_lb_", "i_orie_sim"}, thres_lb_.sim_pair.i_orie_sim);
    loadOneConfig(fs, {"thres_lb_", "correlation"}, thres_lb_.sim_post.correlation);
    loadOneConfig(fs, {"thres_lb_", "area_perc"}, thres_lb_.sim_post.area_perc);
    loadOneConfig(fs, {"thres_lb_", "neg_est_dist"}, thres_lb_.sim_post.neg_est_dist);

    loadOneConfig(fs, {"thres_ub_", "i_ovlp_sum"}, thres_ub_.sim_constell.i_ovlp_sum);
    loadOneConfig(fs, {"thres_ub_", "i_ovlp_max_one"}, thres_ub_.sim_constell.i_ovlp_max_one);
    loadOneConfig(fs, {"thres_ub_", "i_in_ang_rng"}, thres_ub_.sim_constell.i_in_ang_rng);
    loadOneConfig(fs, {"thres_ub_", "i_indiv_sim"}, thres_ub_.sim_pair.i_indiv_sim);
    loadOneConfig(fs, {"thres_ub_", "i_orie_sim"}, thres_ub_.sim_pair.i_orie_sim);
    loadOneConfig(fs, {"thres_ub_", "correlation"}, thres_ub_.sim_post.correlation);
    loadOneConfig(fs, {"thres_ub_", "area_perc"}, thres_ub_.sim_post.area_perc);
    loadOneConfig(fs, {"thres_ub_", "neg_est_dist"}, thres_ub_.sim_post.neg_est_dist);

    loadSeqConfig(fs, {"ContourManagerConfig", "lv_grads_"}, cm_config.lv_grads_);
    loadOneConfig(fs, {"ContourManagerConfig", "reso_row_"}, cm_config.reso_row_);
    loadOneConfig(fs, {"ContourManagerConfig", "reso_col_"}, cm_config.reso_col_);
    loadOneConfig(fs, {"ContourManagerConfig", "n_row_"}, cm_config.n_row_);
    loadOneConfig(fs, {"ContourManagerConfig", "n_col_"}, cm_config.n_col_);
    loadOneConfig(fs, {"ContourManagerConfig", "lidar_height_"}, cm_config.lidar_height_);
    loadOneConfig(fs, {"ContourManagerConfig", "blind_sq_"}, cm_config.blind_sq_);
    loadOneConfig(fs, {"ContourManagerConfig", "min_cont_key_cnt_"}, cm_config.min_cont_key_cnt_);
    loadOneConfig(fs, {"ContourManagerConfig", "min_cont_cell_cnt_"}, cm_config.min_cont_cell_cnt_);

    loadOneConfig(fs, {"fpath_outcome_sav"}, sav_path);

    fs.release();
  }

  ///
  /// \param outer_cnt
  /// \return 0: normal. <0: external signal. 1: load failed
  int spinOnce(int &outer_cnt) {
    CHECK(ptr_contour_db && ptr_evaluator);
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

    bool loaded = ptr_evaluator->loadNewScan();
    if (!loaded) {
      printf("Load new scan failed.\n");
      return 1;
    }
    TicToc clk;
    ros::Time wall_time_ros = ros::Time::now();
    outer_cnt++;

    // 1. Init current scan

    stp.lap();
    stp.start();
    std::shared_ptr<ContourManager> ptr_cm_tgt = ptr_evaluator->getCurrContourManager(cm_config);
    stp.record("make bev");
    const auto laser_info_tgt = ptr_evaluator->getCurrScanInfo();
    printf("\n===\nLoaded: assigned seq: %d, bin path: %s\n", laser_info_tgt.seq, laser_info_tgt.fpath.c_str());

    // 1.1 Prepare and display info: gt/shifted pose, tf
    double ts_curr = laser_info_tgt.ts;
    if (ts_beg < 0) ts_beg = ts_curr;

    Eigen::Isometry3d T_gt_curr = laser_info_tgt.sens_pose;
    Eigen::Vector3d time_translate(0, 0, 10);
    time_translate = time_translate * (ts_curr - ts_beg) / 60;  // 10m per min
    g_poses.insert(std::make_pair(laser_info_tgt.seq, GlobalPoseInfo(T_gt_curr, time_translate.z())));

#if PUB_ROS_MSG
    geometry_msgs::TransformStamped tf_gt_curr = tf2::eigenToTransform(T_gt_curr);
    broadcastCurrPose(tf_gt_curr);  // the stamp is now

    tf_gt_curr.header.seq = laser_info_tgt.seq;
    tf_gt_curr.transform.translation.z += time_translate.z();
    publishPath(wall_time_ros, tf_gt_curr);
    if (laser_info_tgt.seq % 50 == 0)  // It is laggy to display too many characters in rviz
      publishScanSeqText(wall_time_ros, tf_gt_curr, laser_info_tgt.seq);
#endif

    // 1.2. save images of layers

#if SAVE_MID_FILE
    clk.tic();
    for (int i = 0; i < cm_config.lv_grads_.size(); i++) {
      std::string f_name = PROJ_DIR + "/results/layer_img/contour_" + "lv" + std::to_string(i) + "_" +
                           ptr_cm_tgt->getStrID() + ".png";   // TODO: what should be the str name of scans?
      ptr_cm_tgt->saveContourImage(f_name, i);
    }
    std::cout << "Time save layers: " << clk.toctic() << std::endl;
#endif
    ptr_cm_tgt->clearImage();  // a must to save memory

    // 2. query
    std::vector<std::pair<int, int>> new_lc_pairs;
    std::vector<bool> new_lc_tfp;
    std::vector<std::shared_ptr<const ContourManager>> ptr_cands;
    std::vector<double> cand_corr;
    std::vector<Eigen::Isometry2d> bev_tfs;

    clk.tic();
    ptr_contour_db->queryRangedKNN(ptr_cm_tgt, thres_lb_, thres_ub_, ptr_cands, cand_corr, bev_tfs);
    printf("%lu Candidates in %7.5fs: \n", ptr_cands.size(), clk.toc());

//    if(laser_info_tgt.seq == 894){
//      printf("Manual break point here.\n");
//    }

    // 2.1 process query results
    CHECK(ptr_cands.size() < 2);
    PredictionOutcome pred_res;
    if (ptr_cands.empty())
      pred_res = ptr_evaluator->addPrediction(ptr_cm_tgt, 0.0);
    else {
      pred_res = ptr_evaluator->addPrediction(ptr_cm_tgt, cand_corr[0], ptr_cands[0], bev_tfs[0]);
      if (pred_res.tfpn == PredictionOutcome::TP || pred_res.tfpn == PredictionOutcome::FP) {
        new_lc_pairs.emplace_back(ptr_cm_tgt->getIntID(), ptr_cands[0]->getIntID());
        new_lc_tfp.emplace_back(pred_res.tfpn == PredictionOutcome::TP);
#if SAVE_MID_FILE
        // save images of pairs
        std::string f_name =
            PROJ_DIR + "/results/match_comp_img/lc_" + ptr_cm_tgt->getStrID() + "-" + ptr_cands[0]->getStrID() +
            ".png";
        ContourManager::saveMatchedPairImg(f_name, *ptr_cm_tgt, *ptr_cands[0]);
        printf("Image saved: %s-%s\n", ptr_cm_tgt->getStrID().c_str(), ptr_cands[0]->getStrID().c_str());
#endif
      }
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

    printf("TP Error mean: t:%7.4f m, r:%7.4f rad\n", ptr_evaluator->getTPMeanTrans(), ptr_evaluator->getTPMeanRot());
    printf("TP Error rmse: t:%7.4f m, r:%7.4f rad\n", ptr_evaluator->getTPRMSETrans(), ptr_evaluator->getTPRMSERot());
    printf("Accumulated tp poses: %d\n", cnt_tp);
    printf("Accumulated fn poses: %d\n", cnt_fn);
    printf("Accumulated fp poses: %d\n", cnt_fp);

    stp.start();
    // 3. update database
    // add scan
    ptr_contour_db->addScan(ptr_cm_tgt, laser_info_tgt.ts);
    // balance
    clk.tic();
    ptr_contour_db->pushAndBalance(laser_info_tgt.seq, laser_info_tgt.ts);
    stp.record("Update database");
    printf("Rebalance tree cost: %7.5f\n", clk.toc());

#if PUB_ROS_MSG
    // 4. publish new vis
    publishLCConnections(new_lc_pairs, new_lc_tfp, wall_time_ros);
#endif

    return 0;
  }

  void savePredictionResults(const std::string &sav_path) const {
    ptr_evaluator->savePredictionResults(sav_path);
  }

  inline int get_tp() const { return cnt_tp; }

  inline int get_fp() const { return cnt_fp; }

  inline int get_fn() const { return cnt_fn; }
};


int main(int argc, char **argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);

  ros::init(argc, argv, "batch_bin_test");
  ros::NodeHandle nh;

  printf("batch bin test start\n");


//  // Two file path
//  std::string fpath_sens_gt_pose = PROJ_DIR + "/sample_data/ts-sens_pose-kitti00.txt";
//  std::string fpath_lidar_bins = PROJ_DIR + "/sample_data/ts-lidar_bins-kitti00.txt";
//  // Check thres path
//  std::string cand_score_config = PROJ_DIR + "/config/score_thres_kitti_bag_play.cfg";
//  // Sav path
//  std::string fpath_outcome_sav = PROJ_DIR + "/results/outcome_txt/outcome-kitti00.txt";

//  std::string ksq = "08";

//  // KITTI 08
//  std::string fpath_sens_gt_pose = PROJ_DIR + "/sample_data/ts-sens_pose-kitti" + ksq + ".txt";
//  std::string fpath_lidar_bins = PROJ_DIR + "/sample_data/ts-lidar_bins-kitti" + ksq + ".txt";
//  std::string fpath_outcome_sav = PROJ_DIR + "/results/outcome_txt/outcome-kitti" + ksq + ".txt";
//  // Check thres path
//  std::string cand_score_config = PROJ_DIR + "/config/score_thres_kitti_bag_play.cfg";
//
//  // KITTI 51: Mulran as KITTI, KAIST01
//  std::string fpath_sens_gt_pose = PROJ_DIR + "/sample_data/ts-sens_pose-kitti51.txt";
//  std::string fpath_lidar_bins = PROJ_DIR + "/sample_data/ts-lidar_bins-kitti51.txt";
//  std::string fpath_outcome_sav = PROJ_DIR + "/results/outcome_txt/outcome-kitti51.txt";

//  // KITTI 62: Mulran as KITTI, RS02
//  std::string fpath_sens_gt_pose = PROJ_DIR + "/sample_data/ts-sens_pose-kitti62.txt";
//  std::string fpath_lidar_bins = PROJ_DIR + "/sample_data/ts-lidar_bins-kitti62.txt";
//  std::string fpath_outcome_sav = PROJ_DIR + "/results/outcome_txt/outcome-kitti62.txt";

//  // KITTI 72: Mulran as KITTI, DCC02
//  std::string fpath_sens_gt_pose = PROJ_DIR + "/sample_data/ts-sens_pose-kitti72.txt";
//  std::string fpath_lidar_bins = PROJ_DIR + "/sample_data/ts-lidar_bins-kitti72.txt";
//  std::string fpath_outcome_sav = PROJ_DIR + "/results/outcome_txt/outcome-kitti72.txt";


//  // Mulran KAIST 01
//  std::string fpath_sens_gt_pose = PROJ_DIR + "/sample_data/ts-sens_pose-mulran-kaist01.txt";
//  std::string fpath_lidar_bins = PROJ_DIR + "/sample_data/ts-lidar_bins-mulran-kaist01.txt";
//  std::string fpath_outcome_sav = PROJ_DIR + "/results/outcome_txt/outcome-mulran-kaist01.txt";

//  // Mulran Riverside 02
//  std::string fpath_sens_gt_pose = PROJ_DIR + "/sample_data/ts-sens_pose-mulran-rs02.txt";
//  std::string fpath_lidar_bins = PROJ_DIR + "/sample_data/ts-lidar_bins-mulran-rs02.txt";
//  std::string fpath_outcome_sav = PROJ_DIR + "/results/outcome_txt/outcome-mulran-rs02.txt";

//  // Mulran DCC 02
//  std::string fpath_sens_gt_pose = PROJ_DIR + "/sample_data/ts-sens_pose-mulran-dcc02.txt";
//  std::string fpath_lidar_bins = PROJ_DIR + "/sample_data/ts-lidar_bins-mulran-dcc02.txt";
//  std::string fpath_outcome_sav = PROJ_DIR + "/results/outcome_txt/outcome-mulran-dcc.txt";

  // Check thres path
//  std::string cand_score_config = PROJ_DIR + "/config/score_thres_kitti_bag_play.cfg";
  std::string cand_score_config = "/home/lewis/catkin_ws2/src/contour-context/config/batch_bin_test_config.yaml";

  // Main process:
  BatchBinSpinner o(nh);

  std::string fpath_outcome_sav;
  o.loadConfig(cand_score_config, fpath_outcome_sav);

  stp = SequentialTimeProfiler(fpath_outcome_sav);

  ros::Rate rate(300);
  int cnt = 0;

  printf("\nHold for 3 seconds...\n");
  std::this_thread::sleep_for(std::chrono::duration<double>(3.0));  // human readability: have time to see init output

  while (ros::ok()) {
    ros::spinOnce();

    int ret_code = o.spinOnce(cnt);
    if (ret_code == -2 || ret_code == 1)
      ros::Duration(1.0).sleep();
    else if (ret_code == -1)
      break;

    rate.sleep();
  }

  o.savePredictionResults(fpath_outcome_sav);

  stp.printScreen(true);
  const std::string log_dir = PROJ_DIR + "/log/";
  stp.printFile(log_dir + "timing_cont2.txt", true);


  return 0;
}