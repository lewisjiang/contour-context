//
// Created by lewis on 6/24/23.
//
// This file tries to register 2 point clouds in KITTI dataset using cont2

#include <memory>
#include <utility>

#include "cont2/contour_db.h"
#include "eval/evaluator.h"
#include "cont2_ros/spinner_ros.h"
#include "tools/bm_util.h"
#include "tools/config_handler.h"

const std::string PROJ_DIR = std::string(PJSRCDIR);

SequentialTimeProfiler stp;

Eigen::Matrix4d GetTr(const std::string &calib_file) {
  std::fstream f;
  f.open(calib_file, std::ios::in);
  if (!f.is_open()) {
    std::cerr << "Cannot open calib file: " << calib_file << std::endl;
  }
  std::string line;
  Eigen::Matrix4d Tr = Eigen::Matrix4d::Identity();
  while (std::getline(f, line)) {
    std::stringstream ss(line);
    std::string tag;
    ss >> tag;
    if (tag == "Tr:") {
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
          ss >> Tr(i, j);
        }
      }
    }
  }
  return Tr;
}

void LoadKittiPose(const std::string &pose_file, const std::string &calib_file, std::vector<Eigen::Matrix4d> &poses) {
  //    read kitti pose txt
  std::fstream f;
  f.open(pose_file, std::ios::in);
  if (!f.is_open()) {
    LOG(FATAL) << "Cannot open pose file: " << pose_file;
  }
  Eigen::Matrix4d Tr = GetTr(calib_file);
  std::string line;
  while (std::getline(f, line)) {
    std::stringstream ss(line);
    Eigen::Matrix4d T_lcam0_lcam = Eigen::Matrix4d::Identity();
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 4; ++j) {
        ss >> T_lcam0_lcam(i, j);
      }
    }
    // Tr is T_leftcam_velod_
    // T_w_velod = np.linalg.inv(T_leftcam_velod_) @ tmp_T_lc0_lc @ T_leftcam_velod_
    Eigen::Matrix4d Twl = Tr.inverse() * T_lcam0_lcam * Tr;
    poses.push_back(Twl);
  }
}

class RegWorker {
  ContourManagerConfig cm_config;
  ContourDBConfig db_config;

  CandidateScoreEnsemble thres_lb_, thres_ub_;  // check thresholds


public:
  void loadConfig(const std::string &config_fpath) {

    printf("Loading parameters...\n");
    auto yl = yamlLoader(config_fpath);

    std::string fpath_sens_gt_pose, fpath_lidar_bins;
    double corr_thres;

    yl.loadOneConfig({"fpath_sens_gt_pose"}, fpath_sens_gt_pose);
    yl.loadOneConfig({"fpath_lidar_bins"}, fpath_lidar_bins);
    yl.loadOneConfig({"correlation_thres"}, corr_thres);
//    ptr_evaluator = std::make_unique<ContLCDEvaluator>(fpath_sens_gt_pose, fpath_lidar_bins, corr_thres);

    yl.loadOneConfig({"ContourDBConfig", "nnk_"}, db_config.nnk_);
    yl.loadOneConfig({"ContourDBConfig", "max_fine_opt_"}, db_config.max_fine_opt_);
    yl.loadSeqConfig({"ContourDBConfig", "q_levels_"}, db_config.q_levels_);

    yl.loadOneConfig({"ContourDBConfig", "TreeBucketConfig", "max_elapse_"}, db_config.tb_cfg_.max_elapse_);
    yl.loadOneConfig({"ContourDBConfig", "TreeBucketConfig", "min_elapse_"}, db_config.tb_cfg_.min_elapse_);

    yl.loadOneConfig({"ContourDBConfig", "ContourSimThresConfig", "ta_cell_cnt"}, db_config.cont_sim_cfg_.ta_cell_cnt);
    yl.loadOneConfig({"ContourDBConfig", "ContourSimThresConfig", "tp_cell_cnt"}, db_config.cont_sim_cfg_.tp_cell_cnt);
    yl.loadOneConfig({"ContourDBConfig", "ContourSimThresConfig", "tp_eigval"}, db_config.cont_sim_cfg_.tp_eigval);
    yl.loadOneConfig({"ContourDBConfig", "ContourSimThresConfig", "ta_h_bar"}, db_config.cont_sim_cfg_.ta_h_bar);
    yl.loadOneConfig({"ContourDBConfig", "ContourSimThresConfig", "ta_rcom"}, db_config.cont_sim_cfg_.ta_rcom);
    yl.loadOneConfig({"ContourDBConfig", "ContourSimThresConfig", "tp_rcom"}, db_config.cont_sim_cfg_.tp_rcom);

    yl.loadOneConfig({"thres_lb_", "i_ovlp_sum"}, thres_lb_.sim_constell.i_ovlp_sum);
    yl.loadOneConfig({"thres_lb_", "i_ovlp_max_one"}, thres_lb_.sim_constell.i_ovlp_max_one);
    yl.loadOneConfig({"thres_lb_", "i_in_ang_rng"}, thres_lb_.sim_constell.i_in_ang_rng);
    yl.loadOneConfig({"thres_lb_", "i_indiv_sim"}, thres_lb_.sim_pair.i_indiv_sim);
    yl.loadOneConfig({"thres_lb_", "i_orie_sim"}, thres_lb_.sim_pair.i_orie_sim);
    yl.loadOneConfig({"thres_lb_", "correlation"}, thres_lb_.sim_post.correlation);
    yl.loadOneConfig({"thres_lb_", "area_perc"}, thres_lb_.sim_post.area_perc);
    yl.loadOneConfig({"thres_lb_", "neg_est_dist"}, thres_lb_.sim_post.neg_est_dist);

    yl.loadOneConfig({"thres_ub_", "i_ovlp_sum"}, thres_ub_.sim_constell.i_ovlp_sum);
    yl.loadOneConfig({"thres_ub_", "i_ovlp_max_one"}, thres_ub_.sim_constell.i_ovlp_max_one);
    yl.loadOneConfig({"thres_ub_", "i_in_ang_rng"}, thres_ub_.sim_constell.i_in_ang_rng);
    yl.loadOneConfig({"thres_ub_", "i_indiv_sim"}, thres_ub_.sim_pair.i_indiv_sim);
    yl.loadOneConfig({"thres_ub_", "i_orie_sim"}, thres_ub_.sim_pair.i_orie_sim);
    yl.loadOneConfig({"thres_ub_", "correlation"}, thres_ub_.sim_post.correlation);
    yl.loadOneConfig({"thres_ub_", "area_perc"}, thres_ub_.sim_post.area_perc);
    yl.loadOneConfig({"thres_ub_", "neg_est_dist"}, thres_ub_.sim_post.neg_est_dist);

    yl.loadSeqConfig({"ContourManagerConfig", "lv_grads_"}, cm_config.lv_grads_);
    yl.loadOneConfig({"ContourManagerConfig", "reso_row_"}, cm_config.reso_row_);
    yl.loadOneConfig({"ContourManagerConfig", "reso_col_"}, cm_config.reso_col_);
    yl.loadOneConfig({"ContourManagerConfig", "n_row_"}, cm_config.n_row_);
    yl.loadOneConfig({"ContourManagerConfig", "n_col_"}, cm_config.n_col_);
    yl.loadOneConfig({"ContourManagerConfig", "lidar_height_"}, cm_config.lidar_height_);
    yl.loadOneConfig({"ContourManagerConfig", "blind_sq_"}, cm_config.blind_sq_);
    yl.loadOneConfig({"ContourManagerConfig", "min_cont_key_cnt_"}, cm_config.min_cont_key_cnt_);
    yl.loadOneConfig({"ContourManagerConfig", "min_cont_cell_cnt_"}, cm_config.min_cont_cell_cnt_);

    yl.close();
  }

  /// Keep the timestamp with enough distance manually or the tgt may exist in cache instead of DB.
  /// \param ptr_new The new pose looking for a match. (tgt in cont2)
  /// \param ptr_old The old pose to be matched to. (src in cont2)
  /// \return T_old_new
  Eigen::Isometry3d estimateTF(pcl::PointCloud<pcl::PointXYZI>::ConstPtr ptr_new,
                               pcl::PointCloud<pcl::PointXYZI>::ConstPtr ptr_old) const {
    std::unique_ptr<ContourDB> ptr_contour_db;
    ptr_contour_db = std::make_unique<ContourDB>(db_config);

    // prepare database
    std::shared_ptr<ContourManager> cmng_ptr_old(new ContourManager(cm_config, 0));
    cmng_ptr_old->makeBEV<pcl::PointXYZI>(ptr_old, "old_0");
    cmng_ptr_old->makeContoursRecurs();

    ptr_contour_db->addScan(cmng_ptr_old, 0);
    ptr_contour_db->pushAndBalance(0, 50);

    // use new data.
    std::shared_ptr<ContourManager> cmng_ptr_new(new ContourManager(cm_config, 1));
    cmng_ptr_new->makeBEV<pcl::PointXYZI>(ptr_new, "new_0");
    cmng_ptr_new->makeContoursRecurs();

    std::vector<std::pair<int, int>> new_lc_pairs;
    std::vector<bool> new_lc_tfp;
    std::vector<std::shared_ptr<const ContourManager>> ptr_cands;
    std::vector<double> cand_corr;
    std::vector<Eigen::Isometry2d> bev_tfs;  // T_new_old

    ptr_contour_db->queryRangedKNN(cmng_ptr_new, thres_lb_, thres_ub_, ptr_cands, cand_corr, bev_tfs);
    CHECK(ptr_cands.size() < 2);

    if (ptr_cands.empty()) {
      printf("No loop found.\n");
      return Eigen::Isometry3d::Identity();
    } else {
      // recover 3D TF from 3DoF TF: copied from correlation.h

      Eigen::Isometry2d T_so_ssen = Eigen::Isometry2d::Identity(), T_to_tsen;  // {}_sensor in {}_bev_origin frame
      T_so_ssen.translate(V2D(cm_config.n_row_ / 2 - 0.5, cm_config.n_col_ / 2 - 0.5));
      T_to_tsen = T_so_ssen;
      Eigen::Isometry2d T_tsen_ssen2_est = T_to_tsen.inverse() * bev_tfs[0].inverse() * T_so_ssen;
      T_tsen_ssen2_est.translation() *= cm_config.reso_row_;

      Eigen::Isometry3d T_res = Eigen::Isometry3d::Identity();
      Eigen::Matrix3d T_rot_3d = Eigen::Matrix3d::Identity();
      T_rot_3d.topLeftCorner<2, 2>() = T_tsen_ssen2_est.rotation();  // rotate around z axis
      T_res.rotate(Eigen::Quaterniond(T_rot_3d));
      T_res.pretranslate(Eigen::Vector3d(T_tsen_ssen2_est.translation().x(), T_tsen_ssen2_est.translation().y(), 0));
      return T_res;
    }

  }
};

int main(int argc, char **argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);

  ros::init(argc, argv, "kitti_reg_test");
  ros::NodeHandle nh;

  std::string lidar_path = "/media/lewis/S7/Datasets/kitti/odometry/dataset/sequences/00/velodyne/";
  std::string pose_path = "/media/lewis/S7/Datasets/sematic_kitti/odometry/dataset/sequences/00/poses.txt";
  std::string calib_path = "/media/lewis/S7/Datasets/kitti/odometry/dataset/sequences/00/calib.txt";

  RegWorker worker;
  const std::string config_path = PROJ_DIR + "/config/kitti_reg_test_config.yaml";
  worker.loadConfig(config_path);

  for (int i = 1; i < 2; i++) {
    // different cont2 lib, here src means new, tgt means old
    int src_idx = i, dst_idx = 0;
    std::stringstream lidar_data_path;
    lidar_data_path << lidar_path << std::setfill('0') << std::setw(6) << src_idx << ".bin";
    pcl::PointCloud<pcl::PointXYZI>::ConstPtr src_data = readKITTIPointCloudBin<pcl::PointXYZI>(lidar_data_path.str());
    lidar_data_path.str("");
    lidar_data_path << lidar_path << std::setfill('0') << std::setw(6) << dst_idx << ".bin";
    pcl::PointCloud<pcl::PointXYZI>::ConstPtr tgt_data = readKITTIPointCloudBin<pcl::PointXYZI>(lidar_data_path.str());

    std::vector<Eigen::Matrix4d> poses;
    LoadKittiPose(pose_path, calib_path, poses);
    std::cout << "Sucessfully load pose with number: " << poses.size() << std::endl;

    Eigen::Matrix4d T_ts_gt = poses[dst_idx].inverse() * poses[src_idx];

    std::cout << "T_src:\n" << poses[src_idx] << std::endl;
    std::cout << "T_tgt:\n" << poses[dst_idx] << std::endl;

    // start registration:
    Eigen::Isometry3d T_est = worker.estimateTF(src_data, tgt_data);
    std::cout << "T_est:\n" << T_est.matrix() << std::endl;
    std::cout << "T_gt:\n" << T_ts_gt.matrix() << std::endl;
    printf("Est error %d-%d:\n", src_idx, dst_idx);
    std::cout << T_ts_gt * (T_est.matrix().inverse()) << std::endl;
  }


  return 0;

}