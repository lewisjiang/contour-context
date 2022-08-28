//
// Created by lewis on 8/22/22.
//

#include "eval/evaluator.h"

void ContLCDEvaluator::loadCheckThres(const std::string &fpath, CandidateScoreEnsemble &thres_lb,
                                      CandidateScoreEnsemble &thres_ub) {
  std::fstream infile;
  infile.open(fpath, std::ios::in);

  if (infile.rdstate() != std::ifstream::goodbit) {
    std::cerr << "Error opening thres config file: " << fpath << std::endl;
    return;
  }

  std::string sbuf, pname;
  while (std::getline(infile, sbuf)) {
    std::istringstream iss(sbuf);
    if (iss >> pname) {
      std::cout << pname << std::endl;
      if (pname[0] == '#')
        continue;
      if (pname == "i_ovlp_sum") {
        iss >> thres_lb.sim_constell.i_ovlp_sum;
        iss >> thres_ub.sim_constell.i_ovlp_sum;
      } else if (pname == "i_ovlp_max_one") {
        iss >> thres_lb.sim_constell.i_ovlp_max_one;
        iss >> thres_ub.sim_constell.i_ovlp_max_one;
      } else if (pname == "i_in_ang_rng") {
        iss >> thres_lb.sim_constell.i_in_ang_rng;
        iss >> thres_ub.sim_constell.i_in_ang_rng;
      } else if (pname == "i_indiv_sim") {
        iss >> thres_lb.sim_pair.i_indiv_sim;
        iss >> thres_ub.sim_pair.i_indiv_sim;
      } else if (pname == "i_orie_sim") {
        iss >> thres_lb.sim_pair.i_orie_sim;
        iss >> thres_ub.sim_pair.i_orie_sim;
//        } else if (pname == "f_area_perc") {
//          iss >> thres_lb_.sim_pair.f_area_perc;
//          iss >> thres_ub_.sim_pair.f_area_perc;
//        } else if (pname == "correlation") {
//          iss >> thres_lb_.correlation;
//          iss >> thres_ub_.correlation;
//        } else if (pname == "area_perc") {
//          iss >> thres_lb_.area_perc;
//          iss >> thres_ub_.area_perc;
//        }

      } else if (pname == "correlation") {
        iss >> thres_lb.sim_post.correlation;
        iss >> thres_ub.sim_post.correlation;
      } else if (pname == "area_perc") {
        iss >> thres_lb.sim_post.area_perc;
        iss >> thres_ub.sim_post.area_perc;
      } else if (pname == "neg_est_dist") {
        iss >> thres_lb.sim_post.neg_est_dist;
        iss >> thres_ub.sim_post.neg_est_dist;
      }
    }
  }

  infile.close();
}
