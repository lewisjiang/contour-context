//
// Created by lewis on 5/5/22.
//

/*
 * Slice actually...
 * */

#ifndef CONT2_CONTOUR_H
#define CONT2_CONTOUR_H

#include <vector>
#include <set>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <queue>


struct RectRoi {
  int r, c, nr, nc;
};

class ContourView {
  // property:
  int level;
  float h_min, h_max;
  RectRoi aabb; // axis aligned bounding box of the current contour
  float poi[2]; // a point belonging to this contour/slice

  // data
  int cell_cnt;
  float cell_pos_sum[2];
  float cell_pos_sq_sum[2];
  float cell_vol3_sum;  // or "weight" of the elevation mountain. Should we include volumns under the h_min?

  // statistical summary
  float pos_mean[2];
  float pos_cov[2][2];
  float maj_dir[2], min_dir[2]; // gaussian ellipsoid axes
  float vol3_mean;

  // Raw data (the pixels that belong to this Contour. Is is necessary?)
  // TODO

  // hierarchy
  ContourView *parent;
  std::vector<ContourView *> children;


public:
  // TODO: 0. build a contour from 3: pic, roi, height threshold. In manager.
  explicit ContourView() {};

  // TODO: 1. build children from a current contour

};


#endif //CONT2_CONTOUR_H
