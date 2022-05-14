//
// Created by lewis on 5/13/22.
//

#ifndef CONT2_UTIL_H
#define CONT2_UTIL_H

union RangeLimits {
  float data[6];
  struct {
    float x_min;
    float x_max;
    float y_min;
    float y_max;
    float z_min;
    float z_max;
  };

  inline bool init(const float &xx, const float &yy, const float &zz) {
    x_min = x_max = xx;
    y_min = y_max = yy;
    z_min = z_max = zz;
  }

  inline void clear() {
    std::fill(data, data + 6, 0.0f);
  }

  inline float dx() const {
    return x_max - x_min;
  }

  inline float dy() const {
    return y_max - y_min;
  }

  inline float dz() const {
    return z_max - z_min;
  }

  inline float com_x() const {
    return 0.5f * (x_max + x_min);
  }

  inline float com_y() const {
    return 0.5f * (y_max + y_min);
  }

  inline float com_z() const {
    return 0.5f * (z_max + z_min);
  }

  inline bool contains(const PointTMeas &p) const {
//    return x_min < p.x && p.x < x_max && y_min < p.y && p.y < y_max && z_min < p.z && p.z < z_max;
    return x_min <= p.x && p.x < x_max && y_min <= p.y && p.y < y_max && z_min <= p.z && p.z < z_max;
  }

  inline bool contains(const RangeLimits &p) const {
    return x_min < p.x_min && p.x_max < x_max &&
           y_min < p.y_min && p.y_max < y_max &&
           z_min < p.z_min && p.z_max < z_max;
  }

};

#endif //CONT2_UTIL_H
