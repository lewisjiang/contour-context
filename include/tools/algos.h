//
// Created by lewis on 5/18/22.
//

#ifndef CONT2_ALGOS_H
#define CONT2_ALGOS_H

#include <algorithm>
#include <Eigen/Core>
#include <Eigen/Geometry>

inline bool diff_perc(const double &num1, const double &num2, const double &perc) {
  return std::abs((num1 - num2) / std::max(num1, num2)) > perc;
}

inline bool diff_delt(const double &num1, const double &num2, const double &delta) {
  return std::abs(num1 - num2) > delta;
}

/// p_{tgt} = T_{delta} * p_{src}
/// can be solved using umeyama's when points are more.
/// \param s1
/// \param s2
/// \param t1
/// \param t2
/// \return
inline Eigen::Isometry2d estimateTF(const Eigen::Vector2d &s1, const Eigen::Vector2d &s2,
                                    const Eigen::Vector2d &t1, const Eigen::Vector2d &t2) {
  Eigen::Vector2d vs = s2 - s1;
  Eigen::Vector2d vt = t2 - t1;

  double ang = std::atan2(vs.x() * vt.y() - vs.y() * vt.x(), vs.dot(vt));

  Eigen::Isometry2d res = Eigen::Isometry2d::Identity();
  res.rotate(ang);
  Eigen::Vector2d trans = 0.5 * (t1 + t2 - res.rotation() * (s1 + s2));
  res.pretranslate(trans);
  return res; // T_{delta}
}

template<typename T>
inline void clampAng(T &ang) {
  ang = ang - std::floor((ang + M_PI) / (2 * M_PI)) * 2 * M_PI;
}

#endif //CONT2_ALGOS_H
