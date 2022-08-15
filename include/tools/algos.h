//
// Created by lewis on 5/18/22.
//

#ifndef CONT2_ALGOS_H
#define CONT2_ALGOS_H

#include <algorithm>
#include <Eigen/Core>
#include <Eigen/Geometry>

template<typename T>
inline bool diff_perc(const T &num1, const T &num2, const T &perc) {
  return std::abs((num1 - num2) / std::max(num1, num2)) > perc;
}

template<typename T>
inline bool diff_delt(const T &num1, const T &num2, const T &delta) {
  return std::abs(num1 - num2) > delta;
}

/// p_{tgt} = T_{delta} * p_{src}
/// can be solved using umeyama's when points are more.
/// \param s1
/// \param s2
/// \param t1
/// \param t2
/// \return
template<typename T>
inline Eigen::Transform<T, 2, Eigen::Isometry>
estimateTF(const Eigen::Matrix<T, 2, 1> &s1, const Eigen::Matrix<T, 2, 1> &s2,
           const Eigen::Matrix<T, 2, 1> &t1, const Eigen::Matrix<T, 2, 1> &t2) {
  Eigen::Matrix<T, 2, 1> vs = s2 - s1;
  Eigen::Matrix<T, 2, 1> vt = t2 - t1;

  T ang = std::atan2(vs.x() * vt.y() - vs.y() * vt.x(), vs.dot(vt));

  Eigen::Transform<T, 2, Eigen::Isometry> res = Eigen::Transform<T, 2, Eigen::Isometry>::Identity();
  res.rotate(ang);
  Eigen::Matrix<T, 2, 1> trans = T(0.5) * (t1 + t2 - res.rotation() * (s1 + s2));
  res.pretranslate(trans);
  return res; // T_{delta}
}

template<typename T>
inline void clampAng(T &ang) {
  ang = ang - std::floor((ang + M_PI) / (2 * M_PI)) * 2 * M_PI;
}

template<typename T>
inline T gaussPDF(const T &x, const T &mean, const T &sd) {
  return std::exp(-0.5 * ((x - mean) / sd) * ((x - mean) / sd)) / std::sqrt(2 * M_PI * sd * sd);
}

template<typename T>
inline std::pair<int, T> search_vec(const std::vector<std::pair<int, T>> &arr, int p1, int p2, const int &tgt) {
  if (p2 < p1)
    return {-1, T()};
  int mid = (p1 + p2) / 2;
  if (arr[mid].first == tgt)
    return arr[mid];
  else if (arr[mid].first < tgt)
    return search_vec<T>(arr, mid + 1, p2, tgt);
  return search_vec<T>(arr, p1, mid - 1, tgt);
}

/// Align var[i] to at least the lower bound bar[i]
/// \tparam T
/// \tparam N
/// \param bar
/// \param var
template<typename T, int N>
void alignLB(const T bar[N], T var[N]) {
  for (int i = 0; i < N; i++)
    var[i] = var[i] < bar[i] ? bar[i] : var[i];
}

template<typename T>
void alignLB(const T &bar, T &var) {  // stricter type constraint
  for (int i = 0; i < T::SizeAtCompileTime; i++)
    var.data[i] = var.data[i] < bar.data[i] ? bar.data[i] : var.data[i];
}

///// Align var[i] to at least the lower bound `bar`
//template<typename T, int N>
//void alignLB(const T &bar, T var[N]) {
//  for (int i = 0; i < N; i++)
//    var[i] = var[i] < bar ? bar : var[i];
//}

/// Align var[i] to at most the upper bound bar[i]
template<typename T, int N>
void alignUB(const T bar[N], T var[N]) {
  for (int i = 0; i < N; i++)
    var[i] = var[i] > bar[i] ? bar[i] : var[i];
}

template<typename T>
void alignUB(const T &bar, T &var) {
  for (int i = 0; i < T::SizeAtCompileTime; i++)
    var.data[i] = var.data[i] > bar.data[i] ? bar.data[i] : var.data[i];
}

///// Align var[i] to at most the upper bound `bar`
//template<typename T, int N>
//void alignUB(const T &bar, T var[N]) {
//  for (int i = 0; i < N; i++)
//    var[i] = var[i] > bar ? bar : var[i];
//}

#endif //CONT2_ALGOS_H
