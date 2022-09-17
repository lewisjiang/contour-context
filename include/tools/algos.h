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

///
/// \tparam T
/// \param ang to [-pi, pi)?]?
template<typename T>
inline void clampAng(T &ang) {
  ang = ang - std::floor((ang + M_PI) / (2 * M_PI)) * 2 * M_PI;
}

template<typename T>
inline T gaussPDF(const T &x, const T &mean, const T &sd) {
  return std::exp(-0.5 * ((x - mean) / sd) * ((x - mean) / sd)) / std::sqrt(2 * M_PI * sd * sd);
}

template<typename T>
std::pair<int, T> search_vec(const std::vector<std::pair<int, T>> &arr, int p1, int p2, const int &tgt) {
  if (p2 < p1)
    return {-1, T()};
  int mid = (p1 + p2) / 2;
  if (arr[mid].first == tgt)
    return arr[mid];
  else if (arr[mid].first < tgt)
    return search_vec<T>(arr, mid + 1, p2, tgt);
  return search_vec<T>(arr, p1, mid - 1, tgt);
}

/// Find the nearset neighbour in a sorted data vector
/// \tparam T
/// \param q_val query data
/// \param sorted_vec data vector
/// \param tol the maximal tolerable difference between the query and the NN.
/// \return the index of the nn in the vector. -1 for not found
template<typename T>
int lookupNN(const T &q_val, const std::vector<T> &sorted_vec, const T &tol) {
  auto it_low = std::lower_bound(sorted_vec.begin(), sorted_vec.end(), q_val);
  auto it = it_low;
  if (it_low == sorted_vec.begin()) {
    it = it_low;
  } else if (it_low == sorted_vec.end()) {
    it = it_low - 1;
  } else {
    it = std::abs(q_val - *it_low) < std::abs(q_val - *(it_low - 1)) ? it_low : it_low - 1;
  }
  if (sorted_vec.empty() || std::abs(*it - q_val) > tol)
    return -1;
  return it - sorted_vec.begin();
}

/// Apply a sort permutation to a vector so that the vector is "sorted" by the permutation.
/// \tparam T
/// \param p sort permutation, i.e. X[p[0]] <= X[p[1]] <= ... <= X[p[size-1]]
/// \param data The array that need to be permuted according to `p`. If `data` is X, we expect to operate on X inplace
/// such that X[0] <= X[1] <= ... <= X[size-1]
template<typename T>
void apply_sort_permutation(const std::vector<int> &p, std::vector<T> &data) {
  DCHECK_EQ(p.size(), data.size());
  std::vector<bool> used(p.size(), false);
  int i0 = 0, i = 0;

  while (i0 < data.size())
    if (used[i] || used[p[i]])
      i = i0++;
    else {
      std::swap(data[p[i]], data[i]);
      used[i] = true;
      i = p[i];
    }
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

inline bool file_exists(const std::string &name) {
  if (FILE *file = fopen(name.c_str(), "r")) {
    fclose(file);
    return true;
  } else {
    return false;
  }
}

#endif //CONT2_ALGOS_H
