//
// Created by lewis on 2/2/23.
//

#ifndef CONT2_CONFIG_HANDLER_H
#define CONT2_CONFIG_HANDLER_H

#include <vector>
#include <iostream>
#include <glog/logging.h>
#include <opencv2/core.hpp>

// load and print loaded config
template<typename T>
void loadOneConfig(cv::FileStorage &fs, const std::vector<std::string> &keys, T &container) {
  // assume no list/sequence in the middle
  CHECK(!keys.empty());
  cv::FileNode fn;
  for (int i = 0; i < keys.size(); i++) {
    fn = i ? fn[keys[i]] : fs[keys[0]];
    std::cout << "\"" << keys[i] << "\"->";
  }
  if (fn.isNone()) {
    printf(": [!] Cannot find the specified config parameter!\n");
    return;
  }
  container = (T) fn;
  std::cout << ": " << container << std::endl;

}

template<typename T>
void loadSeqConfig(cv::FileStorage &fs, const std::vector<std::string> &keys, std::vector<T> &container) {
  // assume no list/sequence in the middle
  CHECK(!keys.empty());
  cv::FileNode fn;
  for (int i = 0; i < keys.size(); i++) {
    fn = i ? fn[keys[i]] : fs[keys[0]];
    std::cout << "\"" << keys[i] << "\"->";
  }
  if (fn.isNone()) {
    printf(": [!] Cannot find the specified config parameter!\n");
    return;
  }

  CHECK_EQ(fn.type(), cv::FileNode::SEQ);
  container.clear();
  cv::FileNodeIterator it = fn.begin(), it_end = fn.end(); // Go through the node
  for (; it != it_end; ++it)
    container.emplace_back((T) (*it));

  std::cout << ": ";
  for (const auto &dat: container) {
    std::cout << dat << ", ";
  }
  std::cout << std::endl;

}


#endif //CONT2_CONFIG_HANDLER_H
