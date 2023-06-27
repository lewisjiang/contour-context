//
// Created by lewis on 5/5/22.
//

#ifndef CONT2_BM_UTIL_H
#define CONT2_BM_UTIL_H

//
// Created by lewis on 10/21/21.
// Benchmark utility
//
#include <chrono>
#include <algorithm>
#include <string>
#include <numeric>
#include <map>
#include <utility>

class TicToc {
public:
  TicToc() {
    tic();
  }

  void tic() {
    start = std::chrono::steady_clock::now();
  }

  double toc() {
    end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    return elapsed_seconds.count();
  }

  double toctic() {
    double ret = toc();
    tic();
    return ret;
  }

private:
  std::chrono::time_point<std::chrono::steady_clock> start, end;
};

class SequentialTimeProfiler {
protected:
  struct OneLog {
    int idx{}, cnt{};
    double samps{};
    double autocorrs{};

    OneLog() = default;

    explicit OneLog(int i, int a, double b) : idx(i), cnt(a), samps(b), autocorrs(b * b) {}
  };

  TicToc clk;
  std::map<std::string, OneLog> logs;
  int cnt_loops = 0;
  size_t max_len = 5;   // min name length
  std::string desc = "";  // short description
public:

  SequentialTimeProfiler() = default;

  SequentialTimeProfiler(const std::string &name) : desc(name) {};

  inline std::string getDesc() const {
    return desc;
  }

  static std::string getTimeString() {
    std::time_t now = std::time(nullptr);
    struct tm tstruct = *std::localtime(&now);
    char buf[80];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %a %X %z", &tstruct);
    return buf;
  }

  inline void start() {
    clk.tic();
  }

  /// record and reset timer
  /// \param name The name of a log entry
  void record(const std::string &name) {
    const double dt = clk.toc();
    auto it = logs.find(name);
    if (it == logs.end()) {
      logs[name] = OneLog(static_cast<int>(logs.size()), 1, dt);
      max_len = std::max(max_len, name.length());
    } else {
      it->second.cnt++;
      it->second.samps += dt;
      it->second.autocorrs += dt * dt;
    }
    clk.tic();  // auto reset, useful for sequential timing.
  }

  /// record and reset timer
  /// \param name The name of a log entry
  void record(const std::string &name, double &dt_curr) {
    const double dt = clk.toc();
    auto it = logs.find(name);
    if (it == logs.end()) {
      logs[name] = OneLog(static_cast<int>(logs.size()), 1, dt);
      max_len = std::max(max_len, name.length());
    } else {
      it->second.cnt++;
      it->second.samps += dt;
      it->second.autocorrs += dt * dt;
    }
    dt_curr = dt;
    clk.tic();  // auto reset, useful for sequential timing.
  }

  inline void lap() {
    cnt_loops++;
  }

  void printScreen(bool sort_by_cost = false) const {
    printf("\n=== Time Profiling @%s ===\n", getTimeString().c_str());
    printf("=== Description: %s\n", desc.c_str());
    printf("%5s %s %10s %10s %10s %10s %10s %10s\n",
           "Index", (std::string(max_len - 4, ' ') + "Name").c_str(),
           "Count", "Average", "Stddev", "Per loop", "Loop %", "Accum %");
    std::vector<std::pair<std::string, OneLog>> vec(logs.begin(), logs.end());
    if (sort_by_cost)
      std::sort(vec.begin(), vec.end(),
                [&](const std::pair<std::string, OneLog> &e1, const std::pair<std::string, OneLog> &e2) {
                  return e1.second.samps > e2.second.samps;
                });
    else
      std::sort(vec.begin(), vec.end(),
                [&](const std::pair<std::string, OneLog> &e1, const std::pair<std::string, OneLog> &e2) {
                  return e1.second.idx < e2.second.idx;
                });

    double t_total = 0, t_accum = 0;
    for (const auto &itm: vec)
      t_total += itm.second.samps;

    for (const auto &itm: vec) {
      const auto &lg = itm.second;
      double x_bar = lg.samps / lg.cnt;
      double stddev = 0;
      if (lg.cnt > 1)
        stddev = std::sqrt(1.0 / (lg.cnt - 1) * (lg.autocorrs + lg.cnt * x_bar * x_bar - 2 * x_bar * lg.samps));
      t_accum += lg.samps;
      printf("%5d %s %10d %10.2e %10.2e %10.2e %10.2f %10.2f\n",
             lg.idx,
             (std::string(max_len - itm.first.length(), ' ') + itm.first).c_str(),
             lg.cnt,
             x_bar,
             stddev,
             cnt_loops > 0 ? lg.samps / cnt_loops : 0,
             lg.samps / t_total * 100, // count_i * avg_i / (\sum(count*avg))
             t_accum / t_total * 100
      );
    }
    printf("%5s %s %10d %10s %10s %10.2e %10s %10s\n",
           "*", (std::string(max_len - 4, ' ') + "*sum").c_str(), cnt_loops, "*", "*",
           cnt_loops > 0 ? t_total / cnt_loops : 0,
           "*", "*"
    );
  }

  void printFile(const std::string &fpath, bool sort_by_cost = false) const {
    std::FILE *fp;
    fp = std::fopen(fpath.c_str(), "a");

    fprintf(fp, "\n=== Time Profiling @%s ===\n", getTimeString().c_str());
    fprintf(fp, "=== Description: %s\n", desc.c_str());
    fprintf(fp, "%5s %s %10s %10s %10s %10s %10s %10s\n",
            "Index", (std::string(max_len - 4, ' ') + "Name").c_str(),
            "Count", "Average", "Stddev", "Per loop", "Loop %", "Accum %");
    std::vector<std::pair<std::string, OneLog>> vec(logs.begin(), logs.end());
    if (sort_by_cost)
      std::sort(vec.begin(), vec.end(),
                [&](const std::pair<std::string, OneLog> &e1, const std::pair<std::string, OneLog> &e2) {
                  return e1.second.samps > e2.second.samps;
                });
    else
      std::sort(vec.begin(), vec.end(),
                [&](const std::pair<std::string, OneLog> &e1, const std::pair<std::string, OneLog> &e2) {
                  return e1.second.idx < e2.second.idx;
                });

    double t_total = 0, t_accum = 0;
    for (const auto &itm: vec)
      t_total += itm.second.samps;

    for (const auto &itm: vec) {
      const auto &lg = itm.second;
      double x_bar = lg.samps / lg.cnt;
      double stddev = 0;
      if (lg.cnt > 1)
        stddev = std::sqrt(1.0 / (lg.cnt - 1) * (lg.autocorrs + lg.cnt * x_bar * x_bar - 2 * x_bar * lg.samps));
      t_accum += lg.samps;
      fprintf(fp, "%5d %s %10d %10.2e %10.2e %10.2e %10.2f %10.2f\n",
              lg.idx,
              (std::string(max_len - itm.first.length(), ' ') + itm.first).c_str(),
              lg.cnt,
              x_bar,
              stddev,
              cnt_loops > 0 ? lg.samps / cnt_loops : 0,
              lg.samps / t_total * 100, // count_i * avg_i / (\sum(count*avg))
              t_accum / t_total * 100
      );
    }
    fprintf(fp, "%5s %s %10d %10s %10s %10.2e %10s %10s\n",
            "*", (std::string(max_len - 4, ' ') + "*sum").c_str(), cnt_loops, "*", "*",
            cnt_loops > 0 ? t_total / cnt_loops : 0,
            "*", "*"
    );
    std::fclose(fp);
  }
};

#endif //CONT2_BM_UTIL_H
