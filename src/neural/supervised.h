#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <queue>
#include <mutex>
#include <atomic>

#include "neural/training.h"

class Supervised {
public:
    static Supervised &Get();

    // Parse the SGF files and generate the training data.
    void FromSgfs(bool general,
                  std::string sgf_name,
                  std::string out_name_prefix);

private:
    // Will save general alpha-zero data files since we can 
    // parse any type SGF file. But forbidding the final score 
    // and the ownership informations. They are filled zeros.
    bool GeneralSgfProcess(std::string &sgfstring,
                           std::vector<Training> &chunk) const;
    // Will save all types data. The SGF game must be finished 
    // if we use this function.
    bool SgfProcess(std::string &sgfstring,
                    std::vector<Training> &chunk) const;

    bool SaveChunk(std::string out_name_prefix,
                   std::vector<Training> &chunk);

    std::queue<std::string> tasks_;
    std::mutex mtx_;

    std::atomic<int> tot_games_;
    std::atomic<int> file_cnt_;
    std::atomic<int> worker_cnt_;
    std::atomic<int> running_threads_;
    std::atomic<bool> running_;
};
