#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <queue>
#include <mutex>
#include <atomic>

class Supervised {
public:
    static Supervised &Get();

    void FromSgfs(std::string sgf_name,
                      std::string out_name_prefix);

private:
    bool SgfProcess(std::string &sgfstring,
                        std::ostream &out_file) const;


    std::queue<std::string> tasks_;
    std::mutex mtx_;
    std::atomic<int> tot_games_;
    std::atomic<int> file_cnt_;
    std::atomic<int> worker_cnt_;
    std::atomic<bool> running_;

};
