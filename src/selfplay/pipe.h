#pragma once

#include "selfplay/engine.h"

#include <vector>
#include <thread>
#include <string>
#include <atomic>

class SelfPlayPipe {
public:
    SelfPlayPipe();

private:
    void Initialize();
    void Loop();

    std::mutex log_mutex_;

    std::atomic<int> accmulate_games_;
    std::atomic<int> played_games_;
    int max_games_;
    Engine engine_;

    std::string target_directory_;
    std::string sgf_filename_;
    std::string data_filename_;
    std::string filename_hash_;

    std::vector<std::thread> workers_;
};
