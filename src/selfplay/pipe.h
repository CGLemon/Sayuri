#pragma once

#include "selfplay/engine.h"
#include "neural/training_data.h"

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

    bool SaveChunk(const int out_id,
                   std::vector<TrainingData> &chunk);

    bool SaveNetQueries(const size_t queries);

    std::mutex data_mutex_;
    std::mutex log_mutex_;

    std::vector<TrainingData> chunk_;

    std::atomic<int> accmulate_games_;
    std::atomic<int> played_games_;
    std::atomic<int> running_threads_;

    int chunk_games_;
    int max_games_;
    Engine engine_;

    std::string target_directory_;
    std::string sgf_directory_;
    std::string data_directory_;
    std::string data_directory_hash_;
    std::string filename_hash_;

    std::vector<std::thread> workers_;
};
