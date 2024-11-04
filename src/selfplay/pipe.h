#pragma once

#include "selfplay/engine.h"
#include "neural/training_data.h"

#include <atomic>
#include <vector>
#include <thread>
#include <string>
#include <memory>

class SelfPlayPipe {
public:
    SelfPlayPipe();

private:
    using GameTrainingData = std::vector<TrainingData>;

    void Initialize();
    void CreateWorkspace();
    void Loop();

    bool SaveChunk(const int out_id,
                   float vdata_prob,
                   std::vector<TrainingData> &chunk);
    bool SaveNetQueries(int games, std::string net_queries);
    bool GatherChunkFromBuffer(int games, std::vector<TrainingData> &chunk);
    int FancyCeil(int val, int step) const;

    std::mutex data_mutex_;
    std::mutex log_mutex_;

    std::vector<std::shared_ptr<GameTrainingData>> game_chunk_buffer_;
    std::atomic<int> accumulation_games_;
    std::atomic<int> played_games_;
    std::atomic<int> running_threads_;

    int num_saved_chunks_;
    int max_games_;
    Engine engine_;

    std::string target_directory_;
    std::string sgf_directory_;
    std::string queies_directory_;
    std::string tdata_directory_;
    std::string tdata_directory_hash_;
    std::string vdata_directory_;
    std::string vdata_directory_hash_;
    std::string filename_hash_;

    std::vector<std::thread> workers_;
};
