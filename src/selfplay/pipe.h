#pragma once

#include "selfplay/engine.h"
#include "neural/training_data.h"
#include "utils/threadpool.h"

#include <atomic>
#include <vector>
#include <thread>
#include <string>
#include <memory>
#include <utility>
#include <list>

class SelfPlayPipe {
public:
    SelfPlayPipe();

private:
    using GameTrainingData = std::vector<TrainingData>;
    using DataSgfPair = std::pair<GameTrainingData, std::string>;
    using GamesQueriesPair = std::pair<int, std::string>;

    void Initialize();
    void CreateWorkspace();
    void Loop();
    void Finish();

    bool SaveSgf(std::string &sgfstring);
    bool SaveChunk(const int out_id,
                   float vdata_prob,
                   std::vector<TrainingData> &chunk);
    bool SaveNetQueries(int games, std::string net_queries);
    bool GatherChunkFromBuffer(int games, std::vector<TrainingData> &chunk);
    int FancyCeil(int val, int step) const;

    void AssignDataWorker();
    void AssignSelfplayWorkers();

    std::mutex data_mutex_;
    std::mutex log_mutex_;
    std::list<std::shared_ptr<DataSgfPair>> data_sgf_buffer_;
    std::list<GamesQueriesPair> games_queries_buffer_;

    std::atomic<int> accumulation_games_;
    std::atomic<int> played_games_;
    std::atomic<int> running_threads_;
    std::atomic<bool> writing_worker_running_;

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

    std::unique_ptr<ThreadGroup<void>> group_;
};
