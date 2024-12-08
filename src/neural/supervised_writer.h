#pragma once

#include "game/game_state.h"
#include "neural/network.h"
#include "neural/training_data.h"
#include "config.h"

#include <atomic>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <thread>
#include <list>

class SupervisedWriter {
public:
    SupervisedWriter();

private:
    void Initialize();
    void Loop();
    void AssignWorkers();
    void WaitForWorkers();
    std::string TryGetSgfFilename();
    void PushSgfString(std::vector<std::string> &sgfs);
    std::string TryGetSgfString();
    bool SaveChunk(const int out_id,
                   std::vector<TrainingData> &chunk);
    void GenenrateDataAndWrite(std::string &sgfstring);
    void AssignNetworkResult(TrainingData &data, GameState &state);

    std::atomic<bool> reader_done_{false};
    std::atomic<int> num_sgf_games_{0};

    std::unique_ptr<Network> network_{nullptr};
    std::list<std::string> sgf_files_queue_;
    std::list<std::string> sgf_strings_queue_;
    std::vector<std::thread> workers_;

    int threads_;
    std::string input_sgf_directory_;
    std::string target_directory_;
    std::mutex sgf_file_mutex_;
    std::mutex sgf_string_mutex_;
};
