#pragma once

#include "game/game_state.h"
#include "neural/network.h"
#include "neural/training_data.h"
#include "config.h"

#include <string>
#include <memory>
#include <thread>
#include <mutex>

class SupervisedWriter {
public:
    SupervisedWriter();

private:
    void Initialize();
    void Loop();
    void AssignWorkers();
    void WaitForWorkers();
    std::string TryGetSgfFilename();

    std::unique_ptr<Network> network_{nullptr};
    std::vector<std::string> sgf_files_;
    std::vector<std::thread> workers_;

    std::string input_sgf_directory_;
    std::mutex sgf_mutex_;
};
