#pragma once

#include <memory>
#include <vector>

#include "game/agents.h"

class Benchmark {
public:
    Benchmark() {
        agent_ = std::make_unique<GptAgent>();
        Initialize();
        Run();
    }
    ~Benchmark() {
        agent_->Shutdown();
    }

private:
    void Initialize();
    std::vector<std::string> GenerateTestSet(int num_games);
    void Run();

    struct Query {
        int threads;
        int batch_size;
        int games;
    };

    std::vector<Query> queries_list_;
    std::unique_ptr<GptAgent> agent_{nullptr};
};
