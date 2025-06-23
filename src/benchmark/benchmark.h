#pragma once

#include "game/gtp.h"
#include <vector>
#include <memory>

class Benchmark {
public:
    using Agent = GtpLoop::Agent;

    Benchmark() {
        agent_ = std::make_unique<Benchmark::Agent>();
        agent_->Apply();
        Initialize();
        Run();
    }
    ~Benchmark() {
        agent_->Quit();
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
    std::unique_ptr<Benchmark::Agent> agent_;
};
