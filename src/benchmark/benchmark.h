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
    void Run();

    struct Query {
        int batch_size;
        int threads;
        float timelimit;
    };

    std::vector<Query> quries_list_;
    std::unique_ptr<Benchmark::Agent> agent_;
};
