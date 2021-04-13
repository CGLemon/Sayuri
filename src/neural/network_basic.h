#pragma once

#include "game/types.h"
#include <array>

static constexpr size_t kInputChannels = 18;

struct InputData {
    std::array<float, kNumIntersections * kInputChannels> planes;

    std::uint64_t mode{0ULL};
};

struct OutputResult {
    float winrate;
    float final_score;
    float score_width;
    float pass_probability;

    std::array<float, kNumIntersections> probabilities;
    std::array<float, kNumIntersections> ownership; 

    std::uint64_t mode{0ULL};
};

class Compution {
public:
    virtual void Compute();

    virtual std::uint64_t AddInputs(InputData &inpnt);

    virtual OutputResult GetResult(std::uint64_t hash);

    virtual void Evict(std::uint64_t hash);
};



