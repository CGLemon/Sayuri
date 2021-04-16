#pragma once

#include "neural/description.h"
#include "game/types.h"
#include <array>

static constexpr int kInputChannels = 18;
static constexpr int kOuputValueMisc = 3;
static constexpr int kOuputPassProbability = 1;

struct InputData {
    int board_size{-1};
    std::array<float, kNumIntersections * kInputChannels> planes;
    std::uint64_t mode{0ULL};
};

struct OutputResult {
    int board_size{-1};

    float winrate;
    float final_score;
    float score_width;
    float pass_probability;

    std::array<float, kNumIntersections> probabilities;
    std::array<float, kNumIntersections> ownership; 

    std::uint64_t mode{0ULL};
};

class NetworkForwardPipe {
public:
    virtual void Initialize(std::shared_ptr<DNNWeights> weights);

    virtual OutputResult Forward(const InputData &inpnt);

};



