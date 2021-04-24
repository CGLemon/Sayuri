#pragma once

#include "neural/description.h"
#include "game/types.h"
#include <array>
#include <vector>

static constexpr int kInputChannels = 36; // 8 * 3 + 12
static constexpr int kOuputValueMisc = 3;
static constexpr int kOuputPassProbability = 1;

struct InputData {
    int board_size{-1};

    std::array<float, kNumIntersections * kInputChannels> planes;

    int side_to_move{kInvalid};
};

struct OutputResult {
    int board_size{-1};

    float winrate;
    float final_score;
    float score_width;
    float pass_probability;

    std::array<float, kNumIntersections> probabilities;
    std::array<float, kNumIntersections> ownership;
};

class NetworkForwardPipe {
public:
    virtual void Initialize(std::shared_ptr<DNNWeights> weights);

    virtual OutputResult Forward(const InputData &inpnt);

    virtual bool Valid();
};



