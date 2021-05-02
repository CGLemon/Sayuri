#pragma once

#include "neural/description.h"
#include "game/types.h"
#include <array>
#include <memory>

static constexpr int kInputChannels = 36; // 8 * 3 + 12
static constexpr int kOuputValueMisc = 5;
static constexpr int kOuputPassProbability = 1;

struct InputData {
    InputData() : komi(0.f), board_size(-1), side_to_move(kInvalid) {
        planes.fill(0.f);
    };

    float komi;
    int board_size;
    int side_to_move;

    std::array<float, kInputChannels * kNumIntersections> planes;
};

struct OutputResult {
    OutputResult() : board_size(-1),
                     pass_probability(0.f),
                     winrate(0.f),
                     final_score(0.f), 
                     score_width(0.f) {
        wdl.fill(0.0f);
        probabilities.fill(0.f);
        ownership.fill(0.f);
    }

    int board_size;

    float pass_probability;
    float winrate;

    float final_score;
    float score_width;

    std::array<float, 3> wdl;
    std::array<float, kNumIntersections> probabilities;
    std::array<float, kNumIntersections> ownership;
};

class NetworkForwardPipe {
public:
    virtual void Initialize(std::shared_ptr<DNNWeights> weights) = 0;

    virtual OutputResult Forward(const InputData &inpnt) = 0;

    virtual bool Valid() = 0;
};



