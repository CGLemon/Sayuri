#pragma once

#include "neural/description.h"
#include "game/types.h"
#include <array>
#include <memory>

static constexpr int kInputChannels = 38; // 8 past moves * 3 
                                          // 10 binary features
                                          // 4 misc features
static constexpr int kOuputValueMisc = 5;
static constexpr int kOuputPassProbability = 1;
static constexpr int kOuputProbabilitiesChannels = 1;
static constexpr int kOuputOwnershipChannels = 1;

static constexpr int kInputChannels_V3 = 43; // 8 past moves * 3 
                                             // 13 binary features
                                             // 6 misc features

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
                     wdl_winrate(0.f),
                     stm_winrate(0.f),
                     final_score(0.f) {
        wdl.fill(0.0f);
        probabilities.fill(0.f);
        ownership.fill(0.f);
        fp16 = false;
    }

    bool fp16;
    int board_size;
    float komi;

    float pass_probability;
    float wdl_winrate;
    float stm_winrate;
    float final_score;

    std::array<float, 3> wdl;
    std::array<float, kNumIntersections> probabilities;
    std::array<float, kNumIntersections> ownership;
};

class NetworkForwardPipe {
public:
    virtual void Initialize(std::shared_ptr<DNNWeights> weights) = 0;

    virtual OutputResult Forward(const InputData &inpnt) = 0;

    virtual bool Valid() = 0;

    virtual void Load(std::shared_ptr<DNNWeights> weights) = 0;

    virtual void Reload(int) = 0;

    virtual void Release() = 0;

    virtual void Destroy() = 0;
};
