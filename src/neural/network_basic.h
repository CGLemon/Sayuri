#pragma once

#include "neural/description.h"
#include "game/types.h"
#include <array>
#include <memory>

static constexpr int kInputChannels = 43; // 8 past moves * 3
                                          // 13 binary features
                                          // 6 misc features
static constexpr int kOuputValueMisc = 15;
static constexpr int kOuputPassProbability = 5;
static constexpr int kOuputProbabilitiesChannels = 5;
static constexpr int kOuputOwnershipChannels = 1;

enum class PolicyBufferOffset : int {
    kNormal = 0,
    kOpponent = 1,
    kSoft = 2,
    kSoftOpponent = 3,
    kOptimistic = 4,
    kDefault
};

struct InputData {
    InputData() {
        planes.fill(0.f);
    };

    float komi{0.f};
    int board_size{-1};
    int side_to_move{kInvalid};
    int symmetry{-1};
    PolicyBufferOffset offset{PolicyBufferOffset::kDefault};

    std::array<float, kInputChannels * kNumIntersections> planes;
};

struct OutputResult {
    OutputResult() {
        wdl.fill(0.0f);
        probabilities.fill(0.f);
        ownership.fill(0.f);
    }
    void ImportQueryInfo(OutputResult &other) {
        fp16 = other.fp16;
        board_size = other.board_size;
        komi = other.komi;
    }

    bool fp16{false};
    int board_size{-1};
    float komi{0.f};

    float pass_probability{0.f};
    float wdl_winrate{0.f};
    float stm_winrate{0.f};
    float final_score{0.f};
    float q_error{0.f};
    float score_error{0.f};
    PolicyBufferOffset offset{PolicyBufferOffset::kDefault};

    std::array<float, 3> wdl;
    std::array<float, kNumIntersections> probabilities;
    std::array<float, kNumIntersections> ownership;
};

struct ForwardQuery {
    ForwardQuery() = default;

    static ForwardQuery SetTemperature(float temp) {
        ForwardQuery q;
        q.temperature = temp;
        return q;
    }
    static ForwardQuery SetSymmetry(int symm) {
        ForwardQuery q;
        q.symmetry = symm;
        return q;
    }
    static ForwardQuery SetCache(bool use_cache) {
        ForwardQuery q;
        q.read_cache = use_cache;
        q.write_cache = use_cache;
        return q;
    }

    float temperature{1.0f};
    int symmetry{-1};
    bool read_cache{true};
    bool write_cache{true};
    PolicyBufferOffset offset{PolicyBufferOffset::kDefault};
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
