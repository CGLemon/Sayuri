#include "neural/blas/blas_forward_pipe.h"
#include "neural/network_basic.h"
#include "neural/description.h"
#include "game/game_state.h"
#include "utils/cache.h"

#include <memory>
#include <array>
#include <algorithm>
#include <cmath>

class Network {
public:
    enum Ensemble {
        NONE, DIRECT, RANDOM_SYMMETRY
    };

    using Inputs = InputData;
    using Result = OutputResult;
    using Cache = LruCache<Result>;
    
    void Initialize(const std::string &weights);


    Result GetOutput(const GameState &state,
                     const Ensemble ensemble,
                     int symmetry = -1);

private:
    bool ProbeCache(const GameState &state, Result &result);

    Result GetOutputInternal(const GameState &state, const bool symmetry);

    Network::Result DummyForward(const Network::Inputs& inputs) const;
/*
    template<size_t N>
    void ApplySoftmax(const std::array<float, N> &input,
                      const size_t range,
                      const float temperature = 1.0f) const;

    template<size_t N>
    void ApplyTanh(const std::array<float, N> &input, const size_t range) const;
 */

    void ProcessResult(Network::Result &result);

    std::unique_ptr<NetworkForwardPipe> pipe_{nullptr};
    Cache nn_cache_;
};

/*
template<size_t N>
void Network::ApplySoftmax(const std::array<float, N> &input,
                           const size_t range,
                           const float temperature) const {


    const auto alpha = *std::max_element(std::begin(input),
                                         std::begin(input) + range);
    auto denom = 0.0f;

    for (auto idx = size_t{0}; idx < range; ++idx) {
        denom += std::exp((input[idx] - alpha) / temperature);
    }

    for (auto idx = size_t{0}; idx < range; ++idx) {
        input[idx] = input[idx] / denom;
    }
}

template<size_t N>
void Network::ApplyTanh(const std::array<float, N> &input, const size_t range) const {
    for (auto idx = size_t{0}; idx < range; ++idx) {
        input[idx] = std::tanh(input[idx]);
    }
}
 */
