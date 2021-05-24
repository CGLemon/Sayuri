#include "neural/network_basic.h"
#include "neural/description.h"
#include "game/game_state.h"
#include "utils/cache.h"

#include <memory>
#include <array>
#include <algorithm>
#include <cmath>
#include <string>

class Network {
public:
    enum Ensemble {
        NONE, DIRECT, RANDOM_SYMMETRY
    };

    using Inputs = InputData;
    using Result = OutputResult;
    using Cache = LruCache<Result>;
    
    void Initialize(const std::string &weights);
    void Destroy();

    Result GetOutput(const GameState &state,
                     const Ensemble ensemble,
                     int symmetry = -1,
                     const bool read_cache = true,
                     const bool write_cache = true);

    std::string GetOutputString(const GameState &state,
                                const Ensemble ensemble,
                                int symmetry = -1);

    void Reload(int board_size);

private:
    bool ProbeCache(const GameState &state, Result &result);

    Result GetOutputInternal(const GameState &state, const bool symmetry);

    Network::Result DummyForward(const Network::Inputs& inputs) const;

    std::unique_ptr<NetworkForwardPipe> pipe_{nullptr};
    Cache nn_cache_;
};

