#include "neural/blas/blas_forward_pipe.h"
#include "neural/network_basic.h"
#include "neural/description.h"
#include "game/game_state.h"
#include "utils/cache.h"

#include <memory>

class Network {
public:
    enum Ensemble {
        NONE, DIRECT, RANDOM_SYMMETRY
    };

    using Result = OutputResult;
    using Cache = LruCache<Result>;
    
    void Initialize(const std::string &weights);


    Result GetOutput(const GameState &state,
                     const Ensemble ensemble,
                     int symmetry = -1);

private:
    bool ProbeCache(const GameState &state, Result &result);

    Result GetOutputInternal(const GameState &state, const bool symmetry);

    std::unique_ptr<NetworkForwardPipe> pipe_{nullptr};
    Cache nn_cache_;
};
