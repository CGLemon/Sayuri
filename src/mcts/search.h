#include "mcts/time_control.h"
#include "mcts/parameters.h"
#include "mcts/node.h"

#include "neural/network.h"
#include "game/game_state.h"
#include "utils/threadpool.h"

#include <thread>
#include <memory>
#include <atomic>

class Search {
public:
    struct Result {
        int best_move;
    };

    Search(GameState &state, Network &network) : root_state_(state), network_(network) {
        Initialize();
    }
    ~Search();

    void Initialize();

    Result Think();

    int ThinkBestMove();

private:
    int threads_;

    int max_playouts_; 

    std::atomic<int> playouts_; 

    TimeControl time_control_;

    GameState &root_state_;

    Network &network_;

    std::shared_ptr<Node> root_node_; 

    std::shared_ptr<Parameters> param_;

    std::unique_ptr<ThreadGroup<void>> group_;
};
