#include "mcts/rollout.h"
#include "game/types.h"

float GetBlackRolloutResult(GameState &state,
                            float *mcowner,
                            float &black_score) {
    auto fork_state = state;
    const int num_intersections = fork_state.GetNumIntersections();
    int num_curr_moves = 0;
    const int max_move_len = 2*num_intersections+1;

    // Do the random moves until the game is ending.
    while (fork_state.GetPasses() < 2 &&
               num_curr_moves < max_move_len) {
        fork_state.PlayRandomMove();
        num_curr_moves += 1;
    }

    // Compute the final position reuslt.
    black_score = 0;
    std::vector<int> ownership = fork_state.GetOwnership();
    {
        for (int idx = 0; idx < num_intersections; ++idx) {
            int owner = ownership[idx];
            float mcval = 0.f;

            if (owner == kBlack) {
                mcval = 1.f; // black value
            } else if (owner == kWhite) {
                mcval = -1.f; // white value
            }
            black_score += mcval;
            mcowner[idx] = mcval;
        }
    }

    black_score -= fork_state.GetKomi();
    float black_result = 0.5f; // draw

    if (black_score > 1e-4f) {
        black_result = 1; // black won
    } else if (black_score < -1e-4f) {
        black_result = 0; // white won
    }
    return black_result;
}
