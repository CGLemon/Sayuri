#include "mcts/rollout.h"
#include "game/types.h"

#include <cmath>

float GetRolloutWinrate(GameState &state,
                            const int num_sims,
                            const int color,
                            std::vector<float> &mcowner,
                            float &black_score) {
    black_score = 0;

    int black_wins_cnt = 0;
    int white_wins_cnt = 0;
    int num_games = 0;

    for (int s = 0; s < num_sims; ++s) {
        auto fork_state = state;
        int curr_move = 0;

        while (fork_state.GetPasses() < 2 && curr_move++ < 150) {
            fork_state.PlayRandomMove(true);
        }
        while (fork_state.GetPasses() < 2 && curr_move++ < 999) {
            fork_state.PlayRandomMove(false);
        }

        float black_acc_score = 0.f;

        std::vector<int> ownership = fork_state.GetOwnership();
        {
            for (int idx = 0; idx < fork_state.GetNumIntersections(); ++idx) {
                int owner = ownership[idx];
                float mcval = 0.f;

                if (owner == kBlack) {
                    mcval = 1.f;
                } else if (owner == kWhite) {
                    mcval = -1.f;
                }
                black_acc_score += mcval;
                mcowner[idx] += mcval;
            }
        }

        black_acc_score -= fork_state.GetKomi();

        num_games++;
        if (black_acc_score > 1e-4f) {
            black_wins_cnt++;
        } else if (black_acc_score < -1e-4f) {
            white_wins_cnt++;
        }
        black_score += black_acc_score;
    }

    if (num_games == 0) {
        return 0.5f;
    }

    float winrate = (float)black_wins_cnt/(float)num_games;
    black_score /= num_games;

    for (int idx = 0; idx < state.GetNumIntersections(); ++idx) {
        mcowner[idx] /= (float)num_games;
    }

    if (color == kWhite) {
        return 1.f - winrate;
    }
    return winrate;
}
