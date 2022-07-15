#include "mcts/rollout.h"
#include "game/types.h"

#include <mutex>
#include <cmath>
#include <iostream>
std::mutex mcowner_locker;

float GetRolloutWinrate(GameState &state,
                            const int num_sims,
                            const int color,
                            std::vector<float> &mcowner) {
    // TODO: The rollout performance is too slow. The PlayRandomMove()
    //       eats too much time.

    int black_wins_cnt = 0;
    int white_wins_cnt = 0;
    float moving_factor = 0.1f;

    int num_games = 0;
    float confidence = 0.f; 
    float confidence_cnt = 0.f;

    for (int s = 0; s < num_sims; ++s) {
        auto fork_state = state;
        int curr_move = 0;

        while (fork_state.GetPasses() < 2 && curr_move++ < 30) {
            fork_state.PlayRandomMove();
        }

        float black_acc_score = 0.f;

        std::vector<int> ownership = fork_state.GetOwnership();
        {
            std::lock_guard<std::mutex> lk(mcowner_locker);
            for (int idx = 0; idx < fork_state.GetNumIntersections(); ++idx) {
                int owner = ownership[idx];
                float mcval = mcowner[idx];

                if (owner == kBlack) {
                    mcval = (1.f-moving_factor) * mcval + moving_factor *  1.f;
                } else if (owner == kWhite) {
                    mcval = (1.f-moving_factor) * mcval + moving_factor * -1.f;
                } else {
                    mcval = (1.f-moving_factor) * mcval + moving_factor *  0.f;
                }
                black_acc_score += mcval;
                mcowner[idx] = mcval;

                confidence += ((std::exp(std::abs(mcval)) - 1) / 1.71828182846f); // 0 ~ 1
                confidence_cnt += 1;
            }
        }

        num_games++;
        if (black_acc_score > 1e-4f) {
            black_wins_cnt++;
        } else if (black_acc_score < -1e-4f) {
            white_wins_cnt++;
        }
    }

    if (num_games == 0) {
        return 0.5f;
    }
    confidence /= confidence_cnt;

    float winrate = (float)black_wins_cnt/(float)num_games;
    winrate = 0.5f + (winrate-0.5f) * confidence;

    if (color == kWhite) {
        return 1.f - winrate;
    }
    return winrate;
}
