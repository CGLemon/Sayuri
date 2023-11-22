#pragma once

#include <vector>
#include <cstdint>
#include "game/game_state.h"
#include "game/types.h"

// The Progressive Widening algorithm.
// The recursive function is
//
// T(n+1) = T(n) + 40 x 1.4^n, T(0) = 0
//
static const std::vector<std::uint64_t> kProgressiveWideningTable = {
    0,
    40,
    112,
    241,
    474,
    893,
    1648,
    3008,
    5456,
    9863,
    17797,
    32078,
    57785,
    104058,
    187349,
    337274,
    607139,
    1092897,
    1967261,
    3541117,
    6374058,
    11473352,
    20652082,
    37173796,
    66912881,
    120443234,
    216797870,
    390236216,
    702425239,
    1264365481
};

inline int ComputeWidth(int visits) {
    const int size = kProgressiveWideningTable.size();
    int c;
    for (c = 0; c < size; ++c) {
        if (kProgressiveWideningTable[c] > std::uint64_t(visits)) {
            return c;
        }
    }
    return c;
}

inline float GetBlackRolloutResult(GameState &state,
                                   // The MC ownermap value. Set 1 if the final position is
                                   // black. Set -1 if it is white. The another is 0.
                                   // 
                                   // [ black ~ white ]
                                   // [ 1     ~    -1 ]
                                   float *mcowner,
                                   float &black_score) {
    auto fork_state = state;
    const int num_intersections = fork_state.GetNumIntersections();
    int num_curr_moves = 0;
    const int max_move_len = 2 * num_intersections + 1;

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
