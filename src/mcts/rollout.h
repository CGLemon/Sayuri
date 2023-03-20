#pragma once

#include <vector>
#include <cstdint>
#include "game/game_state.h"

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

float GetBlackRolloutResult(GameState &state,
                            // The MC ownermap value. Set 1 if the final position is
                            // black. Set -1 if it is white. The another is 0.
                            // 
                            // [ black ~ white ]
                            // [ 1     ~    -1 ]
                            float *mcowner,
                            float &black_score
                           );
