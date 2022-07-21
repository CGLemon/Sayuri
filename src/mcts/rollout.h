#pragma once

#include <vector>
#include "game/game_state.h"

float GetRolloutWinrate(GameState &state,
                            const int depth,
                            const int num_sims,
                            const int color,

                            // [ black ~ white ]
                            // [ 1     ~    -1 ]
                            std::vector<float> &mcowner,
                            float &black_score
                       );
