#pragma once

#include <vector>
#include "game/game_state.h"

float GetBlackRolloutResult(GameState &state,
                                // [ black ~ white ]
                                // [ 1     ~    -1 ]
                                std::vector<float> &mcowner,
                                float &black_score
                           );
