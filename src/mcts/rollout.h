#pragma once

#include "game/game_state.h"

float GetBlackRolloutResult(GameState &state,
                                // [ black ~ white ]
                                // [ 1     ~    -1 ]
                                float *mcowner,
                                float &black_score
                           );
