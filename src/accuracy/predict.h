#pragma once

#include "mcts/search.h"
#include "game/game_state.h"

#include <string>

float PredictSgfAccuracy(Search &search,
                             GameState &main_state,
                             std::string sgf_name);
