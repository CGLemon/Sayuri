#pragma once

#include "neural/network.h"
#include "game/game_state.h"

#include <string>

struct AccuracyReport {
    int num_positions{0};
    int num_matched{0};

    double GetAccuracy() {
        return (double)num_positions/num_matched;
    }
};

AccuracyReport ComputeNetAccuracy(Network &network,
                                  std::string sgf_name);
