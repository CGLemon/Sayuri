#pragma once

#include "neural/network.h"

#include <string>

struct SelfplayReport {
    int num_games{0};
    size_t accm_playouts{0};

    double GetAccumulationPlayoutsPerGame() {
        return (double)accm_playouts/num_games;
    }
};

SelfplayReport ComputeSelfplayAccumulation(std::string sgf_name);
