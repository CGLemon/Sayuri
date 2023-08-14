#pragma once

#include "neural/network.h"

#include <string>

struct SelfplayReport {
    int num_games{0};
    int accm_playouts{0};
    int accm_moves{0};
};

SelfplayReport ComputeSelfplayAccumulation(std::string sgf_name);
