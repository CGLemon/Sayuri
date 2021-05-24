#pragma once

#include "config.h"

class Parameters {
public:
    Parameters() = default;

    void Reset() {
        threads = GetOption<int>("threads");
        playouts = GetOption<int>("playouts");
    }

    int threads;
    int visits;
    int playouts;
    int random_min_visits;

    bool dirichlet_noise;
    bool collect;

    float draw_threshold;
    float resign_threshold;
    float fpu_root_reduction;
    float fpu_reduction;
    float cpuct_init;
    float cpuct_root_init;
    float cpuct_base;
    float cpuct_root_base;
    float draw_factor;
    float draw_root_factor;
    float dirichlet_epsilon;
    float dirichlet_factor;
    float dirichlet_init;
};


