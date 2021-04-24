#pragma once

class Parameters {
public:
    Parameters() = default;

    int threads;
    int visits;
    int playouts;
    int random_min_visits;
    int forced_checkmate_depth;
    int forced_checkmate_root_depth;

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
