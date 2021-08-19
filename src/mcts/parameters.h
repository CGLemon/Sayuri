#pragma once

#include "game/types.h"
#include "config.h"

#include <array>

class Parameters {
public:
    Parameters() = default;

    void Reset() {
        threads = GetOption<int>("threads");
        playouts = GetOption<int>("playouts");
        visits = GetOption<int>("visits");

        resign_threshold = GetOption<float>("resign_threshold");
        lcb_reduction = GetOption<float>("lcb_reduction");
        fpu_root_reduction = GetOption<float>("fpu_root_reduction");
        fpu_reduction = GetOption<float>("fpu_reduction");

        cpuct_init = GetOption<float>("cpuct_init");
        cpuct_root_init = GetOption<float>("cpuct_root_init");

        cpuct_base = GetOption<float>("cpuct_base");
        cpuct_root_base = GetOption<float>("cpuct_root_base");

        draw_factor = GetOption<float>("draw_factor");
        draw_factor = GetOption<float>("draw_root_factor");

        random_min_visits = GetOption<int>("random_min_visits");
        random_moves_cnt = GetOption<int>("random_moves_cnt");

        dirichlet_noise = GetOption<bool>("dirichlet_noise");
        dirichlet_epsilon = GetOption<float>("dirichlet_epsilon");
        dirichlet_factor = GetOption<float>("dirichlet_factor");
        dirichlet_init = GetOption<float>("dirichlet_init");

        forced_policy_factor = GetOption<float>("forced_policy_factor");
        score_utility_factor = GetOption<float>("score_utility_factor");
        cap_playouts = GetOption<float>("cap_playouts");
    }

    int threads;
    int visits;
    int playouts;
    int random_min_visits;
    int random_moves_cnt;

    float resign_threshold;

    float lcb_reduction;
    float fpu_root_reduction;
    float fpu_reduction;
    float cpuct_init;
    float cpuct_root_init;
    float cpuct_base;
    float cpuct_root_base;
    float draw_factor;
    float draw_root_factor;

    bool dirichlet_noise;
    float dirichlet_epsilon;
    float dirichlet_factor;
    float dirichlet_init;

    float forced_policy_factor;
    float score_utility_factor;

    int cap_playouts;

    std::array<float, kNumVertices + 10> dirichlet_buffer;
};


