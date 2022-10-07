#pragma once

#include "game/types.h"
#include "config.h"

#include <array>

class Parameters {
public:
    Parameters() = default;

    void Reset() {
        threads = GetOption<int>("threads");
        batch_size = GetOption<int>("batch_size");
        playouts = GetOption<int>("playouts");
        ponder_playouts = GetOption<int>("ponder_playouts");
        const_time = GetOption<int>("const_time");

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

        lag_buffer = GetOption<int>("lag_buffer");
        ponder = GetOption<bool>("ponder");
        reuse_tree = GetOption<bool>("reuse_tree");
        friendly_pass = GetOption<bool>("friendly_pass");

        root_policy_temp = GetOption<float>("root_policy_temp");
        policy_temp = GetOption<float>("policy_temp");
        use_rollout = GetOption<bool>("rollout");
        no_dcnn = GetOption<bool>("no_dcnn");
        first_pass_bonus = GetOption<bool>("first_pass_bonus");
        symm_pruning = GetOption<bool>("symm_pruning");
        use_stm_winrate = GetOption<bool>("use_stm_winrate");
        analysis_verbose = GetOption<bool>("analysis_verbose");
    }

    int threads;
    int batch_size;
    int playouts;
    int ponder_playouts;
    int const_time;
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

    float root_policy_temp;
    float policy_temp;

    int cap_playouts;
    int lag_buffer;

    bool ponder;
    bool reuse_tree;
    bool friendly_pass;
    bool use_rollout;
    bool no_dcnn;
    bool first_pass_bonus;
    bool symm_pruning;
    bool use_stm_winrate;
    bool analysis_verbose;

    std::array<float, kNumVertices + 10> dirichlet_buffer;
};
