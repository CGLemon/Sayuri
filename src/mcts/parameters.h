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
        ponder_factor = GetOption<int>("ponder_factor");
        const_time = GetOption<int>("const_time");
        expand_threshold = GetOption<int>("expand_threshold");

        resign_threshold = GetOption<float>("resign_threshold");
        lcb_utility_factor = GetOption<float>("lcb_utility_factor");
        lcb_reduction = GetOption<float>("lcb_reduction");
        fpu_reduction = GetOption<float>("fpu_reduction");
        cpuct_init = GetOption<float>("cpuct_init");
        cpuct_base_factor = GetOption<float>("cpuct_base_factor");
        cpuct_base = GetOption<float>("cpuct_base");
        draw_factor = GetOption<float>("draw_factor");

        random_min_visits = GetOption<int>("random_min_visits");
        random_moves_factor = GetOption<float>("random_moves_factor");

        gumbel_considered_moves = GetOption<int>("gumbel_considered_moves");
        gumbel_playouts = GetOption<int>("gumbel_playouts");
        gumbel = GetOption<bool>("gumbel");
        always_completed_q_policy = GetOption<bool>("always_completed_q_policy");

        dirichlet_noise = GetOption<bool>("dirichlet_noise");
        dirichlet_epsilon = GetOption<float>("dirichlet_epsilon");
        dirichlet_factor = GetOption<float>("dirichlet_factor");
        dirichlet_init = GetOption<float>("dirichlet_init");

        score_utility_factor = GetOption<float>("score_utility_factor");
        score_utility_div = GetOption<float>("score_utility_div");
        resign_playouts = GetOption<float>("resign_playouts");
        reduce_playouts = GetOption<float>("reduce_playouts");
        reduce_playouts_prob = GetOption<float>("reduce_playouts_prob"); 

        lag_buffer = GetOption<int>("lag_buffer");
        ponder = GetOption<bool>("ponder");
        reuse_tree = GetOption<bool>("reuse_tree");
        friendly_pass = GetOption<bool>("friendly_pass");

        root_policy_temp = GetOption<float>("root_policy_temp");
        policy_temp = GetOption<float>("policy_temp");
        use_rollout = GetOption<bool>("rollout");
        no_dcnn = GetOption<bool>("no_dcnn");
        root_dcnn = GetOption<bool>("root_dcnn");
        first_pass_bonus = GetOption<bool>("first_pass_bonus");
        symm_pruning = GetOption<bool>("symm_pruning");
        use_stm_winrate = GetOption<bool>("use_stm_winrate");
        analysis_verbose = GetOption<bool>("analysis_verbose");
    }

    int threads;
    int batch_size;
    int playouts;
    int ponder_factor;
    int const_time;
    int random_min_visits;
    float random_moves_factor;

    float resign_threshold;

    float lcb_utility_factor;
    float lcb_reduction;
    float fpu_reduction;
    float cpuct_init;
    float cpuct_base_factor;
    float cpuct_base;
    float draw_factor;

    int gumbel_considered_moves;
    int gumbel_playouts;
    bool gumbel;

    bool dirichlet_noise;
    float dirichlet_epsilon;
    float dirichlet_factor;
    float dirichlet_init;

    float score_utility_factor;
    float score_utility_div;

    float root_policy_temp;
    float policy_temp;

    int resign_playouts;
    int reduce_playouts;
    float reduce_playouts_prob;
    int lag_buffer;
    int expand_threshold;

    bool ponder;
    bool reuse_tree;
    bool friendly_pass;
    bool use_rollout;
    bool no_dcnn;
    bool root_dcnn;
    bool first_pass_bonus;
    bool symm_pruning;
    bool use_stm_winrate;
    bool analysis_verbose;
    bool always_completed_q_policy;

    std::array<float, kNumVertices + 10> dirichlet_buffer;
};
