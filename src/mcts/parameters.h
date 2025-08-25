#pragma once

#include "game/types.h"
#include "utils/option.h"
#include "config.h"

#include <array>
#include <string>

class Parameters {
public:
    Parameters() = default;

    void Reset() {
        timemanage = GetOption<int>("timemanage");
        threads = GetOption<int>("threads");
        batch_size = GetOption<int>("batch_size");
        playouts = GetOption<int>("playouts");
        ponder_factor = GetOption<int>("ponder_factor");
        const_time = GetOption<int>("const_time");
        virtual_loss_count = GetOption<int>("virtual_loss_count");

        resign_threshold = GetOption<float>("resign_threshold");
        lcb_reduction = GetOption<float>("lcb_reduction");
        fpu_reduction = GetOption<float>("fpu_reduction");
        root_fpu_reduction = GetOption<float>("root_fpu_reduction");
        cpuct_init = GetOption<float>("cpuct_init");
        cpuct_base_factor = GetOption<float>("cpuct_base_factor");
        cpuct_base = GetOption<float>("cpuct_base");
        cpuct_dynamic = GetOption<bool>("cpuct_dynamic");
        cpuct_dynamic_k_factor = GetOption<float>("cpuct_dynamic_k_factor");
        cpuct_dynamic_k_base = GetOption<float>("cpuct_dynamic_k_base");
        forced_playouts_k = GetOption<float>("forced_playouts_k");
        suppress_pass_factor = GetOption<float>("suppress_pass_factor");
        gammas_policy_factor = GetOption<float>("gammas_policy_factor");

        random_min_visits = GetOption<int>("random_min_visits");
        random_min_ratio = GetOption<float>("random_min_ratio");
        random_moves_factor = GetOption<float>("random_moves_factor");
        random_moves_temp = GetOption<float>("random_moves_temp");

        gumbel_c_visit = GetOption<float>("gumbel_c_visit");
        gumbel_c_scale = GetOption<float>("gumbel_c_scale");
        gumbel_prom_visits = GetOption<int>("gumbel_prom_visits");
        gumbel_considered_moves = GetOption<int>("gumbel_considered_moves");
        gumbel_playouts_threshold = GetOption<int>("gumbel_playouts_threshold");
        gumbel = GetOption<bool>("gumbel");
        always_completed_q_policy = GetOption<bool>("always_completed_q_policy");

        dirichlet_noise = GetOption<bool>("dirichlet_noise");
        dirichlet_epsilon = GetOption<float>("dirichlet_epsilon");
        dirichlet_factor = GetOption<float>("dirichlet_factor");
        dirichlet_init = GetOption<float>("dirichlet_init");

        kldgain_per_node = GetOption<double>("kldgain_per_node");
        kldgain_interval = GetOption<int>("kldgain_interval");

        score_utility_factor = GetOption<float>("score_utility_factor");
        score_utility_div = GetOption<float>("score_utility_div");
        resign_playouts = GetOption<int>("resign_playouts");
        fastsearch_playouts = GetOption<int>("fastsearch_playouts");
        fastsearch_playouts_prob = GetOption<float>("fastsearch_playouts_prob");
        random_fastsearch_prob = GetOption<float>("random_fastsearch_prob");
        resign_discard_prob = GetOption<float>("resign_discard_prob");

        lag_buffer = GetOption<float>("lag_buffer");
        ponder = GetOption<bool>("ponder");
        reuse_tree = GetOption<bool>("reuse_tree");
        friendly_pass = GetOption<bool>("friendly_pass");

        root_policy_temp = GetOption<float>("root_policy_temp");
        policy_temp = GetOption<float>("policy_temp");
        first_pass_bonus = GetOption<bool>("first_pass_bonus");
        symm_pruning = GetOption<bool>("symm_pruning");
        use_stm_winrate = GetOption<bool>("use_stm_winrate");
        analysis_verbose = GetOption<bool>("analysis_verbose");
        use_rollout = GetOption<bool>("use_rollout");
        capture_all_dead = GetOption<bool>("capture_all_dead");
        no_exploring_phase = false;
    }

    int timemanage;
    int threads;
    int batch_size;
    int playouts;
    int ponder_factor;
    int const_time;
    int random_min_visits;
    int virtual_loss_count;

    float random_min_ratio;
    float random_moves_factor;
    float random_moves_temp;

    float resign_threshold;

    float lcb_reduction;
    float fpu_reduction;
    float root_fpu_reduction;
    float cpuct_init;
    float cpuct_base_factor;
    float cpuct_base;
    float cpuct_dynamic_k_factor;
    float cpuct_dynamic_k_base;
    float forced_playouts_k;
    float suppress_pass_factor;
    float gammas_policy_factor;

    float gumbel_c_visit;
    float gumbel_c_scale;
    int gumbel_prom_visits;
    int gumbel_considered_moves;
    int gumbel_playouts_threshold;
    bool gumbel;

    bool dirichlet_noise;
    float dirichlet_epsilon;
    float dirichlet_factor;
    float dirichlet_init;

    double kldgain_per_node;
    int kldgain_interval;

    float score_utility_factor;
    float score_utility_div;

    float root_policy_temp;
    float policy_temp;

    int resign_playouts;
    int fastsearch_playouts;
    float fastsearch_playouts_prob;
    float random_fastsearch_prob;
    float lag_buffer;
    float resign_discard_prob;

    bool ponder;
    bool reuse_tree;
    bool friendly_pass;
    bool first_pass_bonus;
    bool symm_pruning;
    bool use_stm_winrate;
    bool analysis_verbose;
    bool always_completed_q_policy;
    bool cpuct_dynamic;
    bool use_rollout;
    bool capture_all_dead;
    bool no_exploring_phase;

    std::array<float, kNumVertices + 10> dirichlet_buffer;
};
