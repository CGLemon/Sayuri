#include "config.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "game/board.h"
#include "game/symmetry.h"
#include "game/types.h"
#include "game/zobrist.h"
#include "mcts/lcb.h"
#include "mcts/score_value.h"
#include "mcts/time_control.h"
#include "neural/network_basic.h"
#include "pattern/pattern.h"
#include "utils/log.h"
#include "utils/mutex.h"
#include "utils/option.h"
#include "utils/probe_gpu.h"

void ArgsParser::InitOptionsMap() const {
    kOptionsMap.clear();

    const auto SetFalse = [](Option& option, std::string_view) { option.Set(false); };

    // generic options
    kOptionsMap << RegisterOption({"--help", "-h"}, "help", false)
                       .Group("generic")
                       .Helper("Print command-line help.");
    kOptionsMap << RegisterOption({"--mode", "-m"}, "mode", std::string{"gtp"})
                       .Group("generic")
                       .Choices<std::string>({"gtp", "selfplay", "benchmark", "test"},
                                             {"gtp", "selfplay", "benchmark", "test"})
                       .Helper("Select the execution mode.");
    kOptionsMap << RegisterOption({"--ponder"}, "ponder", false)
                       .Group("generic")
                       .Helper("Enable thinking on opponent's time.");
    kOptionsMap << RegisterOption({"--reuse-tree"}, "reuse_tree", false)
                       .Group("generic")
                       .Helper("Reuse part of the previous search tree for the next move.");
    kOptionsMap << RegisterOption({"--friendly-pass"}, "friendly_pass", false)
                       .Group("generic")
                       .Helper("Pass after the opponent passes under area scoring when safely "
                               "winning.");
    kOptionsMap << RegisterOption({"--analysis-verbose", "-a"}, "analysis_verbose", false)
                       .Group("generic")
                       .Helper("Print detailed search information.");
    kOptionsMap << RegisterOption({"--quiet", "-q"}, "quiet", false)
                       .Group("generic")
                       .Helper("Disable all diagnostic output.");
    kOptionsMap << RegisterOption({"--capture-all-dead"}, "capture_all_dead", false)
                       .Group("generic")
                       .Helper("Try to remove all dead strings before pass, only effective with "
                               "area scoring.");

    kOptionsMap << RegisterOption({"--timemanage"},
                                  "timemanage",
                                  static_cast<int>(TimeControl::TimeManagement::kOff))
                       .Group("generic")
                       .Choices<int>({"off", "on", "fast", "keep"},
                                     {static_cast<int>(TimeControl::TimeManagement::kOff),
                                      static_cast<int>(TimeControl::TimeManagement::kOn),
                                      static_cast<int>(TimeControl::TimeManagement::kFast),
                                      static_cast<int>(TimeControl::TimeManagement::kKeep)})
                       .Helper("Control search time management behavior.");

    kOptionsMap << RegisterOption({"--board-size", "-s"}, "default_boardsize", kDefaultBoardSize)
                       .Group("generic")
                       .Helper("Set the default board size.");
    kOptionsMap << RegisterOption({"--komi", "-k"}, "default_komi", kDefaultKomi)
                       .Group("generic")
                       .Helper("Set the default komi.");

    kOptionsMap << RegisterOption({"--cache-memory-mib"}, "cache_memory_mib", 400)
                       .Group("generic")
                       .Helper("Set the NN cache size in MiB.");
    kOptionsMap << RegisterOption({"--playouts", "-p"}, "playouts", -1)
                       .Group("generic")
                       .Helper("Set the maximum number of playouts.");
    kOptionsMap << RegisterOption({"--ponder-factor"}, "ponder_factor", 100)
                       .Group("generic")
                       .Helper("Set the ponder search playout multiplier.");
    kOptionsMap << RegisterOption({"--const-time"}, "const_time", 0)
                       .Group("generic")
                       .Helper("Set constant search time per move in seconds.");
    kOptionsMap << RegisterOption({"--threads", "-t"}, "threads", 0)
                       .Group("generic")
                       .Helper("Set the number of search threads.");
    kOptionsMap << RegisterOption({"--virtual-loss-count"}, "virtual_loss_count", 1)
                       .Group("generic")
                       .Helper("Set the virtual loss count used by parallel search.");

    kOptionsMap << RegisterOption({"--kgs-hint"}, "kgs_hint", std::string{})
                       .Group("generic")
                       .Setter([](Option& option, std::string_view raw) {
                           std::string hint(raw);
                           std::replace(hint.begin(), hint.end(), '+', ' ');
                           option.Set(hint);
                       })
                       .Helper("Pass a KGS-compatible hint string to the engine.");
    kOptionsMap << RegisterOption({"--weights", "-w"}, "weights_file", std::string{})
                       .Group("generic")
                       .Helper("Set the network weights file.");
    kOptionsMap << RegisterOption({"--weights-dir"}, "weights_dir", std::string{})
                       .Group("generic")
                       .Helper("Set the directory used to find network weights.");
    kOptionsMap << RegisterOption({"--book"}, "book_file", std::string{})
                       .Group("generic")
                       .Helper("Set the opening book file.");
    kOptionsMap << RegisterOption({"--patterns"}, "patterns_file", std::string{})
                       .Group("generic")
                       .Helper("Set the pattern file.");
    kOptionsMap << RegisterOption({"--logfile", "-l"}, "logfile", std::string{})
                       .Group("generic")
                       .Setter([](Option& option, std::string_view raw) {
                           auto filename = std::string(raw);
                           option.Set(filename);
                           LogWriter::Get().SetFilename(std::move(filename));
                       })
                       .Helper("Set the file used to log input and output.");

    kOptionsMap << RegisterOption({"--resign-threshold", "-r"}, "resign_threshold", 0.1f)
                       .Group("generic")
                       .Range(0.f, 1.f)
                       .Helper("Resign when winrate is below the threshold.");

    kOptionsMap << RegisterOption({"--ci-alpha"}, "ci_alpha", 1e-5f)
                       .Group("generic")
                       .Range(0.f, 1.f)
                       .Helper("Set the confidence interval alpha for LCB.");
    kOptionsMap << RegisterOption({"--lcb-reduction"}, "lcb_reduction", 0.02f)
                       .Group("generic")
                       .Range(0.f, 1.f)
                       .Helper("Reduce the LCB weights during move selection.");
    kOptionsMap << RegisterOption({"--fpu-reduction"}, "fpu_reduction", 0.25f)
                       .Group("generic")
                       .Helper("Set the FPU reduction.");
    kOptionsMap << RegisterOption({"--root-fpu-reduction"}, "root_fpu_reduction", 0.25f)
                       .Group("generic")
                       .Helper("Set the root FPU reduction.");
    kOptionsMap << RegisterOption({"--cpuct-init"}, "cpuct_init", 0.5f)
                       .Group("generic")
                       .Helper("Set the initial cPUCT value for MCTS exploration.");
    kOptionsMap << RegisterOption({"--cpuct-base-factor"}, "cpuct_base_factor", 1.0f)
                       .Group("generic")
                       .Helper("Set the cPUCT base scaling factor.");
    kOptionsMap << RegisterOption({"--cpuct-base"}, "cpuct_base", 19652.f)
                       .Group("generic")
                       .Helper("Set the cPUCT base value.");
    kOptionsMap << RegisterOption({"--no-cpuct-dynamic"}, "cpuct_dynamic", true)
                       .Group("generic")
                       .Setter(SetFalse)
                       .NoValue()
                       .Helper("Disable dynamic cPUCT adjustment.");
    kOptionsMap << RegisterOption({"--cpuct-dynamic-k-factor"}, "cpuct_dynamic_k_factor", 4.f)
                       .Group("generic")
                       .Helper("Set the dynamic cPUCT k factor.");
    kOptionsMap << RegisterOption({"--cpuct-dynamic-k-base"}, "cpuct_dynamic_k_base", 10000.f)
                       .Group("generic")
                       .Helper("Set the dynamic cPUCT k base.");
    kOptionsMap << RegisterOption({"--score-utility-factor"}, "score_utility_factor", 0.4f)
                       .Group("generic")
                       .Helper("Set the score-based utility factor for MCTS.");
    kOptionsMap << RegisterOption({"--score-utility-div"}, "score_utility_div", 1.f)
                       .Group("generic")
                       .Helper("Set the score utility divisor.");
    kOptionsMap << RegisterOption({"--gammas-policy-factor"}, "gammas_policy_factor", 0.f)
                       .Group("generic")
                       .Range(0.f, 1.f)
                       .Helper("Set the gamma policy mixing factor.");

    kOptionsMap << RegisterOption({"--root-policy-temp"}, "root_policy_temp", 1.f)
                       .Group("generic")
                       .Range(0.f, 100.f)
                       .Helper("Set the root policy temperature.");
    kOptionsMap << RegisterOption({"--policy-temp"}, "policy_temp", 1.f)
                       .Group("generic")
                       .Range(0.f, 100.f)
                       .Helper("Set the policy temperature.");
    kOptionsMap << RegisterOption({"--no-cache"}, "no_cache", false)
                       .Group("generic")
                       .Helper("Disable neural network cache usage.");
    kOptionsMap << RegisterOption({"--early-symm-cache"}, "early_symm_cache", false)
                       .Group("generic")
                       .Helper("Use symmetry-equivalent NN cache hits during the opening stage.");
    kOptionsMap << RegisterOption({"--symm-pruning"}, "symm_pruning", false)
                       .Group("generic")
                       .Helper("Prune symmetry-equivalent moves during the opening stage.");
    kOptionsMap << RegisterOption({"--use-stm-winrate"}, "use_stm_winrate", false)
                       .Group("generic")
                       .Helper("Use side-to-move winrate values.");
    kOptionsMap << RegisterOption({"--use-optimistic-policy"},
                                  "policy_buffer_offset",
                                  static_cast<int>(PolicyBufferOffset::kNormal))
                       .Group("generic")
                       .Setter([](Option& option, std::string_view) {
                           option.Set(static_cast<int>(PolicyBufferOffset::kOptimistic));
                       })
                       .NoValue()
                       .Helper("Use the optimistic policy instead of the normal policy.");
    kOptionsMap << RegisterOption({"--use-rollout"}, "use_rollout", false)
                       .Group("generic")
                       .Helper("Use random rollout ownership instead of neural network ownership.");
    kOptionsMap << RegisterOption({"--scoring-rule"}, "scoring_rule", static_cast<int>(kArea))
                       .Group("generic")
                       .Choices<int>({"area", "territory"},
                                     {static_cast<int>(kArea), static_cast<int>(kTerritory)})
                       .Helper("Select the scoring rule.");

    // gpu options
    kOptionsMap << RegisterOption({"--no-winograd"}, "winograd", true)
                       .Group("gpu")
                       .Setter(SetFalse)
                       .NoValue()
                       .Helper("Disable Winograd convolution optimizations.");
    kOptionsMap << RegisterOption({"--no-fp16"}, "fp16", true)
                       .Group("gpu")
                       .Setter(SetFalse)
                       .NoValue()
                       .Helper("Disable FP16 neural network inference.");
    kOptionsMap << RegisterOption({"--fixed-nn-boardsize"}, "fixed_nn_boardsize", 0)
                       .Group("gpu")
                       .Helper("Set the minimum neural network board size for GPU backends.");
    kOptionsMap << RegisterOption({"--no-cache-tensorrt-plan"}, "cache_tensorrt_plan", true)
                       .Group("gpu")
                       .Setter(SetFalse)
                       .NoValue()
                       .Helper("Disable TensorRT plan caching.");
    kOptionsMap << RegisterOption({"--batch-size", "-b"}, "batch_size", 0)
                       .Group("gpu")
                       .Helper("Set the neural network evaluation batch size.");
    kOptionsMap << RegisterOption({"--gpu", "-g"}, "gpus", -1)
                       .Group("gpu")
                       .Helper("Add a GPU device to use.");
    kOptionsMap << RegisterOption({"--gpu-waittime"}, "gpu_waittime", 2)
                       .Group("gpu")
                       .Helper("Set the maximum wait time (in milliseconds) for the batched worker "
                               "before evaluation.");
    kOptionsMap << RegisterOption({"--lag-buffer"}, "lag_buffer", 0.f)
                       .Group("gpu")
                       .Helper("Set the safety margin for time usage in seconds.");

    // self-play options
    kOptionsMap << RegisterOption({"--selfplay-query"}, "selfplay_query", std::string{})
                       .Group("selfplay")
                       .Helper("Add a self-play setup query.");
    kOptionsMap << RegisterOption({"--forced-playouts-k"}, "forced_playouts_k", 0.f)
                       .Group("selfplay")
                       .Helper("Set the forced playouts coefficient.");
    kOptionsMap << RegisterOption({"--suppress-pass-factor"}, "suppress_pass_factor", 0.1667f)
                       .Group("selfplay")
                       .Range(0.f, 1.f)
                       .Helper("Set the board-fill threshold factor for suppressing pass during "
                               "expansion.");
    kOptionsMap << RegisterOption({"--random-min-visits"}, "random_min_visits", 1)
                       .Group("selfplay")
                       .Helper("Set the minimum visit count for random move selection.");
    kOptionsMap << RegisterOption({"--random-min-ratio"}, "random_min_ratio", 0.f)
                       .Group("selfplay")
                       .Range(0.f, 1.f)
                       .Helper("Set the minimum visit ratio for random move selection.");
    kOptionsMap << RegisterOption({"--random-moves-factor"}, "random_moves_factor", 0.f)
                       .Group("selfplay")
                       .Helper("Set the opening random-move phase length as a board-area factor.");
    kOptionsMap << RegisterOption({"--random-moves-temp"}, "random_moves_temp", 1.f)
                       .Group("selfplay")
                       .Range(0.f, 100.f)
                       .Helper("Set the random move temperature.");
    kOptionsMap << RegisterOption({"--random-opening-prob"}, "random_opening_prob", 0.f)
                       .Group("selfplay")
                       .Range(0.f, 1.f)
                       .Helper("Set the probability of starting self-play with a random opening "
                               "sequence.");
    kOptionsMap << RegisterOption({"--random-opening-temp"}, "random_opening_temp", 1.f)
                       .Group("selfplay")
                       .Range(0.f, 100.f)
                       .Helper("Set the random opening temperature.");

    kOptionsMap << RegisterOption({"--gumbel-c-visit"}, "gumbel_c_visit", 50.f)
                       .Group("selfplay")
                       .Helper("Set the Gumbel visit constant.");
    kOptionsMap << RegisterOption({"--gumbel-c-scale"}, "gumbel_c_scale", 1.f)
                       .Group("selfplay")
                       .Helper("Set the Gumbel scale constant.");
    kOptionsMap << RegisterOption({"--gumbel-prom-visits"}, "gumbel_prom_visits", 1)
                       .Group("selfplay")
                       .Helper("Set the promoted visit count for Gumbel search.");
    kOptionsMap << RegisterOption({"--gumbel-considered-moves"}, "gumbel_considered_moves", 16)
                       .Group("selfplay")
                       .Helper("Set the number of moves considered by Gumbel search.");
    kOptionsMap << RegisterOption({"--gumbel-playouts-threshold"}, "gumbel_playouts_threshold", 400)
                       .Group("selfplay")
                       .Helper("Set the playout threshold for Gumbel search.");
    kOptionsMap << RegisterOption({"--gumbel"}, "gumbel", false)
                       .Group("selfplay")
                       .Helper("Enable Gumbel search.");
    kOptionsMap << RegisterOption(
                       {"--always-completed-q-policy"}, "always_completed_q_policy", false)
                       .Group("selfplay")
                       .Helper("Always use completed-Q policy values.");

    kOptionsMap << RegisterOption({"--dirichlet-noise", "--noise", "-n"}, "dirichlet_noise", false)
                       .Group("selfplay")
                       .Helper("Enable Dirichlet noise at the root.");
    kOptionsMap << RegisterOption({"--dirichlet-epsilon"}, "dirichlet_epsilon", 0.25f)
                       .Group("selfplay")
                       .Helper("Set the Dirichlet noise mixing weight.");
    kOptionsMap << RegisterOption({"--dirichlet-init"}, "dirichlet_init", 0.03f)
                       .Group("selfplay")
                       .Helper("Set the initial Dirichlet alpha value.");
    kOptionsMap << RegisterOption({"--dirichlet-factor"}, "dirichlet_factor", 361.f)
                       .Group("selfplay")
                       .Helper("Set the Dirichlet alpha scaling factor.");

    kOptionsMap << RegisterOption({"--kldgain-per-node"}, "kldgain_per_node", 0.0)
                       .Group("selfplay")
                       .Range(0.0, 100.0)
                       .Helper("Set the KLD gain-per-visit threshold for early stopping.");
    kOptionsMap << RegisterOption({"--kldgain-interval"}, "kldgain_interval", 0)
                       .Group("selfplay")
                       .Helper("Set the interval for KLD gain updates.");

    kOptionsMap << RegisterOption({"--resign-playouts"}, "resign_playouts", 0)
                       .Group("selfplay")
                       .Helper("Set the fast-search playout cap after the resign threshold is "
                               "reached.");
    kOptionsMap << RegisterOption({"--fastsearch-playouts"}, "fastsearch_playouts", 0)
                       .Group("selfplay")
                       .Helper("Set the reduced playout count for fast search.");
    kOptionsMap << RegisterOption({"--fastsearch-playouts-prob"}, "fastsearch_playouts_prob", 0.f)
                       .Group("selfplay")
                       .Range(0.f, 1.f)
                       .Helper("Set the probability of using reduced playouts.");
    kOptionsMap << RegisterOption({"--random-fastsearch-prob"}, "random_fastsearch_prob", 0.f)
                       .Group("selfplay")
                       .Range(0.f, 1.f)
                       .Helper("Set the probability of choosing a random move during fast search.");
    kOptionsMap << RegisterOption({"--first-pass-bonus"}, "first_pass_bonus", false)
                       .Group("selfplay")
                       .Helper("Enable endgame score bonuses for pass and cleanup moves.");
    kOptionsMap << RegisterOption({"--resign-discard-prob"}, "resign_discard_prob", 0.f)
                       .Group("selfplay")
                       .Range(0.f, 1.f)
                       .Helper("Set the probability of discarding resigned self-play games.");

    kOptionsMap << RegisterOption({"--num-games"}, "num_games", 0)
                       .Group("selfplay")
                       .Helper("Set the number of self-play games to generate.");
    kOptionsMap << RegisterOption({"--parallel-games"}, "parallel_games", 1)
                       .Group("selfplay")
                       .Helper("Set the number of self-play games to run in parallel.");
    kOptionsMap << RegisterOption({"--komi-stddev"}, "komi_stddev", 0.f)
                       .Group("selfplay")
                       .Helper("Set the standard deviation for random komi.");
    kOptionsMap << RegisterOption({"--komi-big-stddev"}, "komi_big_stddev", 0.f)
                       .Group("selfplay")
                       .Helper("Set the larger standard deviation for random komi.");
    kOptionsMap << RegisterOption({"--komi-big-stddev-prob"}, "komi_big_stddev_prob", 0.f)
                       .Group("selfplay")
                       .Range(0.f, 1.f)
                       .Helper("Set the probability of using the larger random komi deviation.");
    kOptionsMap << RegisterOption({"--handicap-fair-komi-prob"}, "handicap_fair_komi_prob", 0.f)
                       .Group("selfplay")
                       .Range(0.f, 1.f)
                       .Helper("Set the probability of using handicap fair komi.");
    kOptionsMap << RegisterOption({"--target-directory"}, "target_directory", std::string{})
                       .Group("selfplay")
                       .Helper("Set the target directory for generated output.");

    // benchmark options
    kOptionsMap << RegisterOption({"--benchmark-query"}, "benchmark_query", std::string{})
                       .Group("benchmark")
                       .Helper("Add a benchmark setup query.");
}

void ArgsParser::InitBasicParameters() const {
    PatternHashAndCoordsInit();
    Board::InitPattern3();
    Zobrist::Initialize();
    Symmetry::Get().Initialize();
    LcbEntries::Get().Initialize(GetOption<float>("ci_alpha"));
    ScoreValue::Get().Initialize();
    LogOptions::Get().SetQuiet(GetOption<bool>("quiet"));

    // Try to select a reasonable number for const time and playouts.
    bool already_set_time = !IsOptionDefault("const_time");
    bool already_set_playouts = !IsOptionDefault("playouts");

    if (!already_set_time && !already_set_playouts) {
        SetOption("const_time", 10); // 10 seconds
    }
    if (!already_set_playouts) {
        SetOption("playouts", std::numeric_limits<int>::max() / 2);
    }

    // If the threads/batchsize are zero, program will select a reasonable
    // number based on your device and others setting.
    bool already_set_specific_gpus = !IsOptionDefault("gpus");
    const bool use_gpu = IsGpuAvailable();
    const int num_gpus = !use_gpu                  ? 0
                       : already_set_specific_gpus ? GetOptionCount("gpus")
                                                   : GetGpuCount();

    const int cores = std::max((int)std::thread::hardware_concurrency(), 1);
    int select_threads = GetOption<int>("threads");
    int select_batchsize = GetOption<int>("batch_size");

    const auto mode = GetOption<std::string>("mode");
    if (mode == "gtp" || mode == "benchmark") {
        // GPU case:
        // case 1. if number of threads are 0, use thread count of reasonable number
        // case 2. number of threads and batches are given
        // other cases. thread count are equal to (batch size) * 2
        //
        // CPU case (ingore batch size):
        // case 1. if number of threads are 0, use thread count of reasonable number
        // case 2. number of threads are given

        bool already_set_threads = !IsOptionDefault("threads");
        int reasonable_threads = 0;
        int leela_base_threads = 10; // leela uses 10 threads and 5 batches by default

        // Now we compute the reasonable base on known information, like playouts or
        // CPU cores number.
        if (already_set_playouts) {
            // Assume the bound is playouts count. Avoid reducing the strength
            // too much. We select thread count based on playouts count.
            int playouts = GetOption<int>("playouts");

            // TODO: Design a approximate function for 'reasonable_threads'.
            if (playouts <= 400 && use_gpu) {
                reasonable_threads = 1;
            } else if (playouts <= 1600) {
                reasonable_threads = 4;
            } else if (playouts <= 6400) {
                reasonable_threads = 6;
            } else if (playouts <= 12800) {
                reasonable_threads = 10;
            } else if (playouts <= 25600) {
                reasonable_threads = 16;
            } else if (playouts <= 51200) {
                reasonable_threads = 32;
            } else {
                // Will take long time, so allocate the threads as many as
                // possible.
                reasonable_threads = 0;
            }
        }

        // The performance increase is slow down on normal device (eg. RTX 4060Ti) after
        // the batch size reaches 32. Maybe we will change this value in the future. Notice
        // priority of this bound is lower than threads bound.
        const int reasonable_batchsize_bound_per_gpu = 32;

        // The performance would be reduced with large number of search threads because
        // single playout may be fail when many threads are on the same path. After testing,
        // the bound should not be greater than 64 even if it runs on the multi-gpu device.
        // (eg. 4 x RTX 3080Ti)
        const int reasonable_threads_bound = 64;

        if (reasonable_threads == 0) {
            // Assume the bound is thinking time. Actually we can't know the thinking time
            // so only selecting the maximum thread count.
            reasonable_threads = use_gpu ? 2 * num_gpus * reasonable_batchsize_bound_per_gpu
                                         : reasonable_threads_bound;
        }

        if (!already_set_threads) {
            // Assume new comer forgets to set thread count. Avoid eating too
            // much computation resource so we set a lower thread count bound.
            // We are so nice :-)
            reasonable_threads = std::min(reasonable_threads, leela_base_threads);
        }
        reasonable_threads = std::min(reasonable_threads, reasonable_threads_bound);

        if (use_gpu) {
            if (select_threads == 0 && select_batchsize == 0) {
                select_threads = reasonable_threads;
                select_batchsize = (select_threads + (2 * num_gpus) - 1) / (2 * num_gpus);
            } else if (select_threads == 0 && select_batchsize != 0) {
                select_threads = 2 * num_gpus * select_batchsize;
            } else if (select_threads != 0 && select_batchsize == 0) {
                // No idea why somebody wants to use threads less than the number of GPUs
                // but should at least prevent hiccup.
                select_threads = std::max(select_threads, num_gpus);

                select_batchsize = (select_threads + (2 * num_gpus) - 1) / (2 * num_gpus);
            }
        } else {
            if (select_threads == 0) {
                select_threads = std::min(reasonable_threads, cores);
            }
            // The CPU-only backend's batch size is always 1.
            select_batchsize = 1;
        }
    } else if (mode == "selfplay") {
        const int parallel_games = GetOption<int>("parallel_games");
        if (select_batchsize == 0) {
            if (use_gpu) {
                select_batchsize = parallel_games / (2 * num_gpus) +
                                   static_cast<bool>(parallel_games % (2 * num_gpus));
            } else {
                // The CPU-only backend's batch size is always 1.
                select_batchsize = 1;
            }
        }
        select_threads = 1;
    }
    SetOption("use_gpu", use_gpu);
    SetOption("threads", std::max(select_threads, 1));
    SetOption("batch_size", std::max(select_batchsize, 1));

    // Assign the GPUs index.
    if (use_gpu && !already_set_specific_gpus) {
        for (int idx = 0; idx < num_gpus; ++idx) {
            SetOption("gpus", idx);
        }
    }

    // Set the root fpu value.
    bool already_set_fpu_root = !IsOptionDefault("root_fpu_reduction");
    if (!already_set_fpu_root) {
        bool as_default = true;
        SetOption("root_fpu_reduction", GetOption<float>("fpu_reduction"), as_default);
    }

    // Set the root temperature value.
    bool already_set_root_temp = !IsOptionDefault("root_policy_temp");
    if (!already_set_root_temp) {
        bool as_default = true;
        SetOption("root_policy_temp", GetOption<float>("policy_temp"), as_default);
    }

    // Set the lag buffer time.
    bool already_set_lagbuffer = !IsOptionDefault("lag_buffer");
    if (!already_set_lagbuffer) {
        float lag_buffer_base = 0.25f;
        if (use_gpu) {
            SetOption("lag_buffer", lag_buffer_base);
        } else {
            // The time of CPU hiccup is longer than GPU backend. We
            // a bigger value.
            SetOption("lag_buffer", 2 * lag_buffer_base);
        }
    }
}

bool IsParameter(const std::string& param) {
    if (param.empty()) {
        return false;
    }
    return param[0] != '-';
};

std::string RemoveComment(std::string line) {
    auto out = std::string{};
    for (auto c : line) {
        if (c == '#') {
            break;
        }
        out += c;
    }
    return out;
}

ArgsParser::ArgsParser(int argc, char** argv) {
    auto spt = Splitter(argc, argv);

    InitOptionsMap();

    // Remove the name.
    const auto name = spt.RemoveWord(0);
    (void)name;

    auto config = std::string{};

    if (const auto res = spt.FindNext({"--config", "-config"})) {
        if (IsParameter(res->Get<>())) {
            config = res->Get<>();
            spt.RemoveSlice(res->Index() - 1, res->Index() + 1);
        }
    }

    if (!config.empty()) {
        auto file = std::ifstream{};

        file.open(config);
        if (file.is_open()) {
            auto lines = std::string{};
            auto line = std::string{};

            while (std::getline(file, line)) {
                line = RemoveComment(line);
                if (!line.empty()) {
                    lines += (line + ' ');
                }
            }
            file.close();

            auto cspt = Splitter(lines);
            Parse(cspt);
        }
    }

    Parse(spt);

    DumpWarning();
    InitBasicParameters();
}

void ArgsParser::Parse(Splitter& spt) {
    std::vector<std::string> args;
    args.reserve(spt.GetCount() + 1);
    args.emplace_back("sayuri");
    for (auto i = size_t{0}; i < spt.GetCount(); ++i) {
        args.emplace_back(spt.GetWord(i)->Get<>());
    }

    std::vector<char*> argv;
    argv.reserve(args.size());
    for (auto& arg : args) {
        argv.emplace_back(arg.data());
    }

    try {
        kOptionsMap.ParseArgs(static_cast<int>(argv.size()), argv.data());
    } catch (const std::exception& e) {
        LOGGING << "Command Error: " << e.what() << std::endl;
        DumpHelper();
    }

    UniqueOption("gpus");
    UniqueOption("selfplay_query");
    UniqueOption("benchmark_query");

    if (GetOption<bool>("help")) {
        DumpHelper();
    }
}

void ArgsParser::DumpHelper() const {
    LOGGING << OptionHelpersToString();
    exit(0);
}

void ArgsParser::DumpWarning() const {
    const auto friendly_pass = GetOption<bool>("friendly_pass");
    const auto capture_all_dead = GetOption<bool>("capture_all_dead");
    if (friendly_pass && capture_all_dead) {
        LOGGING << "Nonsensical options: The --capture-all-dead option may suppress the effect of "
                   "--friendly-pass.\n";
    }
}
