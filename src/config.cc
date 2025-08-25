#include "utils/option.h"
#include "utils/log.h"
#include "utils/mutex.h"
#include "utils/probe_gpu.h"
#include "game/zobrist.h"
#include "game/symmetry.h"
#include "game/types.h"
#include "game/board.h"
#include "pattern/pattern.h"
#include "mcts/lcb.h"
#include "mcts/time_control.h"
#include "neural/network_basic.h"
#include "config.h"

#include <limits>
#include <sstream>
#include <fstream>

void ArgsParser::InitOptionsMap() const {
    kOptionsMap["help"] << Option::SetOption(false);
    kOptionsMap["mode"] << Option::SetOption(std::string{"gtp"});
    kOptionsMap["inputs"] << Option::SetOption(std::string{});

    // engine options
    kOptionsMap["ponder"] << Option::SetOption(false);
    kOptionsMap["reuse_tree"] << Option::SetOption(false);
    kOptionsMap["friendly_pass"] << Option::SetOption(false);
    kOptionsMap["analysis_verbose"] << Option::SetOption(false);
    kOptionsMap["quiet"] << Option::SetOption(false);
    kOptionsMap["winograd"] << Option::SetOption(true);
    kOptionsMap["fp16"] << Option::SetOption(true);
    kOptionsMap["capture_all_dead"] << Option::SetOption(false);

    kOptionsMap["timemanage"] << Option::SetOption(static_cast<int>(TimeControl::TimeManagement::kOff));

    kOptionsMap["fixed_nn_boardsize"] << Option::SetOption(0);
    kOptionsMap["defualt_boardsize"] << Option::SetOption(kDefaultBoardSize);
    kOptionsMap["defualt_komi"] << Option::SetOption(kDefaultKomi);

    kOptionsMap["cache_memory_mib"] << Option::SetOption(400);
    kOptionsMap["playouts"] << Option::SetOption(-1);
    kOptionsMap["ponder_factor"] << Option::SetOption(100);
    kOptionsMap["const_time"] << Option::SetOption(0);
    kOptionsMap["batch_size"] << Option::SetOption(0);
    kOptionsMap["threads"] << Option::SetOption(0);
    kOptionsMap["virtual_loss_count"] << Option::SetOption(1);

    kOptionsMap["kgs_hint"] << Option::SetOption(std::string{});
    kOptionsMap["weights_file"] << Option::SetOption(std::string{});
    kOptionsMap["weights_dir"] << Option::SetOption(std::string{});
    kOptionsMap["book_file"] << Option::SetOption(std::string{});
    kOptionsMap["patterns_file"] << Option::SetOption(std::string{});

    kOptionsMap["use_gpu"] << Option::SetOption(false);
    kOptionsMap["gpus"] << Option::SetOption(-1);
    kOptionsMap["gpu_waittime"] << Option::SetOption(2);

    kOptionsMap["resign_threshold"] << Option::SetOption(0.1f, 1.f, 0.f);

    kOptionsMap["ci_alpha"] << Option::SetOption(1e-5f, 1.f, 0.f);
    kOptionsMap["lcb_reduction"] << Option::SetOption(0.02f, 1.f, 0.f);
    kOptionsMap["fpu_reduction"] << Option::SetOption(0.25f);
    kOptionsMap["root_fpu_reduction"] << Option::SetOption(0.25f);
    kOptionsMap["cpuct_init"] << Option::SetOption(0.5f);
    kOptionsMap["cpuct_base_factor"] << Option::SetOption(1.0f);
    kOptionsMap["cpuct_base"] << Option::SetOption(19652.f);
    kOptionsMap["cpuct_dynamic"] << Option::SetOption(true);
    kOptionsMap["cpuct_dynamic_k_factor"] << Option::SetOption(4.f);
    kOptionsMap["cpuct_dynamic_k_base"] << Option::SetOption(10000.f);
    kOptionsMap["score_utility_factor"] << Option::SetOption(0.1f);
    kOptionsMap["score_utility_div"] << Option::SetOption(20.f);
    kOptionsMap["forced_playouts_k"] << Option::SetOption(0.f);
    kOptionsMap["suppress_pass_factor"] << Option::SetOption(0.1667f, 1.f, 0.f);
    kOptionsMap["gammas_policy_factor"] << Option::SetOption(0.f, 1.f, 0.f);

    kOptionsMap["root_policy_temp"] << Option::SetOption(1.f, 100.f, 0.f);
    kOptionsMap["policy_temp"] << Option::SetOption(1.f, 100.f, 0.f);
    kOptionsMap["lag_buffer"] << Option::SetOption(0.f);
    kOptionsMap["no_cache"] << Option::SetOption(false);
    kOptionsMap["early_symm_cache"] << Option::SetOption(false);
    kOptionsMap["symm_pruning"] << Option::SetOption(false);
    kOptionsMap["use_stm_winrate"] << Option::SetOption(false);
    kOptionsMap["policy_buffer_offset"] << Option::SetOption(static_cast<int>(PolicyBufferOffset::kNormal));
    kOptionsMap["use_rollout"] << Option::SetOption(false);
    kOptionsMap["scoring_rule"] << Option::SetOption(static_cast<int>(kArea));

    // self-play options
    kOptionsMap["selfplay_query"] << Option::SetOption(std::string{});
    kOptionsMap["random_min_visits"] << Option::SetOption(1);
    kOptionsMap["random_min_ratio"] << Option::SetOption(0.f, 1.f, 0.f);
    kOptionsMap["random_moves_factor"] << Option::SetOption(0.f);
    kOptionsMap["random_moves_temp"] << Option::SetOption(1.f, 100.f, 0.f);
    kOptionsMap["random_opening_prob"] << Option::SetOption(0.f, 1.f, 0.f);
    kOptionsMap["random_opening_temp"] << Option::SetOption(1.f, 100.f, 0.f);

    kOptionsMap["gumbel_c_visit"] << Option::SetOption(50.f);
    kOptionsMap["gumbel_c_scale"] << Option::SetOption(1.f);
    kOptionsMap["gumbel_prom_visits"] << Option::SetOption(1);
    kOptionsMap["gumbel_considered_moves"] << Option::SetOption(16);
    kOptionsMap["gumbel_playouts_threshold"] << Option::SetOption(400);
    kOptionsMap["gumbel"] << Option::SetOption(false);
    kOptionsMap["always_completed_q_policy"] << Option::SetOption(false);

    kOptionsMap["dirichlet_noise"] << Option::SetOption(false);
    kOptionsMap["dirichlet_epsilon"] << Option::SetOption(0.25f);
    kOptionsMap["dirichlet_init"] << Option::SetOption(0.03f);
    kOptionsMap["dirichlet_factor"] << Option::SetOption(361.f);

    kOptionsMap["kldgain_per_node"] << Option::SetOption(0.0, 100.0, 0.0);
    kOptionsMap["kldgain_interval"] << Option::SetOption(0);

    kOptionsMap["resign_playouts"] << Option::SetOption(0);
    kOptionsMap["fastsearch_playouts"] << Option::SetOption(0);
    kOptionsMap["fastsearch_playouts_prob"] << Option::SetOption(0.f, 1.f, 0.f);
    kOptionsMap["random_fastsearch_prob"] << Option::SetOption(0.f, 1.f, 0.f);
    kOptionsMap["first_pass_bonus"] << Option::SetOption(false);
    kOptionsMap["resign_discard_prob"] << Option::SetOption(0.f, 1.f, 0.f);

    kOptionsMap["num_games"] << Option::SetOption(0);
    kOptionsMap["parallel_games"] << Option::SetOption(1);
    kOptionsMap["komi_stddev"] << Option::SetOption(0.f);
    kOptionsMap["komi_big_stddev"] << Option::SetOption(0.f);
    kOptionsMap["komi_big_stddev_prob"] << Option::SetOption(0.f, 1.f, 0.f);
    kOptionsMap["handicap_fair_komi_prob"] << Option::SetOption(0.f, 1.f, 0.f);
    kOptionsMap["target_directory"] << Option::SetOption(std::string{});

    // benchmark options
    kOptionsMap["benchmark_query"] << Option::SetOption(std::string{});
}

void ArgsParser::InitBasicParameters() const {
    PatternHashAndCoordsInit();
    Board::InitPattern3();
    Zobrist::Initialize();
    Symmetry::Get().Initialize();
    LcbEntries::Get().Initialize(GetOption<float>("ci_alpha"));
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
    const bool use_gpu = IsOptionDefault("use_gpu") ?
                             IsGpuAvailable() :
                             IsGpuAvailable() ?
                                 GetOption<bool>("use_gpu") : false;
    const int num_gpus = !use_gpu ?
                             0 :
                             already_set_specific_gpus ?
                                 GetOptionCount("gpus") : GetGpuCount();

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
            reasonable_threads = use_gpu ?
                2 * num_gpus * reasonable_batchsize_bound_per_gpu :
                reasonable_threads_bound;
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
                select_batchsize = (select_threads + (2 * num_gpus) - 1)/(2 * num_gpus);
            } else if (select_threads == 0 && select_batchsize != 0) {
                select_threads = 2 * num_gpus * select_batchsize;
            } else if (select_threads != 0 && select_batchsize == 0) {
                // No idea why somebody wants to use threads less than the number of GPUs
                // but should at least prevent hiccup.
                select_threads = std::max(select_threads, num_gpus);

                select_batchsize = (select_threads + (2 * num_gpus) - 1)/(2 * num_gpus);
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
        SetOption("root_fpu_reduction",
                      GetOption<float>("fpu_reduction"),
                      as_default);
    }

    // Set the root temperature value.
    bool already_set_root_temp = !IsOptionDefault("root_policy_temp");
    if (!already_set_root_temp) {
        bool as_default = true;
        SetOption("root_policy_temp",
                      GetOption<float>("policy_temp"),
                      as_default);
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

bool IsParameter(const std::string &param) {
    if (param.empty()) {
        return false;
    }
    return param[0] != '-';
};

std::string RemoveComment(std::string line) {
    auto out = std::string{};
    for (auto c : line) {
        if (c == '#') break;
        out += c;
    }
    return out;
}

std::string SplitterToString(Splitter &spt) {
    auto out = std::string{};
    const auto cnt = spt.GetCount();
    for (auto i = size_t{0}; i < cnt; ++i) {
        const auto res = spt.GetWord(i)->Get<>();
        out += (res + " \0"[i+1 == cnt]);
    }
    return out;
}

ArgsParser::ArgsParser(int argc, char** argv) {
    auto spt = Splitter(argc, argv);

    InitOptionsMap();
    inputs_ = std::string{};

    // Remove the name.
    const auto name = spt.RemoveWord(0);
    (void) name;

    auto config = std::string{};

    if (const auto res = spt.FindNext({"--config", "-config"})) {
        if (IsParameter(res->Get<>())) {
            config = res->Get<>();
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (!config.empty()) {
        auto file = std::ifstream{};

        file.open(config);
        if (file.is_open()) {
            auto lines = std::string{};
            auto line = std::string{};

            while(std::getline(file, line)) {
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
    SetOption("inputs", inputs_);

    DumpWarning();
    InitBasicParameters();
}

void ArgsParser::Parse(Splitter &spt) {
    const auto ErrorCommands = [](Splitter & spt) -> bool {
        const auto cnt = spt.GetCount();
        if (cnt == 0) {
            return false;
        }

        LOGGING << "Command(s) Error:" << std::endl;
        for (auto i = size_t{0}; i < cnt; ++i) {
            const auto command = spt.GetWord(i)->Get<>();
            if (!IsParameter(command)) {
                LOGGING << " " << i+1 << ". " << command << std::endl;
            }
        }
        LOGGING << " are not understood." << std::endl;
        return true;
    };

    const auto TransferHint = [](std::string hint) {
        for (auto &c : hint) {
            if (c == '+') {
                c = ' ';
            }
        }
        return hint;
    };

    const auto AcceptSet = [](
        std::string in, const std::initializer_list<std::string> list) {
        bool accept = false;
        for (auto &v: list) {
            if (in == v) accept = true;
        }
        return accept;
    };

    inputs_ += (SplitterToString(spt) + ' ');

    if (const auto res = spt.FindNext({"--mode", "-m"})) {
        if (IsParameter(res->Get<>()) &&
                AcceptSet(res->Get<>(), {"gtp", "selfplay", "benchmark"})) {
            SetOption("mode", res->Get<>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext("--timemanage")) {
        if (IsParameter(res->Get<>()) &&
                AcceptSet(res->Get<>(), {"off", "on", "fast", "keep"})) {
            if (res->Get<>() == "off") {
                SetOption("timemanage", static_cast<int>(TimeControl::TimeManagement::kOff));
            } else if (res->Get<>() == "on") {
                SetOption("timemanage", static_cast<int>(TimeControl::TimeManagement::kOn));
            } else if (res->Get<>() == "fast") {
                SetOption("timemanage", static_cast<int>(TimeControl::TimeManagement::kFast));
            } else if (res->Get<>() == "keep") {
                SetOption("timemanage", static_cast<int>(TimeControl::TimeManagement::kKeep));
            }
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext("--scoring-rule")) {
        if (IsParameter(res->Get<>()) &&
                AcceptSet(res->Get<>(), {"area", "territory"})) {
            if (res->Get<>() == "area") {
                SetOption("scoring_rule", static_cast<int>(kArea));
            } else if (res->Get<>() == "territory") {
                SetOption("scoring_rule", static_cast<int>(kTerritory));
            }
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.Find({"--help", "-h"})) {
        SetOption("help", true);
        spt.RemoveWord(res->Index());
    }

    if (const auto res = spt.Find({"--quiet", "-q"})) {
        SetOption("quiet", true);
        spt.RemoveWord(res->Index());
    }

    if (const auto res = spt.Find("--ponder")) {
        SetOption("ponder", true);
        spt.RemoveWord(res->Index());
    }

    if (const auto res = spt.Find("--reuse-tree")) {
        SetOption("reuse_tree", true);
        spt.RemoveWord(res->Index());
    }

    if (const auto res = spt.Find("--friendly-pass")) {
        SetOption("friendly_pass", true);
        spt.RemoveWord(res->Index());
    }

    if (const auto res = spt.Find("--no-cache")) {
        SetOption("no_cache", true);
        spt.RemoveWord(res->Index());
    }

    if (const auto res = spt.Find("--early-symm-cache")) {
        SetOption("early_symm_cache", true);
        spt.RemoveWord(res->Index());
    }

    if (const auto res = spt.Find("--symm-pruning")) {
        SetOption("symm_pruning", true);
        spt.RemoveWord(res->Index());
    }

    if (const auto res = spt.Find("--first-pass-bonus")) {
        SetOption("first_pass_bonus", true);
        spt.RemoveWord(res->Index());
    }

    if (const auto res = spt.Find("--use-stm-winrate")) {
        SetOption("use_stm_winrate", true);
        spt.RemoveWord(res->Index());
    }

    if (const auto res = spt.Find("--use-optimistic-policy")) {
        SetOption("policy_buffer_offset",
                  static_cast<int>(PolicyBufferOffset::kOptimistic));
        spt.RemoveWord(res->Index());
    }

    if (const auto res = spt.Find("--use-rollout")) {
        SetOption("use_rollout", true);
        spt.RemoveWord(res->Index());
    }

    if (const auto res = spt.Find("--no-winograd")) {
        SetOption("winograd", false);
        spt.RemoveWord(res->Index());
    }

    if (const auto res = spt.Find("--no-fp16")) {
        SetOption("fp16", false);
        spt.RemoveWord(res->Index());
    }

    if (const auto res = spt.Find("--capture-all-dead")) {
        SetOption("capture_all_dead", true);
        spt.RemoveWord(res->Index());
    }

    if (const auto res = spt.FindNext({"--resign-threshold", "-r"})) {
        if (IsParameter(res->Get<>())) {
            SetOption("resign_threshold", res->Get<float>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext("--kgs-hint")) {
        if (IsParameter(res->Get<>())) {
            SetOption("kgs_hint", TransferHint(res->Get<>()));
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.Find({"--analysis-verbose", "-a"})) {
        SetOption("analysis_verbose", true);
        spt.RemoveWord(res->Index());
    }

    if (const auto res = spt.Find({"--dirichlet-noise", "--noise", "-n"})) {
        SetOption("dirichlet_noise", true);
        spt.RemoveWord(res->Index());
    }

    if (const auto res = spt.FindNext("--gumbel-c-visit")) {
        if (IsParameter(res->Get<>())) {
            SetOption("gumbel_c_visit", res->Get<float>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext("--gumbel-c-scale")) {
        if (IsParameter(res->Get<>())) {
            SetOption("gumbel_c_scale", res->Get<float>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext("--gumbel-prom-visits")) {
        if (IsParameter(res->Get<>())) {
            SetOption("gumbel_prom_visits", res->Get<int>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext("--gumbel-considered-moves")) {
        if (IsParameter(res->Get<>())) {
            SetOption("gumbel_considered_moves", res->Get<int>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext("--gumbel-playouts-threshold")) {
        if (IsParameter(res->Get<>())) {
            SetOption("gumbel_playouts_threshold", res->Get<int>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.Find("--gumbel")) {
        SetOption("gumbel", true);
        spt.RemoveWord(res->Index());
    }

    if (const auto res = spt.Find("--always-completed-q-policy")) {
        SetOption("always_completed_q_policy", true);
        spt.RemoveWord(res->Index());
    }

    if (const auto res = spt.Find("--no-cpuct-dynamic")) {
        SetOption("cpuct_dynamic", false);
        spt.RemoveWord(res->Index());
    }

    if (const auto res = spt.FindNext("--dirichlet-epsilon")) {
        if (IsParameter(res->Get<>())) {
            SetOption("dirichlet_epsilon", res->Get<float>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext("--dirichlet-init")) {
        if (IsParameter(res->Get<>())) {
            SetOption("dirichlet_init", res->Get<float>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext("--dirichlet-factor")) {
        if (IsParameter(res->Get<>())) {
            SetOption("dirichlet_factor", res->Get<float>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext("--kldgain-per-node")) {
        if (IsParameter(res->Get<>())) {
            SetOption("kldgain_per_node", res->Get<double>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext("--kldgain-interval")) {
        if (IsParameter(res->Get<>())) {
            SetOption("kldgain_interval", res->Get<int>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext("--random-min-visits")) {
        if (IsParameter(res->Get<>())) {
            SetOption("random_min_visits", res->Get<int>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext("--random-min-ratio")) {
        if (IsParameter(res->Get<>())) {
            SetOption("random_min_ratio", res->Get<float>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext("--random-moves-factor")) {
        if (IsParameter(res->Get<>())) {
            SetOption("random_moves_factor", res->Get<float>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext("--random-moves-temp")) {
        if (IsParameter(res->Get<>())) {
            SetOption("random_moves_temp", res->Get<float>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext("--random-opening-prob")) {
        if (IsParameter(res->Get<>())) {
            SetOption("random_opening_prob", res->Get<float>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext("--random-opening-temp")) {
        if (IsParameter(res->Get<>())) {
            SetOption("random_opening_temp", res->Get<float>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext("--gpu-waittime")) {
        if (IsParameter(res->Get<>())) {
            SetOption("gpu_waittime", res->Get<int>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    while (const auto res = spt.FindNext({"--gpu", "-g"})) {
        if (IsParameter(res->Get<>())) {
            SetOption("gpus", res->Get<int>());
            UniqueOption("gpus");
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext({"--threads", "-t"})) {
        if (IsParameter(res->Get<>())) {
            SetOption("threads", res->Get<int>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext({"--batch-size", "-b"})) {
        if (IsParameter(res->Get<>())) {
            SetOption("batch_size", res->Get<int>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext("--cache-memory-mib")) {
        if (IsParameter(res->Get<>())) {
            SetOption("cache_memory_mib", res->Get<int>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext({"--playouts", "-p"})) {
        if (IsParameter(res->Get<>())) {
            SetOption("playouts", res->Get<int>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext("--ponder-factor")) {
       if (IsParameter(res->Get<>())) {
           SetOption("ponder_factor", res->Get<int>());
           spt.RemoveSlice(res->Index()-1, res->Index()+1);
       }
    }

    if (const auto res = spt.FindNext("--const-time")) {
        if (IsParameter(res->Get<>())) {
            SetOption("const_time", res->Get<int>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext("--virtual-loss-count")) {
        if (IsParameter(res->Get<>())) {
            SetOption("virtual_loss_count", res->Get<int>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext({"--logfile", "-l"})) {
        if (IsParameter(res->Get<>())) {
            auto fname = res->Get<>();
            LogWriter::Get().SetFilename(fname);
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext("--fixed-nn-boardsize")) {
        if (IsParameter(res->Get<>())) {
            SetOption("fixed_nn_boardsize", res->Get<int>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext({"--board-size", "-s"})) {
        if (IsParameter(res->Get<>())) {
            SetOption("defualt_boardsize", res->Get<int>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext({"--komi", "-k"})) {
        if (IsParameter(res->Get<>())) {
            SetOption("defualt_komi", res->Get<float>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext("--ci-alpha")) {
        if (IsParameter(res->Get<>())) {
            SetOption("ci_alpha", res->Get<float>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext({"--weights", "-w"})) {
        if (IsParameter(res->Get<>())) {
            SetOption("weights_file", res->Get<>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext("--weights-dir")) {
        if (IsParameter(res->Get<>())) {
            SetOption("weights_dir", res->Get<>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext("--book")) {
        if (IsParameter(res->Get<>())) {
            SetOption("book_file", res->Get<>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext("--patterns")) {
        if (IsParameter(res->Get<>())) {
            SetOption("patterns_file", res->Get<>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext("--score-utility-factor")) {
        if (IsParameter(res->Get<>())) {
            SetOption("score_utility_factor", res->Get<float>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext("--score-utility-div")) {
        if (IsParameter(res->Get<>())) {
            SetOption("score_utility_div", res->Get<float>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext("--forced-playouts-k")) {
        if (IsParameter(res->Get<>())) {
            SetOption("forced_playouts_k", res->Get<float>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext("--suppress-pass-factor")) {
        if (IsParameter(res->Get<>())) {
            SetOption("suppress_pass_factor", res->Get<float>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext("--gammas-policy-factor")) {
        if (IsParameter(res->Get<>())) {
            SetOption("gammas_policy_factor", res->Get<float>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext("--lcb-reduction")) {
        if (IsParameter(res->Get<>())) {
            SetOption("lcb_reduction", res->Get<float>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext("--fpu-reduction")) {
        if (IsParameter(res->Get<>())) {
            SetOption("fpu_reduction", res->Get<float>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext("--root-fpu-reduction")) {
        if (IsParameter(res->Get<>())) {
            SetOption("root_fpu_reduction", res->Get<float>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext("--cpuct-init")) {
        if (IsParameter(res->Get<>())) {
            SetOption("cpuct_init", res->Get<float>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext("--cpuct-base-factor")) {
        if (IsParameter(res->Get<>())) {
            SetOption("cpuct_base_factor", res->Get<float>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext("--cpuct-base")) {
        if (IsParameter(res->Get<>())) {
            SetOption("cpuct_base", res->Get<float>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext("--cpuct-dynamic-k-factor")) {
        if (IsParameter(res->Get<>())) {
            SetOption("cpuct_dynamic_k_factor", res->Get<float>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext("--cpuct-dynamic-k-base")) {
        if (IsParameter(res->Get<>())) {
            SetOption("cpuct_dynamic_k_base", res->Get<float>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext("--root-policy-temp")) {
        if (IsParameter(res->Get<>())) {
            SetOption("root_policy_temp", res->Get<float>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext("--policy-temp")) {
        if (IsParameter(res->Get<>())) {
            SetOption("policy_temp", res->Get<float>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext("--resign-discard-prob")) {
        if (IsParameter(res->Get<>())) {
            SetOption("resign_discard_prob", res->Get<float>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext("--resign-playouts")) {
        if (IsParameter(res->Get<>())) {
            SetOption("resign_playouts", res->Get<int>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext({"--reduce-playouts", "--fastsearch-playouts"})) {
        if (IsParameter(res->Get<>())) {
            SetOption("fastsearch_playouts", res->Get<int>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext({"--reduce-playouts-prob", "--fastsearch-playouts-prob"})) {
        if (IsParameter(res->Get<>())) {
            SetOption("fastsearch_playouts_prob", res->Get<float>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext("--random-fastsearch-prob")) {
        if (IsParameter(res->Get<>())) {
            SetOption("random_fastsearch_prob", res->Get<float>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext("--lag-buffer")) {
        if (IsParameter(res->Get<>())) {
            SetOption("lag_buffer", res->Get<float>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext("--num-games")) {
        if (IsParameter(res->Get<>())) {
            SetOption("num_games", res->Get<int>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext("--parallel-games")) {
        if (IsParameter(res->Get<>())) {
            SetOption("parallel_games", res->Get<int>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext("--komi-stddev")) {
        if (IsParameter(res->Get<>())) {
            SetOption("komi_stddev", res->Get<float>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext("--komi-big-stddev")) {
        if (IsParameter(res->Get<>())) {
            SetOption("komi_big_stddev", res->Get<float>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext("--komi-big-stddev-prob")) {
        if (IsParameter(res->Get<>())) {
            SetOption("komi_big_stddev_prob", res->Get<float>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext("--handicap-fair-komi-prob")) {
        if (IsParameter(res->Get<>())) {
            SetOption("handicap_fair_komi_prob", res->Get<float>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = spt.FindNext("--target-directory")) {
        if (IsParameter(res->Get<>())) {
            SetOption("target_directory", res->Get<>());
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    while (const auto res = spt.FindNext("--selfplay-query")) {
        if (IsParameter(res->Get<>())) {
            SetOption("selfplay_query", res->Get<>());
            UniqueOption("selfplay_query");
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    while (const auto res = spt.FindNext("--benchmark-query")) {
        if (IsParameter(res->Get<>())) {
            SetOption("benchmark_query", res->Get<>());
            UniqueOption("benchmark_query");
            spt.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (ErrorCommands(spt) || GetOption<bool>("help")) {
        DumpHelper();
    }
}

void ArgsParser::DumpHelper() const {
    LOGGING << "Arguments:" << std::endl
                << "\t--quiet, -q\n"
                << "\t\tDisable all diagnostic output.\n\n"

                << "\t--analysis-verbose, -a\n"
                << "\t\tPrint detailed search information.\n\n"

                << "\t--ponder\n"
                << "\t\tEnable thinking on opponent's time.\n\n"

                << "\t--reuse-tree\n"
                << "\t\tReuse part of the previous search tree for the next move.\n\n"

                << "\t--early-symm-cache\n"
                << "\t\tAccelerate the search on the opening stage.\n\n"

                << "\t--friendly-pass\n"
                << "\t\tPass when the engine believes it is winning.\n\n"

                << "\t--capture-all-dead\n"
                << "\t\tTry to remove all dead strings before pass. Only effective with area scoring.\n\n"

                << "\t--use-optimistic-policy\n"
                << "\t\tUse the optimistic policy insteal of normal policy.\n\n"

                << "\t--cache-memory-mib <integer>\n"
                << "\t\tSet the NN cache size in MiB.\n\n"

                << "\t--playouts, -p <integer>\n"
                << "\t\tThe number of maximum playouts.\n\n"

                << "\t--const-time <integer>\n"
                << "\t\tConst time of search per move in seconds.\n\n"

                << "\t--gpu, -g <integer>\n"
                << "\t\tSelect a specific GPU device. Default is all devices.\n\n"

                << "\t--threads, -t <integer>\n"
                << "\t\tThe number of threads used. Select 0 to let engine pick a reasonable default.\n\n"

                << "\t--batch-size, -b <integer>\n"
                << "\t\tThe number of batches for a single evaluation. Select 0 to let engine pick a reasonable default.\n\n"

                << "\t--lag-buffer <float>\n"
                << "\t\tSafety margin for time usage in seconds.\n\n"

                << "\t--cpuct-init <float>\n"
                << "\t\tcPUCT value for MCTS exploration.\n\n"

                << "\t--score-utility-factor <float>\n"
                << "\t\tScore-based utility factor for MCTS.\n\n"

                << "\t--lcb-reduction <float>\n"
                << "\t\tReduce the LCB weights. Select 1 to let the most visits node as the best move in MCTS.\n\n"

                << "\t--resign-threshold, -r <float>\n"
                << "\t\tResign when winrate is less than x. Default is 0.1.\n\n"

                << "\t--timemanage [off/on/fast/keep]\n"
                << "\t\tWill save the thinking time if we enable the option.\n\n"

                << "\t--scoring-rule [area/territory]\n"
                << "\t\tSelect a specific scoring rule.\n\n"

                << "\t--weights, -w <weight file name>\n"
                << "\t\tFile with network weights.\n\n"

                << "\t--book <book file name>\n"
                << "\t\tFile with opening book.\n\n"

                << "\t--logfile, -l <log file name>\n"
                << "\t\tFile to log input/output to.\n\n"
          ;
    exit(0);
}

void ArgsParser::DumpWarning() const {
    const auto friendly_pass = GetOption<bool>("friendly_pass");
    const auto capture_all_dead = GetOption<bool>("capture_all_dead");
    if (friendly_pass && capture_all_dead) {
        LOGGING << "Nonsensical options: The --capture-all-dead option may suppress the effect of --friendly-pass.\n";
    }
}