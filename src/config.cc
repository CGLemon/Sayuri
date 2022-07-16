#include "utils/option.h"
#include "utils/parser.h"
#include "utils/log.h"
#include "utils/mutex.h"
#include "game/zobrist.h"
#include "game/symmetry.h"
#include "game/types.h"
#include "mcts/lcb.h"
#include "config.h"

#include <limits>

std::unordered_map<std::string, Option> options_map;

#define OPTIONS_EXPASSION(T)                        \
template<>                                          \
T GetOption<T>(std::string name) {                  \
    return options_map.find(name)->second.Get<T>(); \
}                                                   \

OPTIONS_EXPASSION(std::string)
OPTIONS_EXPASSION(const char*)
OPTIONS_EXPASSION(bool)
OPTIONS_EXPASSION(int)
OPTIONS_EXPASSION(float)
OPTIONS_EXPASSION(char)

#undef OPTIONS_EXPASSION

#define OPTIONS_SET_EXPASSION(T)                    \
template<>                                          \
bool SetOption<T>(std::string name, T val) {        \
    auto res = options_map.find(name);              \
    if (res != std::end(options_map)) {             \
        res->second.Set<T>(val);                    \
        return true;                                \
    }                                               \
    return false;                                   \
}

OPTIONS_SET_EXPASSION(std::string)
OPTIONS_SET_EXPASSION(const char*)
OPTIONS_SET_EXPASSION(bool)
OPTIONS_SET_EXPASSION(int)
OPTIONS_SET_EXPASSION(float)
OPTIONS_SET_EXPASSION(char)

#undef OPTIONS_SET_EXPASSION


void InitOptionsMap() {
    options_map["help"] << Option::setoption(false);
    options_map["mode"] << Option::setoption(std::string{"gtp"});

    // engine options
    options_map["ponder"] << Option::setoption(false);
    options_map["reuse_tree"] << Option::setoption(false);
    options_map["friendly_pass"] << Option::setoption(false);
    options_map["analysis_verbose"] << Option::setoption(false);
    options_map["quiet"] << Option::setoption(false);
    options_map["rollout"] << Option::setoption(false);
    options_map["no_dcnn"] << Option::setoption(false);
    options_map["mode"] << Option::setoption(std::string{"gtp"});

    options_map["defualt_boardsize"] << Option::setoption(kDefaultBoardSize);
    options_map["defualt_komi"] << Option::setoption(kDefaultKomi);

    options_map["cache_memory_mib"] << Option::setoption(100);
    options_map["playouts"] << Option::setoption(0);
    options_map["const_time"] << Option::setoption(0);
    options_map["batch_size"] << Option::setoption(0);
    options_map["threads"] << Option::setoption(0);

    options_map["weights_file"] << Option::setoption(std::string{});
    options_map["book_file"] << Option::setoption(std::string{});

    options_map["use_gpu"] << Option::setoption(false);
    options_map["gpus"] << Option::setoption(std::string{});
    options_map["gpu_waittime"] << Option::setoption(2);

    options_map["resign_threshold"] << Option::setoption(0.1f, 1.f, 0.f);

    options_map["ci_alpha"] << Option::setoption(1e-5f, 1.f, 0.f);
    options_map["lcb_reduction"] << Option::setoption(0.02f, 1.f, 0.f);
    options_map["fpu_reduction"] << Option::setoption(0.25f);
    options_map["fpu_root_reduction"] << Option::setoption(0.25f);
    options_map["cpuct_init"] << Option::setoption(1.9f);
    options_map["cpuct_root_init"] << Option::setoption(1.9f);
    options_map["cpuct_base"] << Option::setoption(19652.f);
    options_map["cpuct_root_base"] << Option::setoption(19652.f);
    options_map["draw_factor"] << Option::setoption(0.f);
    options_map["draw_root_factor"] << Option::setoption(0.f);
    options_map["score_utility_factor"] << Option::setoption(0.05f);

    options_map["root_policy_temp"] << Option::setoption(1.f, 1.f, 0.f);
    options_map["policy_temp"] << Option::setoption(1.f, 1.f, 0.f);
    options_map["lag_buffer"] << Option::setoption(0);
    options_map["early_symm_cache"] << Option::setoption(false);

    // self-play options
    options_map["random_min_visits"] << Option::setoption(1);
    options_map["random_moves_cnt"] << Option::setoption(0);

    options_map["dirichlet_noise"] << Option::setoption(false);
    options_map["dirichlet_epsilon"] << Option::setoption(0.25f);
    options_map["dirichlet_init"] << Option::setoption(0.03f);
    options_map["dirichlet_factor"] << Option::setoption(361.f);

    options_map["forced_policy_factor"] << Option::setoption(0.f);
    options_map["cap_playouts"] << Option::setoption(0);

    options_map["num_games"] << Option::setoption(0);
    options_map["parallel_games"] << Option::setoption(1);
    options_map["komi_mean"] << Option::setoption(0.f);
    options_map["komi_variant"] << Option::setoption(0.f);
    options_map["target_directory"] << Option::setoption(std::string{});
}

void InitBasicParameters() {
    Zobrist::Initialize();
    Symmetry::Get().Initialize();
    LcbEntries::Get().Initialize(GetOption<float>("ci_alpha"));
    LogOptions::Get().SetQuiet(GetOption<bool>("quiet"));

    bool already_set_thread = GetOption<int>("threads") > 0;
    bool already_set_batchsize = GetOption<int>("batch_size") > 0;
    bool use_gpu = GetOption<bool>("use_gpu");

    const int cores = std::max((int)std::thread::hardware_concurrency(), 1);
    int select_threads = GetOption<int>("threads");
    int select_batchsize = GetOption<int>("batch_size");

    // Try to select a reasonable number for threads and batch
    // size.
    if (!already_set_thread && !already_set_batchsize) {
        select_threads = (1 + (int)use_gpu) * cores;
        select_batchsize = select_threads/2;
    } else if (!already_set_thread && already_set_batchsize) {
        if (use_gpu) {
            select_threads = 2 * select_batchsize; 
        } else {
            select_threads = cores;
        }
    } else if (already_set_thread && !already_set_batchsize) {
        select_batchsize = select_threads/2;
    }

    SetOption("threads", std::max(select_threads, 1));
    SetOption("batch_size", std::max(select_batchsize, 1));

    // Try to select a reasonable number for const time and playouts.
    bool already_set_time = GetOption<int>("const_time") > 0;
    bool already_set_playouts = GetOption<int>("playouts") > 0;

    if (!already_set_time && !already_set_playouts) {
        SetOption("const_time", 15); // 15 seconds
    }
    if (!already_set_playouts) {
        SetOption("playouts", std::numeric_limits<int>::max() / 2);
    }
}

ArgsParser::ArgsParser(int argc, char** argv) {
    auto parser = CommandParser(argc, argv);
    const auto IsParameter = [](const std::string &param) -> bool {
        if (param.empty()) {
            return false;
        }
        return param[0] != '-';
    };

    const auto ErrorCommands = [IsParameter](CommandParser & parser) -> bool {
        const auto cnt = parser.GetCount();
        if (cnt == 0) {
            return false;
        }
        int t = 1;
        LOGGING << "Command(s) Error:" << std::endl;
        for (auto i = size_t{0}; i < cnt; ++i) {
            const auto command = parser.GetCommand(i)->Get<std::string>();
            if (!IsParameter(command)) {
                LOGGING << " " << t << ". " << command << std::endl;
            }
        }
        LOGGING << " are not understood." << std::endl;
        return true;
    };

    InitOptionsMap();

    const auto name = parser.RemoveCommand(0);
    (void) name;

    if (const auto res = parser.FindNext({"--mode", "-m"})) {
        if (IsParameter(res->Get<std::string>())) {
            SetOption("mode", res->Get<std::string>());
            parser.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = parser.Find({"--help", "-h"})) {
        SetOption("help", true);
        parser.RemoveCommand(res->Index());
    }

    if (const auto res = parser.Find({"--quiet", "-q"})) {
        SetOption("quiet", true);
        parser.RemoveCommand(res->Index());
    }

    if (const auto res = parser.Find("--ponder")) {
        SetOption("ponder", true);
        parser.RemoveCommand(res->Index());
    }

    if (const auto res = parser.Find("--reuse-tree")) {
        SetOption("reuse_tree", true);
        parser.RemoveCommand(res->Index());
    }

    if (const auto res = parser.Find("--friendly-pass")) {
        SetOption("friendly_pass", true);
        parser.RemoveCommand(res->Index());
    }

    if (const auto res = parser.Find("--early-symm-cache")) {
        SetOption("early_symm_cache", true);
        parser.RemoveCommand(res->Index());
    }

    if (const auto res = parser.Find("--rollout")) {
        SetOption("rollout", true);
        parser.RemoveCommand(res->Index());
    }

    if (const auto res = parser.Find("--no-dcnn")) {
        SetOption("no_dcnn", true);
        parser.RemoveCommand(res->Index());
    }

    if (const auto res = parser.FindNext({"--resign-threshold", "-r"})) {
        if (IsParameter(res->Get<std::string>())) {
            SetOption("resign_threshold", res->Get<float>());
            parser.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = parser.Find({"--analysis-verbose", "-a"})) {
        SetOption("analysis_verbose", true);
        parser.RemoveCommand(res->Index());
    }

    if (const auto res = parser.Find({"--noise", "-n"})) {
        SetOption("dirichlet_noise", true);
        parser.RemoveCommand(res->Index());
    }

    if (const auto res = parser.FindNext("--random-moves")) {
        if (IsParameter(res->Get<std::string>())) {
            SetOption("random_moves_cnt", res->Get<int>());
            parser.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = parser.FindNext("--gpu-waittime")) {
        if (IsParameter(res->Get<std::string>())) {
            SetOption("gpu_waittime", res->Get<int>());
            parser.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    while (const auto res = parser.FindNext({"--gpu", "-g"})) {
        if (IsParameter(res->Get<std::string>())) {
            auto gpus = GetOption<std::string>("gpus");
            gpus += (res->Get<std::string>() + " ");
            SetOption("gpus", gpus);
            parser.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = parser.FindNext({"--threads", "-t"})) {
        if (IsParameter(res->Get<std::string>())) {
            SetOption("threads", res->Get<int>());
            parser.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = parser.FindNext({"--batch-size", "-b"})) {
        if (IsParameter(res->Get<std::string>())) {
            SetOption("batch_size", res->Get<int>());
            parser.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = parser.FindNext("--cache-memory-mib")) {
        if (IsParameter(res->Get<std::string>())) {
            SetOption("cache_memory_mib", res->Get<int>());
            parser.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = parser.FindNext({"--playouts", "-p"})) {
        if (IsParameter(res->Get<std::string>())) {
            SetOption("playouts", res->Get<int>());
            parser.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = parser.FindNext("--const-time")) {
        if (IsParameter(res->Get<std::string>())) {
            SetOption("const_time", res->Get<int>());
            parser.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = parser.FindNext({"--logfile", "-l"})) {
        if (IsParameter(res->Get<std::string>())) {
            auto fname = res->Get<std::string>();
            LogWriter::Get().SetFilename(fname);
            parser.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = parser.FindNext({"--board-size", "-s"})) {
        if (IsParameter(res->Get<std::string>())) {
            SetOption("defualt_boardsize", res->Get<int>());
            parser.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = parser.FindNext({"--komi", "-k"})) {
        if (IsParameter(res->Get<std::string>())) {
            SetOption("defualt_komi", res->Get<float>());
            parser.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = parser.FindNext("--ci-alpha")) {
        if (IsParameter(res->Get<std::string>())) {
            SetOption("ci_alpha", res->Get<float>());
            parser.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = parser.FindNext({"--weights", "-w"})) {
        if (IsParameter(res->Get<std::string>())) {
            SetOption("weights_file", res->Get<std::string>());
            parser.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = parser.FindNext("--book")) {
        if (IsParameter(res->Get<std::string>())) {
            SetOption("book_file", res->Get<std::string>());
            parser.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = parser.FindNext("--score-utility-factor")) {
        if (IsParameter(res->Get<std::string>())) {
            SetOption("score_utility_factor", res->Get<float>());
            parser.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = parser.FindNext("--lcb-reduction")) {
        if (IsParameter(res->Get<std::string>())) {
            SetOption("lcb_reduction", res->Get<float>());
            parser.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = parser.FindNext("--fpu-reduction")) {
        if (IsParameter(res->Get<std::string>())) {
            SetOption("fpu_reduction", res->Get<float>());
            parser.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = parser.FindNext("--fpu-root-reduction")) {
        if (IsParameter(res->Get<std::string>())) {
            SetOption("fpu_root_reduction", res->Get<float>());
            parser.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = parser.FindNext("--cpuct-init")) {
        if (IsParameter(res->Get<std::string>())) {
            SetOption("cpuct_init", res->Get<float>());
            parser.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = parser.FindNext("--cpuct-root-init")) {
        if (IsParameter(res->Get<std::string>())) {
            SetOption("cpuct_root_init", res->Get<float>());
            parser.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = parser.FindNext("--cpuct-base")) {
        if (IsParameter(res->Get<std::string>())) {
            SetOption("cpuct_base", res->Get<float>());
            parser.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = parser.FindNext("--cpuct-root-base")) {
        if (IsParameter(res->Get<std::string>())) {
            SetOption("cpuct_root_base", res->Get<float>());
            parser.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = parser.FindNext("--draw-factor")) {
        if (IsParameter(res->Get<std::string>())) {
            SetOption("draw_factor", res->Get<float>());
            parser.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = parser.FindNext("--draw-root-factor")) {
        if (IsParameter(res->Get<std::string>())) {
            SetOption("draw_root_factor", res->Get<float>());
            parser.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = parser.FindNext("--root-policy-temp")) {
        if (IsParameter(res->Get<std::string>())) {
            SetOption("root_policy_temp", res->Get<float>());
            parser.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = parser.FindNext("--policy-temp")) {
        if (IsParameter(res->Get<std::string>())) {
            SetOption("policy_temp", res->Get<float>());
            parser.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = parser.FindNext("--forced-policy-factor")) {
        if (IsParameter(res->Get<std::string>())) {
            SetOption("forced_policy_factor", res->Get<float>());
            parser.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = parser.FindNext("--cap-playouts")) {
        if (IsParameter(res->Get<std::string>())) {
            SetOption("cap_playouts", res->Get<int>());
            parser.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = parser.FindNext("--lag-buffer")) {
        if (IsParameter(res->Get<std::string>())) {
            SetOption("lag_buffer", res->Get<int>());
            parser.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = parser.FindNext("--num-games")) {
        if (IsParameter(res->Get<std::string>())) {
            SetOption("num_games", res->Get<int>());
            parser.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = parser.FindNext("--parallel-games")) {
        if (IsParameter(res->Get<std::string>())) {
            SetOption("parallel_games", res->Get<int>());
            parser.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = parser.FindNext("--komi-mean")) {
        if (IsParameter(res->Get<std::string>())) {
            SetOption("komi_mean", res->Get<float>());
            parser.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = parser.FindNext("--komi-variant")) {
        if (IsParameter(res->Get<std::string>())) {
            SetOption("komi_variant", res->Get<float>());
            parser.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = parser.FindNext("--target-directory")) {
        if (IsParameter(res->Get<std::string>())) {
            SetOption("target_directory", res->Get<std::string>());
            parser.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

#ifdef USE_CUDA
    SetOption("use_gpu", true);
#endif

    if (ErrorCommands(parser) || GetOption<bool>("help")) {
        DumpHelper();
    }

    InitBasicParameters();
}

void ArgsParser::DumpHelper() const {
    LOGGING << "Arguments:" << std::endl
                << "\t--quiet, -q\n"
                << "\t\tDisable all diagnostic verbose.\n\n" 

                << "\t--analysis-verbose, -a\n"
                << "\t\tDump the search verbose.\n\n"

                << "\t--ponder\n"
                << "\t\tThinking on opponent's time.\n\n"

                << "\t--reuse-tree\n"
                << "\t\tWill reuse the sub-tree.\n\n"

                << "\t--early-symm-cache\n"
                << "\t\tAccelerate the searching on the opening stage.\n\n"

                << "\t--friendly-pass\n"
                << "\t\tDo pass move if the engine wins the game.\n\n"

                << "\t--no-dcnn\n"
                << "\t\tDisable the Neural Network forwarding pipe. Very weak.\n\n"

                << "\t--cache-memory-mib <integer>\n"
                << "\t\tSet the NN cache size.\n\n"

                << "\t--playouts, -p <integer>\n"
                << "\t\tThe number of maximum playouts.\n\n"

                << "\t--const-time <integer>\n"
                << "\t\tConst time of search in seconds.\n\n"

                << "\t--gpu, -g <integer>\n"
                << "\t\tSelect a specific GPU device. Default is all devices.\n\n"

                << "\t--threads, -t <integer>\n"
                << "\t\tThe number of threads used. Set 0 will select a reasonable number.\n\n"

                << "\t--batch-size, -b <integer>\n"
                << "\t\tThe number of batches for a single evaluation. Set 0 will select a reasonable number.\n\n"

                << "\t--lag-buffer <integer>\n"
                << "\t\tSafety margin for time usage in seconds.\n\n"

                << "\t--score-utility-factor <float>\n"
                << "\t\tScore utility heuristic value.\n\n"

                << "\t--lcb-reduction <float>\n"
                << "\t\tReduce the LCB weights. Set 1 will select most visits node as best move in MCTS.\n\n"

                << "\t--resign-threshold, -r <float>\n"
                << "\t\tResign when winrate is less than x. Default is 0.1.\n\n"

                << "\t--weights, -w <weight file name>\n"
                << "\t\tFile with network weights.\n\n"

                << "\t--book <book file name>\n"
                << "\t\tFile with opening book.\n\n"

                << "\t--logfile, -l <log file name>\n"
                << "\t\tFile to log input/output to.\n\n"

          ;
    exit(-1);
}
