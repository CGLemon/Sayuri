#include "utils/option.h"
#include "utils/parser.h"
#include "utils/log.h"
#include "game/zobrist.h"
#include "game/symmetry.h"
#include "game/types.h"
#include "config.h"

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
    options_map["quiet"] << Option::setoption(false);
    options_map["ponder"] << Option::setoption(false);
    options_map["analysis_verbose"] << Option::setoption(false);
    options_map["mode"] << Option::setoption("gtp");

    options_map["defualt_boardsize"] << Option::setoption(kDefaultBoardSize);
    options_map["defualt_komi"] << Option::setoption(kDefaultKomi);

    options_map["cache_buffer_factor"] << Option::setoption(30);
    options_map["playouts"] << Option::setoption(1600);
    options_map["batch_size"] << Option::setoption(1);
    options_map["threads"] << Option::setoption(1);

    options_map["weights_file"] << Option::setoption(std::string{});
    options_map["gpu"] << Option::setoption(-1);
    options_map["gpu_waittime"] << Option::setoption(2);

    options_map["resign_threshold"] << Option::setoption(0.1f, 1.f, 0.f);

    options_map["lcb_reduction"] << Option::setoption(0.02f);
    options_map["fpu_reduction"] << Option::setoption(0.25f);
    options_map["fpu_root_reduction"] << Option::setoption(0.25f);
    options_map["cpuct_init"] << Option::setoption(1.9f);
    options_map["cpuct_root_init"] << Option::setoption(1.9f);
    options_map["cpuct_base"] << Option::setoption(19652.f);
    options_map["cpuct_root_base"] << Option::setoption(19652.f);
    options_map["draw_factor"] << Option::setoption(0.f);
    options_map["draw_root_factor"] << Option::setoption(0.f);
    options_map["score_utility_factor"] << Option::setoption(0.05f);

    options_map["random_min_visits"] << Option::setoption(1);
    options_map["random_moves_cnt"] << Option::setoption(0);

    options_map["dirichlet_noise"] << Option::setoption(false);
    options_map["dirichlet_epsilon"] << Option::setoption(0.25f);
    options_map["dirichlet_init"] << Option::setoption(0.03f);
    options_map["dirichlet_factor"] << Option::setoption(361.f);

    options_map["forced_policy_factor"] << Option::setoption(0.f);
    options_map["cap_playouts"] << Option::setoption(0);
    options_map["lag_buffer"] << Option::setoption(0);
    options_map["early_symm_cache"] << Option::setoption(false);
}

void InitBasicParameters() {
    Zobrist::Initialize();
    Symmetry::Get().Initialize(GetOption<int>("defualt_boardsize"));
    LogOptions::Get().SetQuiet(GetOption<bool>("quiet"));
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
        ERROR << "Command(s) Error:" << std::endl;
        for (auto i = size_t{0}; i < cnt; ++i) {
            const auto command = parser.GetCommand(i)->Get<std::string>();
            if (!IsParameter(command)) {
                ERROR << " " << t << ". " << command << std::endl;
            }
        }
        ERROR << " are not understood." << std::endl;
        return true;
    };

    InitOptionsMap();

    const auto name = parser.RemoveCommand(0);
    (void) name;

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

    if (const auto res = parser.Find("--early-symm-cache")) {
        SetOption("early_symm_cache", true);
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

    if (const auto res = parser.FindNext({"--gpu", "-g"})) {
        if (IsParameter(res->Get<std::string>())) {
            SetOption("gpu", res->Get<int>());
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

    if (const auto res = parser.FindNext("--cache-buffer-factor")) {
        if (IsParameter(res->Get<std::string>())) {
            SetOption("cache_buffer_factor", res->Get<int>());
            parser.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = parser.FindNext({"--playouts", "-p"})) {
        if (IsParameter(res->Get<std::string>())) {
            SetOption("playouts", res->Get<int>());
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

    if (const auto res = parser.FindNext({"--weights", "-w"})) {
        if (IsParameter(res->Get<std::string>())) {
            SetOption("weights_file", res->Get<std::string>());
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

    if (ErrorCommands(parser) || GetOption<bool>("help")) {
        DumpHelper();
    }

    InitBasicParameters();
}

void ArgsParser::DumpHelper() const {
    ERROR << "Arguments:" << std::endl
              << "\t--quiet, -q" << std::endl
              << "\t\t Disable all diagnostic verbose." << std::endl

              << "\t--analysis-verbose, -a" << std::endl
              << "\t\t Dump the search verbose." << std::endl

              << "\t--ponder" << std::endl
              << "\t\t Thinking on opponent's time." << std::endl

              << "\t--early-symm-cache" << std::endl
              << "\t\t Accelerate search at the early step." << std::endl

              << "\t--resign-threshold, -r <float>" << std::endl
              << "\t\t Resign when winrate is less than x. Defulat is 0.1." << std::endl

              << "\t--playouts, -p <integer>" << std::endl
              << "\t\t Number of playouts." << std::endl

              << "\t--threads, -t <integer>" << std::endl
              << "\t\t Number of threads." << std::endl

              << "\t--batch-size, -b <integer>" << std::endl
              << "\t\t Number of batch size." << std::endl

              << "\t--logfile, -l <log file name>" << std::endl
              << "\t\t File to log input/output to." << std::endl

              << "\t--weights, -w <weight file name>" << std::endl
              << "\t\t File with network weights" << std::endl
        ;
    exit(-1);
}
