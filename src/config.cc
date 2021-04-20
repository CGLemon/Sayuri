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
    options_map["name"] << Option::setoption(kProgram);
    options_map["version"] << Option::setoption(kVersion);
    options_map["help"] << Option::setoption(false);

    options_map["mode"] << Option::setoption("gtp");

    options_map["playouts"] << Option::setoption(1600);
    options_map["threads"] << Option::setoption(1);

    options_map["boardsize"] << Option::setoption(kDefaultBoardSize);
    options_map["komi"] << Option::setoption(kDefaultKomi);

    options_map["weights_file"] << Option::setoption(kNoWeightsFile);
    options_map["log_file"] << Option::setoption(kNologFile);
}

void InitBasicParameters() {
    Zobrist::Initialize();
    Symmetry::Initialize(GetOption<int>("boardsize"));
}

ArgsParser::ArgsParser(int argc, char** argv) {
    auto parser = CommandParser(argc, argv);
    const auto is_parameter = [](const std::string &param) -> bool {
        if (param.empty()) {
            return false;
        }
        return param[0] != '-';
    };

    const auto error_commands = [is_parameter](CommandParser & parser) -> bool {
        const auto cnt = parser.GetCount();
        if (cnt == 0) {
            return false;
        }
        int t = 1;
        ERROR << "Command(s) Error:" << std::endl;
        for (auto i = size_t{0}; i < cnt; ++i) {
            const auto command = parser.GetCommand(i)->Get<std::string>();
            if (!is_parameter(command)) {
                ERROR << " " << t << command << std::endl;
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

    if (const auto res = parser.Find({"--playouts", "-h"})) {
        SetOption("playouts", true);
        parser.RemoveCommand(res->Index());
    }

    if (error_commands(parser)) {
        Helper();
    }

    Dump();
    InitBasicParameters();
}

void ArgsParser::Helper() const {
    ERROR << "Arguments:" << std::endl
              << "  --help, -h" << std::endl
              << "  --mode, -m [gtp]" << std::endl
              << "  --playouts, -p <integer>" << std::endl
              << "  --threads, -t <integer>" << std::endl
              << "  --weights, -w <weight file name>" << std::endl;
    exit(-1);
}

void ArgsParser::Dump() const {
    if (GetOption<bool>("help")) {
        Helper();
    }
}
