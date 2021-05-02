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

    options_map["defualt_boardsize"] << Option::setoption(kDefaultBoardSize);
    options_map["defualt_komi"] << Option::setoption(kDefaultKomi);

    options_map["playouts"] << Option::setoption(1600);
    options_map["threads"] << Option::setoption(1);

    options_map["weights_file"] << Option::setoption(kNoWeightsFile);
}

void InitBasicParameters() {
    Zobrist::Initialize();
    Symmetry::Initialize(GetOption<int>("defualt_boardsize"));
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

    if (const auto res = parser.Find({"--playouts", "-p"})) {
        SetOption("playouts", true);
        parser.RemoveCommand(res->Index());
    }

    if (const auto res = parser.FindNext({"--logfile", "-l"})) {
        if (IsParameter(res->Get<std::string>())) {
            auto fname = res->Get<std::string>();
            LogWriter::Get().SetFilename(fname);
            parser.RemoveSlice(res->Index()-1, res->Index()+1);
        }
    }

    if (const auto res = parser.FindNext({"--boardsize", "-b"})) {
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

    if (ErrorCommands(parser)) {
        Helper();
    }

    Dump();
    InitBasicParameters();
}

void ArgsParser::Helper() const {
    ERROR << "Arguments:" << std::endl
              << "\t--help, -h" << std::endl
              << "\t--mode, -m [gtp]" << std::endl
              << "\t--playouts, -p <integer>" << std::endl
              << "\t--threads, -t <integer>" << std::endl
              << "\t--logfile, -l <log file name>" << std::endl
              << "\t--weights, -w <weight file name>" << std::endl;
    exit(-1);
}

void ArgsParser::Dump() const {
    if (GetOption<bool>("help")) {
        Helper();
    }
}
