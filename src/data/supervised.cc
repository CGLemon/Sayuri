#include "data/supervised.h"
#include "data/training.h"
#include "game/game_state.h"
#include "game/sgf.h"
#include "utils/log.h"

#include <fstream>

void Supervised::FromSgf(std::string sgf_name, std::string out_name) {
    auto sgfs = SgfParser::Get().ChopAll(sgf_name);
    auto file = std::ofstream{};

    file.open(out_name, std::ios_base::app);

    if (!file.is_open()) {
        ERROR << "Fail to create the file: " << out_name << '!' << std::endl; 
        return;
    }

    for (const auto &sgf : sgfs) {
        GameState state = Sgf::Get().FormString(sgf, 9999);
        auto buf = TrainingBuffer{};
        file << buf;
    }
}
