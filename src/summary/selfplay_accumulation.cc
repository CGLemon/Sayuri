#include "summary/selfplay_accumulation.h"
#include "game/game_state.h"
#include "game/sgf.h"
#include "utils/format.h"
#include "utils/log.h"
#include "utils/random.h"

#include <random>
#include <sstream>

SelfplayReport ComputeSelfplayAccumulation(std::string sgf_name) {
    SelfplayReport report;
    auto sgfs = SgfParser::Get().ChopAll(sgf_name);

    std::shuffle(std::begin(sgfs),
                     std::end(sgfs),
                     Random<>::Get());

    for (const auto &sgfstring: sgfs) {
        GameState state;
        try {
            state = Sgf::Get().FromString(sgfstring, 9999);
        } catch (const char *err) {
            LOGGING << "Fail to load the SGF file! Discard it." << std::endl
                        << Format("\tCause: %s.", err) << std::endl;
            continue;
        }

        const auto move_num = state.GetMoveNumber();
        report.num_games++;

        for (int i = 0; i <= move_num; ++i) {
            auto line = state.GetComment(i);

            if (i == 0) {
                // root
            } else {
                for (auto &c : line) {
                    if (c == ',') {
                        c = ' ';
                    }
                }

                int playouts;
                std::istringstream iss{line};

                if (iss >> playouts) {
                    // Search move.
                    report.accm_playouts += playouts;
                } else {
                    // Fast policy opening move.
                    report.accm_playouts += 1;
                }
            }
        }

        if (report.num_games % 1000 == 0) {
            LOGGING << Format(
                "games %d -> accum %zu, avg %.2f(p/game)",
                report.num_games,
                report.accm_playouts,
                report.GetAccumulationPlayoutsPerGame())
                    << std::endl;
        }
    }
    return report;
}
