#include "summary/accuracy.h"
#include "game/sgf.h"
#include "game/iterator.h"
#include "utils/format.h"
#include "utils/log.h"
#include "utils/random.h"

#include <random>
#include <stdexcept>

AccuracyReport ComputeNetAccuracy(Network &network,
                                  std::string sgf_name) {
    AccuracyReport report;
    auto sgfs = SgfParser::Get().ChopAll(sgf_name);

    std::shuffle(std::begin(sgfs),
                     std::end(sgfs),
                     Random<>::Get());

    for (const auto &sgfstring: sgfs) {
        GameState state;
        try {
            state = Sgf::Get().FromString(sgfstring, 9999);
        } catch (const std::exception& e) {
            LOGGING << "Fail to load the SGF file! Discard it." << std::endl
                        << Format("\tCause: %s.", e.what()) << std::endl;
            continue;
        }
        auto game_ite = GameStateIterator(state);

        if (game_ite.MaxMoveNumber() == 0) {
            continue;
        }

        do {
            report.num_positions++;
            auto main_state = game_ite.GetState();

            const auto vertex = game_ite.GetVertex();
            const auto move = network.GetVertexWithPolicy(
                                  main_state, 0.001f, true);
            if (vertex == move) {
                report.num_matched++;
            }
            if (report.num_positions % 1000 == 0) {
                LOGGING << Format("Current accuracy is %.2f%\n",
                    report.GetAccuracy() * 100, report.num_positions);
            }
        } while (game_ite.Next());
    }
    return report;
}
