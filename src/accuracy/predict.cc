#include "accuracy/predict.h"
#include "game/sgf.h"
#include "game/iterator.h"
#include "utils/format.h"
#include "utils/log.h"

float PredictSgfAccuracy(Search &search, GameState &main_state, std::string sgf_name) {
    auto sgfs = SgfParser::Get().ChopAll(sgf_name);
    int num_positions = 0;
    int num_correct = 0;

    for (const auto &sgfstring: sgfs) {
        GameState state;
        try {
            state = Sgf::Get().FromString(sgfstring, 9999);
        } catch (const char *err) {
            LOGGING << "Fail to load the SGF file! Discard it." << std::endl
                        << Format("\tCause: %s.", err) << std::endl;
            continue;
        }

        auto game_ite = GameStateIterator(state);

        if (game_ite.MaxMoveNumber() == 0) {
            continue;
        }

        do {
            num_positions++;

            main_state = game_ite.GetState();

            const auto vertex = game_ite.GetVertex();
            const auto move = search.ThinkBestMove();

            if (vertex == move) {
                num_correct++;
            }
            if (num_positions % 1000 == 0) {
                auto current_acc = (double)num_correct/num_positions;
                LOGGING << Format("Current accuracy is %.2f%\n", current_acc * 100, num_positions);
            }

        } while (game_ite.Next());
    }

    return (double)num_correct/num_positions;
}
