#include "data/supervised.h"
#include "data/training.h"
#include "game/game_state.h"
#include "game/sgf.h"
#include "game/types.h"
#include "neural/encoder.h"
#include "utils/log.h"

#include <fstream>
#include <cassert>

Supervised &Supervised::Get() {
    static Supervised supervised;
    return supervised;
}

void Supervised::FromSgf(std::string sgf_name, std::string out_name) {
    auto sgfs = SgfParser::Get().ChopAll(sgf_name);
    auto file = std::ofstream{};

    file.open(out_name, std::ios_base::app);

    if (!file.is_open()) {
        ERROR << "Fail to create the file: " << out_name << '!' << std::endl; 
        return;
    }

    for (auto &sgf : sgfs) {
        SgfProcess(sgf, file);
    }
}

void Supervised::SgfProcess(std::string &sgfstring, std::ostream &out_file) {
    GameState state = Sgf::Get().FormString(sgfstring, 9999);

    auto history = state.GetHistory();
    auto movelist = std::vector<int>{};

    for (const auto &board : history) {
        auto vtx = board->GetLastMove();
        if (vtx != kNullVertex) {
            movelist.emplace_back(vtx);
        }
    }

    assert(movelist.size() == history.size());

    GameState main_state;
    main_state.Reset(state.GetBoardSize(), state.GetKomi());

    const auto board_size = state.GetBoardSize();
    const auto intersections = state.GetNumIntersections();
    const auto komi = state.GetKomi();
    const auto winner = state.GetWinner();

    auto train_datas = std::vector<TrainingBuffer>{};

    const auto VertexToIndex = [](GameState &state, int vertex) -> int {
        auto x = state.GetX(vertex);
        auto y = state.GetY(vertex);
        return state.GetIndex(x, y);
    };

    for (const auto &vtx : movelist) {
        auto buf = TrainingBuffer{};

        buf.version = 0;
        buf.mode = 0;
        buf.board_size = board_size;
        buf.komi = komi;
        buf.side_to_move = main_state.GetToMove();

        buf.planes = Encoder::Get().GetPlanes(main_state);
        buf.probabilities = std::vector<float>(intersections, 0);

        buf.probabilities[VertexToIndex(main_state, vtx)] = 1.0f;

        main_state.PlayMove(vtx);

        assert(winner != kUndecide);
        if (winner == kDraw) {
            buf.result = 0;
        } else {
            buf.result = (int)winner == (int)buf.side_to_move ? 1 : -1;
        }

        train_datas.emplace_back(buf);
    }

    for (const auto &buf : train_datas) {
        // out_file << buf;
    }
}
