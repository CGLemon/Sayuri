#include "data/supervised.h"
#include "data/training.h"
#include "game/game_state.h"
#include "game/sgf.h"
#include "game/types.h"
#include "neural/encoder.h"
#include "utils/log.h"

#include <fstream>
#include <cassert>

#include <cmath>
#include "utils/log.h"

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

    int games = 0;
    for (auto &sgf : sgfs) {
        if (++games % 200 == 0) {
            LOGGING << "Process games: " << games << std::endl;
        }
        SgfProcess(sgf, file);
    }
}

void Supervised::SgfProcess(std::string &sgfstring, std::ostream &out_file) {
    GameState state = Sgf::Get().FormString(sgfstring, 9999);

    auto ownership = state.GetOwnership(200);

    auto success = true;
    auto black_score_on_board = 0;
    const auto board_size = state.GetBoardSize();

    for (int y = 0; y < board_size; ++y) {
        for (int x = 0; x < board_size; ++x) {
            auto index = state.GetIndex(x, board_size-y-1);
            auto owner = ownership[index];

            if (owner == kBlack) {
                black_score_on_board += 1;
            } else if (owner == kWhite) {
                black_score_on_board -= 1;
            } else if (owner == kInvalid) {
                success = false;
            }
        }
    }

    const auto black_final_score = (float)black_score_on_board - state.GetKomi();
    auto game_winner = kUndecide;

    if (std::abs(black_final_score) < 1e-4) {
        game_winner = kDraw;
    } else if (black_final_score > 0) {
        game_winner = kBlackWon;
    } else if (black_final_score < 0) {
        game_winner = kWhiteWon;
    }

    (void) success;

    auto history = state.GetHistory();
    auto movelist = std::vector<int>{};

    for (const auto &board : history) {
        auto vtx = board->GetLastMove();
        if (vtx != kNullVertex) {
            movelist.emplace_back(vtx);
        }
    }

    assert(movelist.size() == history.size()-1);

    GameState main_state;
    main_state.Reset(state.GetBoardSize(), state.GetKomi());

    const auto num_intersections = state.GetNumIntersections();
    const auto komi = state.GetKomi();
    const auto winner = game_winner;

    auto train_datas = std::vector<Training>{};

    const auto VertexToIndex = [](GameState &state, int vertex) -> int {
        if (vertex == kPass) {
            return state.GetNumIntersections();
        }

        auto x = state.GetX(vertex);
        auto y = state.GetY(vertex);
        return state.GetIndex(x, y);
    };

    for (auto i = size_t{0}; i < movelist.size(); ++i) {
        auto vtx = movelist[i];
        auto aux_vtx = kPass;

        if (i != movelist.size()-1) {
            aux_vtx = movelist[i+1];
        }

        auto buf = Training{};

        buf.version = GetTrainigVersion();
        buf.mode = GetTrainigMode();
        buf.board_size = board_size;
        buf.komi = komi;
        buf.side_to_move = main_state.GetToMove();

        buf.planes = Encoder::Get().GetPlanes(main_state);

        buf.probabilities = std::vector<float>(num_intersections+1, 0);
        buf.auxiliary_probabilities = std::vector<float>(num_intersections+1, 0);
        buf.ownership = std::vector<int>(num_intersections, 0);

        buf.probabilities_index = VertexToIndex(main_state, vtx);
        buf.probabilities[VertexToIndex(main_state, vtx)] = 1.0f;

        buf.auxiliary_probabilities_index = VertexToIndex(main_state, aux_vtx);
        buf.auxiliary_probabilities[VertexToIndex(main_state, aux_vtx)] = 1.0f;

        for (int idx = 0; idx < num_intersections; ++idx) {
            if (ownership[idx] == buf.side_to_move) {
                buf.ownership[idx] = 1; 
            } else if (ownership[idx] == !buf.side_to_move) {
                buf.ownership[idx] = -1;
            }
        }

        assert(winner != kUndecide);
        if (winner == kDraw) {
            buf.final_score = 0;
            buf.result = 0;
        } else {
            buf.result = (int)winner == (int)buf.side_to_move ? 1 : -1;
            buf.final_score = buf.side_to_move == kBlack ? black_final_score : -black_final_score;
        }

        main_state.PlayMove(vtx);
        train_datas.emplace_back(buf);
    }

    for (const auto &buf : train_datas) {
        buf.StreamOut(out_file);
    }
}
