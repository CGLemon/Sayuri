#include "neural/encoder.h"
#include "utils/format.h"

#include <sstream>
#include <iomanip>
#include <iostream>

Encoder& Encoder::Get() {
    static Encoder encoder;
    return encoder;
}

InputData Encoder::GetInputs(const GameState &state, int symmetry) const {
    auto data = InputData{};

    data.board_size = state.GetBoardSize();
    data.side_to_move = state.GetToMove();
    data.komi = state.GetKomi();

    auto planes = GetPlanes(state, symmetry);
    auto plane_size = planes.size();
    auto it = std::begin(planes);

    std::copy(it, it + plane_size,
                  std::begin(data.planes));

    return data;
}

std::vector<float> Encoder::GetPlanes(const GameState &state, int symmetry) const {
    auto num_intersections = state.GetNumIntersections();
    auto plane_size = num_intersections * kPlaneChannels;
    auto planes = std::vector<float>(plane_size, 0.f);
    auto it = std::begin(planes);

    EncoderHistoryMove(state, kHistoryMoves, it);
    it += kHistoryMoves * 3 * num_intersections;

    EncoderFeatures(state, it);
    it += kNumFeatures * num_intersections;

    assert(it == std::end(planes));

    SymmetryPlanes(state, planes, symmetry);

    return planes;
}

std::string Encoder::GetPlanesString(const GameState &state, int symmetry) const {
    auto out = std::ostringstream{};
    auto boardsize = state.GetBoardSize();
    auto num_intersections = state.GetNumIntersections();

    auto planes = GetPlanes(state, symmetry);

    for (int p = 0; p < kPlaneChannels; ++p) {
        out << "plane: " << (p+1) << std::endl;
        for (int y = 0; y < boardsize; ++y) {
            for (int x = 0; x < boardsize; ++x) {
                auto idx = state.GetIndex(x, y);
                auto offset = p * num_intersections;

                if (std::abs(planes[offset + idx]) < std::abs(1e-4))
                    out << Format("%6c", 'x');
                else
                    out << Format("%6.2f", planes[offset + idx]);
            }
            out << std::endl;
        }
    }
    return out.str();
}

void Encoder::SymmetryPlanes(const GameState &state, std::vector<float> &planes, int symmetry) const {
    auto boardsize = state.GetBoardSize();
    auto num_intersections = state.GetNumIntersections();
    auto buffer = std::vector<float>(num_intersections);
    auto planes_it = std::begin(planes);

    for (int p = 0; p < kPlaneChannels; ++p) {
        for (int index = 0; index < num_intersections; ++index) {
            auto symm_index = Symmetry::Get().TransformIndex(boardsize, symmetry, index);
            buffer[index] = planes_it[symm_index];
        }

        std::copy(std::begin(buffer), std::end(buffer), planes_it);
        planes_it += num_intersections;
    }
}

void Encoder::FillColorStones(const Board* board,
                              std::vector<float>::iterator black_it,
                              std::vector<float>::iterator white_it) const {
    auto num_intersections = board->GetNumIntersections();

    for (int index = 0; index < num_intersections; ++index) {
        auto vtx = board->IndexToVertex(index);
        auto state = board->GetState(vtx);

        if (state == kBlack) {
            black_it[index] = static_cast<float>(true);
        } else if (state == kWhite) {
            white_it[index] = static_cast<float>(true);
        }
    }
}

void Encoder::FillMove(const Board* board,
                       std::vector<float>::iterator move_it) const {
    auto num_intersections = board->GetNumIntersections();

    auto last_move = board->GetLastMove();
    if (last_move == kNullVertex || last_move == kPass || last_move == kResign) {
        return;
    } else {
        for (int index = 0; index < num_intersections; ++index) {
            auto vtx = board->IndexToVertex(index);

            if (vtx == last_move) {
                move_it[index] = static_cast<float>(true);
            }
        }
    }
}

void Encoder::EncoderHistoryMove(const GameState &state,
                                 int counter,
                                 std::vector<float>::iterator it) const {
    auto move_num = state.GetMoveNumber();
    auto past = std::min(move_num+1, counter);

    auto num_intersections = state.GetNumIntersections();

    auto black_to_move = state.GetToMove() == kBlack;

    auto black_it = it + (black_to_move ? 0 : num_intersections);
    auto white_it = it + (black_to_move ? num_intersections : 0);
    auto move_it  = it + 2 * num_intersections;

    for (auto p = 0; p < past; ++p) {
        auto board = state.GetPastBoard(p);
        FillColorStones(board.get(), black_it, white_it);
        FillMove(board.get(), move_it);

        if (p != past-1) {
            black_it += 3 * num_intersections;
            white_it += 3 * num_intersections;
            move_it  += 3 * num_intersections;
        }
    }
}

void Encoder::FillKoMove(const Board* board,
                         std::vector<float>::iterator ko_it) const {
    auto ko_move = board->GetKoMove();

    if (ko_move == kNullVertex) {
        return;
    }

    auto x = board->GetX(ko_move);
    auto y = board->GetY(ko_move);
    auto index = board->GetIndex(x, y);
    ko_it[index] = static_cast<float>(true);
}

void Encoder::FillArea(const Board* board,
                       const int to_move,
                       const int scoring,
                       std::vector<float>::iterator area_it) const {
    if (scoring == kTerritory) {
        return;
    }
    auto num_intersections = board->GetNumIntersections();

    auto ownership = std::vector<int>(num_intersections, kInvalid);
    auto safe_area = std::vector<bool>(num_intersections, false);
    auto virtual_helper = std::vector<int>(num_intersections, kEmpty);

    board->ComputeScoreArea(ownership, kArea, virtual_helper);
    board->ComputeSafeArea(safe_area, false);

    for (int index = 0; index < num_intersections; ++index) {
        bool safe = safe_area[index];
        int owner = ownership[index];

        if (safe) {
            if (owner == to_move) {
                area_it[index + 0 * num_intersections] = static_cast<float>(true);
            } else if (owner == (!to_move)) {
                area_it[index + 1 * num_intersections] = static_cast<float>(true);
            }
        }
        if (owner == to_move) {
            area_it[index + 2 * num_intersections] = static_cast<float>(true);
        } else if (owner == (!to_move)) {
            area_it[index + 3 * num_intersections] = static_cast<float>(true);
        }
    }
}

void Encoder::FillLiberties(const Board* board,
                            std::vector<float>::iterator liberties_it) const {
    auto num_intersections = board->GetNumIntersections();

    for (int index = 0; index < num_intersections; ++index) {
        auto vtx = board->IndexToVertex(index);
        auto state = board->GetState(vtx);

        if (state == kBlack || state == kWhite) {
            auto liberties = board->GetLiberties(vtx);

            if (liberties == 1) {
                liberties_it[index + 0 * num_intersections] = static_cast<float>(true);
            } else if (liberties == 2) {
                liberties_it[index + 1 * num_intersections] = static_cast<float>(true);
            } else if (liberties == 3) {
                liberties_it[index + 2 * num_intersections] = static_cast<float>(true);
            } else if (liberties == 4) {
                liberties_it[index + 3 * num_intersections] = static_cast<float>(true);
            }
        }
    }
}

void Encoder::FillLadder(const Board* board,
                         std::vector<float>::iterator ladder_it) const {
    auto num_intersections = board->GetNumIntersections();
    auto ladders = board->GetLadderMap();

    for (int index = 0; index < num_intersections; ++index) {
        auto ladder = ladders[index];

        if (ladder == kLadderDeath) {
            ladder_it[index + 0 * num_intersections] = static_cast<float>(true);
        } else if (ladder == kLadderEscapable) {
            ladder_it[index + 1 * num_intersections] = static_cast<float>(true);
        } else if (ladder == kLadderAtari) {
            ladder_it[index + 2 * num_intersections] = static_cast<float>(true);
        } else if (ladder == kLadderTake) {
            ladder_it[index + 3 * num_intersections] = static_cast<float>(true);
        }
    }
}

void Encoder::FillMisc(const Board* board,
                       const int to_move,
                       const int scoring, float wave, float komi,
                       std::vector<float>::iterator misc_it) const {
    auto num_intersections = board->GetNumIntersections();

    if (to_move == kWhite) {
        komi = 0.0f - komi;
    }

    // scoring rule
    float scoring_val = scoring == kArea ? 0.f : 1.f;
    std::fill(misc_it+ 0 * num_intersections,
                  misc_it+ 1 * num_intersections, scoring_val);

    // wave
    std::fill(misc_it+ 1 * num_intersections,
                  misc_it+ 2 * num_intersections, wave);

    // komi
    std::fill(misc_it+ 2 * num_intersections,
                  misc_it+ 3 * num_intersections, komi/20.f);

    // negative komi
    std::fill(misc_it+ 3 * num_intersections,
                  misc_it+ 4 * num_intersections, -komi/20.f);

    // number of intersections
    std::fill(misc_it+ 4 * num_intersections,
                  misc_it+ 5 * num_intersections, static_cast<float>(num_intersections)/361.f);

    // ones
    std::fill(misc_it+ 5 * num_intersections,
                  misc_it+ 6 * num_intersections, static_cast<float>(true));
}

void Encoder::EncoderFeatures(const GameState &state,
                              std::vector<float>::iterator it) const {
    auto board = state.GetPastBoard(0);
    const auto shift = board->GetNumIntersections();

    auto ko_it        = it +  0 * shift; // 1p, ko move
    auto area_it      = it +  1 * shift; // 4p, pass-alive and pass-dead area
    auto liberties_it = it +  5 * shift; // 4p, strings with 1, 2, 3 and 4 liberties
    auto ladder_it    = it +  9 * shift; // 4p, ladder features
    auto misc_it      = it + 13 * shift; // 6p, others

    auto color = state.GetToMove();
    auto scoring = state.GetScoringRule();

    FillKoMove(board.get(), ko_it);
    FillArea(board.get(), color, scoring, area_it);
    FillLiberties(board.get(), liberties_it);
    FillLadder(board.get(), ladder_it);
    FillMisc(board.get(),
                 color,
                 scoring,
                 state.GetWave(),
                 state.GetKomiWithPenalty(),
                 misc_it);
}
