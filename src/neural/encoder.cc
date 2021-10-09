#include "neural/encoder.h"
#include <sstream>
#include <iomanip>

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
    auto boardsize = state.GetBoardSize();
    auto num_intersections = state.GetNumIntersections();
    auto plane_size = num_intersections * kPlaneChannels;
    auto planes = std::vector<float>(plane_size);
    auto it = std::begin(planes);

    EncoderHistoryMove(state, kHistoryMove, it);
    it += kHistoryMove * 3 * num_intersections;

    EncoderFeatures(state, it);
    it += kNumFeatures * num_intersections;

    assert(it == std::end(planes));

    Symmetry::Get().Initialize(boardsize);
    SymmetryPlanes(state, planes, symmetry);

    return planes;
}

std::string Encoder::GetPlanesString(const GameState &state, int symmetry) const {
    auto out = std::ostringstream{};
    auto boardsize = state.GetBoardSize();
    auto num_intersections = state.GetNumIntersections();

    auto planes = GetPlanes(state, symmetry);

    for (int p = 0; p < kPlaneChannels; ++p) {
        out << "plane: " << p << std::endl;
        for (int y = 0; y < boardsize; ++y) {
            for (int x = 0; x < boardsize; ++x) {
                auto idx = state.GetIndex(x, y);
                auto offset = p * num_intersections;

                if (planes[offset + idx] == 0)
                    out << std::setw(6) << 'x';
                else 
                    out << std::setw(6) << std::fixed << std::setprecision(2) << planes[offset + idx];
            }
            out << std::endl;
        }
        out << std::endl;
    }
    return out.str();
}

void Encoder::SymmetryPlanes(const GameState &state, std::vector<float> &planes, int symmetry) const {
    auto num_intersections = state.GetNumIntersections();
    auto buffer = std::vector<float>(num_intersections);
    auto planes_it = std::begin(planes);

    for (int p = 0; p < kPlaneChannels; ++p) {
        for (int index = 0; index < num_intersections; ++index) {
            auto symm_index = Symmetry::Get().TransformIndex(symmetry, index);
            buffer[index] = planes_it[symm_index];
        }

        std::copy(std::begin(buffer), std::end(buffer), planes_it);
        planes_it += num_intersections;
    }
}

void Encoder::FillColorStones(std::shared_ptr<const Board> board,
                              std::vector<float>::iterator black_it,
                              std::vector<float>::iterator white_it) const {
    auto boardsize = board->GetBoardSize();
    auto num_intersections = board->GetNumIntersections();

    for (int index = 0; index < num_intersections; ++index) {
        auto x = index % boardsize;
        auto y = index / boardsize;
        auto vtx = board->GetVertex(x, y);
        auto state = board->GetState(vtx);

        if (state == kBlack) {
            black_it[index] = static_cast<float>(true);
        } else if (state == kWhite) {
            white_it[index] = static_cast<float>(true);
        }
    }
}

void Encoder::FillMove(std::shared_ptr<const Board> board,
                       std::vector<float>::iterator move_it) const {
    auto boardsize = board->GetBoardSize();
    auto num_intersections = board->GetNumIntersections();

    auto last_move = board->GetLastMove();
    if (last_move == kNullVertex) {
        return;
    } else if (last_move == kPass || last_move == kResign) {
        for (int index = 0; index < num_intersections; ++index) {
            move_it[index] = static_cast<float>(true);
        }
    } else {
        for (int index = 0; index < num_intersections; ++index) {
            auto x = index % boardsize;
            auto y = index / boardsize;
            auto vtx = board->GetVertex(x, y);

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
        FillColorStones(board, black_it, white_it);
        FillMove(board, move_it);

        if (p != past-1) {
            black_it += 3 * num_intersections;
            white_it += 3 * num_intersections;
            move_it  += 3 * num_intersections;
        }
    }
}

void Encoder::FillKoMove(std::shared_ptr<const Board> board,
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

void Encoder::FillSafeArea(std::shared_ptr<const Board> board,
                           std::vector<float>::iterator safearea_it) const {
    auto num_intersections = board->GetNumIntersections();

    auto safe_area = std::vector<bool>(num_intersections, false);
    board->ComputePassAlive(safe_area, kBlack, true, true);
    board->ComputePassAlive(safe_area, kWhite, true, true);

    for (int index = 0; index < num_intersections; ++index) {
        if (safe_area[index]) {
            safearea_it[index] = static_cast<float>(true);
        }
    }
}

void Encoder::FillLiberties(std::shared_ptr<const Board> board,
                            std::vector<float>::iterator liberties_it) const {
    auto boardsize = board->GetBoardSize();
    auto num_intersections = board->GetNumIntersections();

    for (int index = 0; index < num_intersections; ++index) {
        auto x = index % boardsize;
        auto y = index / boardsize;
        auto vtx = board->GetVertex(x, y);
        auto liberties = board->GetLiberties(vtx);

        if (liberties == 1) {
            liberties_it[index + 0 * num_intersections] = static_cast<float>(true);
        } else if (liberties == 2) {
            liberties_it[index + 1 * num_intersections] = static_cast<float>(true);
        } else if (liberties == 3) {
            liberties_it[index + 2 * num_intersections] = static_cast<float>(true);
        } else if (liberties >= 4 && liberties <= 1024) {
            liberties_it[index + 3 * num_intersections] = static_cast<float>(true);
        }
    }
}

void Encoder::FillLadder(std::shared_ptr<const Board> board,
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

void Encoder::FillMisc(std::shared_ptr<const Board> board,
                       float komi,
                       std::vector<float>::iterator misc_it) const {
    auto num_intersections = board->GetNumIntersections();
    auto boardsize = board->GetBoardSize();
    auto side_to_move = board->GetToMove();
    if (side_to_move == kWhite) {
        komi = 0.0f - komi;
    }

    // komi
    std::fill(misc_it+ 0 * num_intersections,
                  misc_it+ 1 * num_intersections, komi/10.f);

    // board size
    std::fill(misc_it+ 1 * num_intersections,
                  misc_it+ 2 * num_intersections, float(boardsize)/10.f);

    // side to move
    if (side_to_move == kBlack) {
        std::fill(misc_it+ 2 * num_intersections,
                      misc_it+ 3 * num_intersections, static_cast<float>(true));
    } else {
        std::fill(misc_it+ 3 * num_intersections,
                      misc_it+ 4 * num_intersections, static_cast<float>(true));
    }
}

void Encoder::EncoderFeatures(const GameState &state,
                              std::vector<float>::iterator it) const {
    auto board = state.GetPastBoard(0);
    auto num_intersections = board->GetNumIntersections();

    auto ko_it = it;
    auto safearea_it = it + 1 * num_intersections;
    auto liberties_it = it + 2 * num_intersections;
    auto ladder_it = it + 6 * num_intersections;
    auto misc_it = it + 10 * num_intersections;

    FillKoMove(board, ko_it);
    FillSafeArea(board, safearea_it);
    FillLiberties(board, liberties_it);
    FillLadder(board, ladder_it);
    FillMisc(board, state.GetKomi(), misc_it);
}
