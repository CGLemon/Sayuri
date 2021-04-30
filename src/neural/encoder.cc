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

    Symmetry::Initialize(boardsize);

    auto plane_size = num_intersections * kPlaneChannels;
    auto planes = std::vector<float>(plane_size);
    auto it = std::begin(planes);

    EncoderHistoryMove(state, kHistoryMove, it, symmetry);
    it += kHistoryMove * 3 * num_intersections;

    EncoderFeatures(state, it, symmetry);
    it += kNumFeatures * num_intersections;

    assert(it == std::end(planes));

    return planes;
}

std::string Encoder::GetPlanesString(const GameState &state, int symmetry) const {
    auto out = std::ostringstream{};
    auto boardsize = state.GetBoardSize();
    auto num_intersections = state.GetNumIntersections();

    auto planes = GetPlanes(state, symmetry);

    for (int p = 0; p < kInputChannels; ++p) {
        out << "plane: " << p << std::endl;
        for (int y = 0; y < boardsize; ++y) {
            for (int x = 0; x < boardsize; ++x) {
                auto idx = state.GetIndex(x, y);
                auto offset = p * num_intersections;
                out << std::setw(7) << std::fixed << std::setprecision(4) << planes[offset + idx];
            }
            out << std::endl;
        }
        out << std::endl;
    }
    return out.str();
}

void Encoder::FillColorStones(std::shared_ptr<const Board> board,
                              std::vector<float>::iterator black_it,
                              std::vector<float>::iterator white_it,
                             int symmetry) const {
    auto boardsize = board->GetBoardSize();
    auto num_intersections = board->GetNumIntersections();

    for (int index = 0; index < num_intersections; ++index) {
        auto symm_index = Symmetry::Get().TransformIndex(symmetry, index);
        auto x = symm_index % boardsize;
        auto y = symm_index / boardsize;
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
                       std::vector<float>::iterator move_it, int symmetry) const {
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
            auto symm_index = Symmetry::Get().TransformIndex(symmetry, index);
            auto x = symm_index % boardsize;
            auto y = symm_index / boardsize;
            auto vtx = board->GetVertex(x, y);

            if (vtx == last_move) {
                move_it[index] = static_cast<float>(true);
            }
        }
    }
}

void Encoder::EncoderHistoryMove(const GameState &state,
                                 int counter,
                                 std::vector<float>::iterator it,
                                 int symmetry) const {
    auto move_num = state.GetMoveNumber();
    auto past = std::min(move_num+1, counter);

    auto num_intersections = state.GetNumIntersections();

    auto black_to_move = state.GetToMove() == kBlack;

    auto black_it = it + (black_to_move ? 0 : num_intersections);
    auto white_it = it + (black_to_move ? num_intersections : 0);
    auto move_it  = it + 2 * num_intersections;

    for (auto p = 0; p < past; ++p) {
        auto board = state.GetPastBoard(p);
        FillColorStones(board, black_it, white_it, symmetry);
        FillMove(board, move_it, symmetry);

        if (p != past-1) {
            black_it += 3 * num_intersections;
            white_it += 3 * num_intersections;
            move_it  += 3 * num_intersections;
        }
    }
}

void Encoder::FillKoMove(std::shared_ptr<const Board> board,
                         std::vector<float>::iterator ko_it, int symmetry) const {
    auto boardsize = board->GetBoardSize();
    auto num_intersections = board->GetNumIntersections();

    auto ko_move = board->GetKoMove();

    if (ko_move == kNullVertex) {
        return;
    }

    for (int index = 0; index < num_intersections; ++index) {
        auto symm_index = Symmetry::Get().TransformIndex(symmetry, index);
        auto x = symm_index % boardsize;
        auto y = symm_index / boardsize;
        auto vtx = board->GetVertex(x, y);
        if (vtx == ko_move) {
            ko_it[index] = static_cast<float>(true);
        }
    }
}

void Encoder::FillCaptureMove(std::shared_ptr<const Board> board,
                              std::vector<float>::iterator capture_it, int symmetry) const {
    auto boardsize = board->GetBoardSize();
    auto num_intersections = board->GetNumIntersections();
    auto color = board->GetToMove();

    for (int index = 0; index < num_intersections; ++index) {
        auto symm_index = Symmetry::Get().TransformIndex(symmetry, index);
        auto x = symm_index % boardsize;
        auto y = symm_index / boardsize;
        auto vtx = board->GetVertex(x, y);
        auto state = board->GetState(vtx);

        if (board->IsCaptureMove(vtx, color) && state == kEmpty) {
            capture_it[index] = static_cast<float>(true);
        }
    }
}

void Encoder::FillLiberties(std::shared_ptr<const Board> board,
                            std::vector<float>::iterator liberties_it, int symmetry) const {
    auto boardsize = board->GetBoardSize();
    auto num_intersections = board->GetNumIntersections();

    for (int index = 0; index < num_intersections; ++index) {
        auto symm_index = Symmetry::Get().TransformIndex(symmetry, index);
        auto x = symm_index % boardsize;
        auto y = symm_index / boardsize;
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
                         std::vector<float>::iterator ladder_it, int symmetry) const {
    auto num_intersections = board->GetNumIntersections();
    auto ladders = board->GetLadderPlane();

    for (int index = 0; index < num_intersections; ++index) {
        auto symm_index = Symmetry::Get().TransformIndex(symmetry, index);
        auto ladder = ladders[symm_index];

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

void Encoder::FillSideToMove(std::shared_ptr<const Board> board,
                             std::vector<float>::iterator color_it) const {
    auto num_intersections = board->GetNumIntersections();
    auto side_to_move = board->GetToMove();

    auto offset = side_to_move == kBlack ? 0 : num_intersections;

    for (int index = 0; index < num_intersections; ++index) {
        color_it[offset + index] = static_cast<float>(true);
    }
}

void Encoder::EncoderFeatures(const GameState &state,
                              std::vector<float>::iterator it, int symmetry) const {
    auto board = state.GetPastBoard(0);
    auto num_intersections = board->GetNumIntersections();

    auto ko_it = it;
    auto capture_it = it + 1 * num_intersections;
    auto liberties_it = it + 2 * num_intersections;
    auto ladder_it = it + 6 * num_intersections;
    auto color_it = it + 10 * num_intersections;


    FillKoMove(board, ko_it, symmetry);
    FillCaptureMove(board, capture_it, symmetry);
    FillLiberties(board, liberties_it, symmetry);
    FillLadder(board, ladder_it, symmetry);
    FillSideToMove(board, color_it);
}




