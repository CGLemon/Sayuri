#include "neural/encoder.h"
#include "utils/format.h"

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

void Encoder::FillMove(const Board* board,
                       std::vector<float>::iterator move_it) const {
    auto boardsize = board->GetBoardSize();
    auto num_intersections = board->GetNumIntersections();

    auto last_move = board->GetLastMove();
    if (last_move == kNullVertex || last_move == kPass || last_move == kResign) {
        return;
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
        FillColorStones(board.get(), black_it, white_it);
        FillMove(board.get(), move_it);

        if (p != past-1) {
            black_it += 3 * num_intersections;
            white_it += 3 * num_intersections;
            move_it  += 3 * num_intersections;
        }
    }
}

void Encoder::FillFeatures(const Board* board,
                           const int to_move,
                           std::vector<float>::iterator feat_it) const {
    auto boardsize = board->GetBoardSize();
    auto num_intersections = board->GetNumIntersections();

    for (int index = 0; index < num_intersections; ++index) {
        auto x = index % boardsize;
        auto y = index / boardsize;
        auto vtx = board->GetVertex(x, y);
        auto state = board->GetState(vtx);

        // my legal position
        if (board->IsLegalMove(vtx, to_move)) {
            feat_it[index + 0 * num_intersections] = static_cast<float>(true);
        }

        // opp's legal position
        if (board->IsLegalMove(vtx, !to_move)) {
            feat_it[index + 1 * num_intersections] = static_cast<float>(true);
        }

        if (board->IsCorner(vtx) && state == to_move) {
            feat_it[index + 2 * num_intersections] = static_cast<float>(true);
        } else if (board->IsCorner(vtx) && state == !to_move) {
            feat_it[index + 3 * num_intersections] = static_cast<float>(true);
        } else if (board->IsEdge(vtx) && state == to_move) {
            feat_it[index + 4 * num_intersections] = static_cast<float>(true);
        } else if (board->IsEdge(vtx) && state == !to_move) {
            feat_it[index + 5 * num_intersections] = static_cast<float>(true);
        }

        if (board->IsThreatPass(vtx, to_move)) {
            feat_it[index + 6 * num_intersections] = static_cast<float>(true);
        }

        // feat_it[index + 7 * num_intersections];
        // feat_it[index + 8 * num_intersections];
        // feat_it[index + 9 * num_intersections];
    }
}

void Encoder::FillMisc(const Board* board,
                       const int to_move,
                       float komi,
                       std::vector<float>::iterator misc_it) const {
    auto num_intersections = board->GetNumIntersections();

    if (to_move == kWhite) {
        komi = 0.0f - komi;
    }

    // komi
    std::fill(misc_it+ 0 * num_intersections,
                  misc_it+ 1 * num_intersections, komi/20.f);

    // negative komi
    std::fill(misc_it+ 1 * num_intersections,
                  misc_it+ 2 * num_intersections, -komi/20.f);

    // number of intersections
    std::fill(misc_it+ 2 * num_intersections,
                  misc_it+ 3 * num_intersections, static_cast<float>(num_intersections)/361.f);

    // ones
    std::fill(misc_it+ 3 * num_intersections,
                  misc_it+ 4 * num_intersections, static_cast<float>(true));
}

void Encoder::EncoderFeatures(const GameState &state,
                              std::vector<float>::iterator it) const {
    auto board = state.GetPastBoard(0);
    auto num_intersections = board->GetNumIntersections();

    auto feat_it = it + 0 * num_intersections;
    auto misc_it = it + 10 * num_intersections;

    FillFeatures(board.get(), state.GetToMove(), feat_it);
    FillMisc(board.get(), state.GetToMove(), state.GetKomi(), misc_it);
}
