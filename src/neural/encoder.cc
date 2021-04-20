#include "neural/encoder.h"

Encoder& Encoder::Get() {
    static Encoder encoder;
    return encoder;
}

InputData Encoder::GetInputs(GameState &state, int symmetry) const {
    auto data = InputData{};
    auto boardsize = state.GetBoardSize();
    int num_intersections = state.GetNumIntersections();

    Symmetry::Initialize(boardsize);
    data.board_size = boardsize;

    auto plane_size = num_intersections * kPlaneChannels;
    auto planes = std::vector<float>(num_intersections * kPlaneChannels);
    auto it = std::begin(planes);
    EncoderHistoryMove(state, it, symmetry);

    std::copy(it, it + plane_size,
              std::begin(data.planes));

    return data;
}

void Encoder::FillColorStones(std::shared_ptr<const Board> board,
                              std::vector<float>::iterator it,
                              int color, int symmetry) const {

    int boardsize = board->GetBoardSize();
    int num_intersections = board->GetNumIntersections();

    for (int index = 0; index < num_intersections; ++index) {
        auto symm_index = Symmetry::Get().TransformIndex(symmetry, index);
        auto x = symm_index % boardsize;
        auto y = symm_index / boardsize;
        auto vtx = board->GetVertex(x, y);
        auto state = board->GetState(vtx);
        if (state == color) {
            it[index] = static_cast<float>(true);
        }
    }
}

void Encoder::EncoderHistoryMove(GameState &state,
                                 std::vector<float>::iterator it,
                                 int symmetry) const {

    auto move_num = state.GetMoveNumber();
    auto past = std::min(move_num+1, kHistoryMove);

    auto boardsize = state.GetBoardSize();
    int num_intersections = state.GetNumIntersections();

    auto black_it = it + 0;
    auto white_it = it + kHistoryMove * num_intersections;

    for (auto p = 0; p < past; ++p) {
        auto board = state.GetPastBoard(p);

        FillColorStones(board, black_it, kBlack, symmetry);
        FillColorStones(board, white_it, kWhite, symmetry);
    }
}
