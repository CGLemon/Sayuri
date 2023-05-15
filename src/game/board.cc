#include <sstream>
#include <algorithm>
#include <set>
#include <queue>

#include "game/board.h"
#include "game/symmetry.h"

void Board::Reset(const int boardsize) {
    SetBoardSize(boardsize);

    ResetBoard();
    ResetBasicData();
}

void Board::ResetBoard() {
    const auto boardsize = GetBoardSize();
    for (int vtx = 0; vtx < kNumVertices; ++vtx) {
        state_[vtx] = kInvalid;
    }

    for (int y = 0; y < boardsize; ++y) {
        for (int x = 0; x < boardsize; ++x) {
            state_[GetVertex(x, y)] = kEmpty;
        }
    }

    const auto center = boardsize / 2;
    state_[GetVertex(center-1, center-1)] = kWhite;
    state_[GetVertex(center-1, center)]   = kBlack;
    state_[GetVertex(center, center-1)]   = kBlack;
    state_[GetVertex(center, center)]     = kWhite;
}

void Board::ResetBasicData() {
    last_move_ = kNullVertex;
    to_move_ = kBlack;
    passes_ = 0;
    move_number_ = 0;

    const int x_shift = GetLetterBoxSize();
    directions_[0] = (-x_shift);
    directions_[1] = (-1);
    directions_[2] = (+1);
    directions_[3] = (+x_shift);
    directions_[4] = (-x_shift - 1);
    directions_[5] = (-x_shift + 1);
    directions_[6] = (+x_shift - 1);
    directions_[7] = (+x_shift + 1);

    hash_ = ComputeHash();
}

bool Board::IsStar(const int x, const int y) const {
    const int size = GetBoardSize();
    const int point = GetIndex(x, y);
    int stars[3];
    int points[2];
    int hits = 0;

    if (size % 2 == 0 || size < 9) {
        return false;
    }

    stars[0] = size >= 13 ? 3 : 2;
    stars[1] = size / 2;
    stars[2] = size - 1 - stars[0];

    points[0] = point / size;
    points[1] = point % size;

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            if (points[i] == stars[j]) {
                hits++;
            }
        }
    }

    return hits >= 2;
}

std::string Board::GetStateString(const VertexType color, bool is_star) const {
    auto res = std::ostringstream{};

    color == kBlack ? res << 'x' :
        color == kWhite   ? res << 'o' : 
        is_star == true   ? res << '+' :
        color == kEmpty   ? res << '.' :
        color == kInvalid ? res << '-' : res << "error";
 
    return res.str();
}

std::string Board::GetSpcacesString(const int times) const {
    auto res = std::ostringstream{};
    for (int i = 0; i < times; ++i) {
        res << ' ';
    }
    return res.str();
}

std::string Board::GetColumnsString(const int bsize) const {
    auto res = std::ostringstream{};
    for (int i = 0; i < bsize; ++i) {
        if (i < 25) {
            res << (char)(('a' + i < 'i') ? 'a' + i : 'a' + i + 1);
        } else {
            res << (char)(('A' + (i - 25) < 'I') ? 'A' + (i - 25) : 'A' + (i - 25) + 1);
        }
        res << ' ';
    }
    res << std::endl;
    return res.str();
}

std::string Board::GetStoneCountString() const {
    auto res = std::ostringstream{};
    res << "(b) "<< ComputeStoneCount(kBlack) << " | "
            << "(w) " << ComputeStoneCount(kWhite) << std::endl;
    return res.str();
}

std::string Board::GetHashingString() const {
    auto out = std::ostringstream{};
    out << std::hex;
    out << "Hash: " << GetHash();
    out << std::dec << std::endl;
    return out.str();
}

std::string Board::GetBoardString(const int last_move, bool is_sgf) const {
    auto out = std::ostringstream{};
    auto boardsize = GetBoardSize();
    boardsize > 9 ? (out << GetSpcacesString(3))
                  : (out << GetSpcacesString(2));
    out << GetColumnsString(boardsize);

    for (int y = 0; y < boardsize; y++) {
        const auto row = is_sgf ? y : boardsize - y - 1;

        out << std::to_string(row + 1);
        if (row < 9 && boardsize > 9) {
            out << GetSpcacesString(1);
        }
        if (last_move == GetVertex(0, row)) {
            out << "(";
        } else {
            out << GetSpcacesString(1);
        }

        for (int x = 0; x < boardsize; x++) {
            const auto vtx = GetVertex(x, row);
            const auto state = GetState(vtx);
            out << GetStateString(
                       static_cast<VertexType>(state), IsStar(x, row));

            if (last_move == GetVertex(x, row)) {
                out << ")";
            } else if (x != boardsize - 1 && last_move == GetVertex(x, row) + 1) {
                out << "(";
            } else {
                out << GetSpcacesString(1);
            }
        }
        out << std::to_string(row + 1);
        out << std::endl;
    }
    boardsize > 9 ? (out << GetSpcacesString(3))
                  : (out << GetSpcacesString(2));
    out << GetColumnsString(boardsize);
    out << GetStoneCountString();
    out << GetHashingString();

    return out.str();
}

bool Board::IsLegalMove(const int vtx, const int color) const {
    return IsLegalMove(vtx, color, [](int /* vtx */, int /* color */) { return false; });
}

bool Board::IsLegalForPass(const int color) const {
    for (int vtx = 0 ; vtx < num_vertices_; ++vtx){
        if (state_[vtx] == kEmpty) {
            if (IsLegalMove(vtx, color)) {
                return false;
            }
        }
    }
    return true;
}

bool Board::IsLegalMove(const int vtx, const int color,
                        std::function<bool(int, int)> AvoidToMove) const {
    if (vtx == kResign) {
        return true;
    }

    if (vtx == kPass) {
        return IsLegalForPass(color);
    }

    if (AvoidToMove(vtx, color)) {
        return false;
    }

    const auto opp_color = !color;
    if(state_[vtx] == kEmpty){
        for(auto k = 0; k < 8; ++k){
            int avtx = vtx;
            do {
                avtx += directions_[k];
            } while (state_[avtx] == opp_color);

            if (avtx != (vtx + directions_[k]) &&
                            state_[avtx] == color) {
                return true;
            }
        }
    }
    return false;
}

void Board::SetMoveNumber(int number) {
    move_number_ = number;
}

void Board::SetBoardSize(int boardsize) {
    if (boardsize > kBoardSize) {
        boardsize = kBoardSize;
    } else if (boardsize < kMinGTPBoardSize) {
        boardsize = kMinGTPBoardSize;
    }

    board_size_ = boardsize;
    letter_box_size_ = boardsize+2;
    num_vertices_ = letter_box_size_ * letter_box_size_;
    num_intersections_ = board_size_ * board_size_;
}

void Board::SetLastMove(int first_vtx, int second_vtx) {
    last_move_ = first_vtx;
    last_move_2_ = second_vtx;
}

std::uint64_t Board::ComputeHash() const {
    return ComputeHash([](const auto vertex) { return vertex; });
}

std::uint64_t Board::ComputeSymmetryHash(int symmetry) const {
    return ComputeHash([this, symmetry](const auto vertex) {
        return Symmetry::Get().TransformVertex(board_size_, symmetry, vertex);
    });
}

std::uint64_t Board::ComputeHash(std::function<int(int)> transform) const {
    auto res = Zobrist::kEmpty;
    for (int v = 0; v < num_vertices_; ++v) {
        if (state_[v] != kInvalid) {
            res ^= Zobrist::kState[state_[v]][transform(v)];
        }
    }

    if (to_move_ == kBlack) {
        res ^= Zobrist::kBlackToMove;
    }
    res ^= Zobrist::KPass[GetPasses()];

    return res;
}

void Board::SetToMove(int color) {
    assert(color == kBlack || color == kWhite);
    UpdateZobristToMove(color, to_move_);
    to_move_ = color;
}

void Board::ExchangeToMove() {
    to_move_ = !(to_move_);
    UpdateZobristToMove(kBlack, kWhite);
}

void Board::UpdateStone(const int vtx, const int color) {
    assert(color == kEmpty || color == kBlack || color == kWhite);
    const int old_color = state_[vtx];
    state_[vtx] = static_cast<VertexType>(color);

    UpdateZobrist(vtx, color, old_color);
}

void Board::UpdateBoard(const int vtx, const int color) {
    assert(vtx != kPass && vtx != kResign);

    const auto opp_color = !color;
    auto updated_vertex = std::vector<int>{};
    UpdateStone(vtx, color);

    for(int k = 0; k < 8; ++k){
        int avtx = vtx;
        int res = 0;

        do {
            avtx += directions_[k];
            ++res;
        } while (state_[avtx] == opp_color);

        if (res > 1 && state_[avtx] == color) {
            for (int i = 0; i < (res-1); ++i){
                avtx -= directions_[k];
                UpdateStone(avtx, color);
                updated_vertex.emplace_back(avtx);
            }
        }
    }
}

int Board::ComputeStoneCount(const int color) const {
    int cnt = 0;
    for (const auto val : state_) {
        if (val == color) {
            cnt++;
        }
    }
    return cnt;
}

void Board::SetPasses(int val) {
    if (val > 4) {
        val = 4;
     }
     UpdateZobristPass(val, passes_);
     passes_ = val;
}

void Board::IncrementPasses() {
    int old_passes = passes_;
    passes_++;
    if (passes_ > 4) {
        passes_ = 4;
    }
    UpdateZobristPass(passes_, old_passes);
}

void Board::PlayMoveAssumeLegal(const int vtx, const int color) {
    assert(vtx != kResign);

    SetToMove(color);

    if (vtx == kPass) {
        IncrementPasses();
    } else {
        if (GetPasses() != 0) {
            SetPasses(0);
        }
        UpdateBoard(vtx, color);
    }
    last_move_ = vtx;
    move_number_++;

    ExchangeToMove();
}

int Board::GetMoveNumber() const {
    return move_number_;
}

bool Board::IsEdge(const int vtx) const {
    int invalid = 0;
    for(int k = 0; k < 4; ++k){
        int avtx = vtx + directions_[k];
        if (state_[avtx] == kInvalid) {
            ++invalid;
        }
    }
    return invalid == 1;
}

bool Board::IsCorner(const int vtx) const {
    int invalid = 0;
    for(int k = 0; k < 4; ++k){
        int avtx = vtx + directions_[k];
        if (state_[avtx] == kInvalid) {
            ++invalid;
        }
    }
    return invalid >= 2;
}

bool Board::IsThreatPass(const int vtx, const int color) const {
    if (!IsLegalMove(vtx, color)) {
        return false;
    }
    Board fork_board = *this;
    fork_board.PlayMoveAssumeLegal(vtx, color);

    return fork_board.IsLegalForPass(!color);
}

std::string Board::GetMoveTypesString(int, int) const {
    // this function is for go game, do nothing special in othello game
    return std::string{};
}

int Board::GetEmptyCount() const {
    // this function is for go game, do nothing special in othello game
    return 0;
}

int Board::GetEmpty(const int) const {
    // this function is for go game, do nothing special in othello game
    return 0;
}

bool Board::IsCaptureMove(const int, const int) const {
    // this function is for go game, do nothing special in othello game
    return 0;
}

int Board::ComputeScoreOnBoard() const {
    auto score_area = std::vector<int>(GetNumIntersections(), kInvalid);
    int black_score_lead = 0;

    ComputeScoreArea(score_area);

    for (int y = 0; y < board_size_; ++y) {
        for (int x = 0; x < board_size_; ++x) {
            const auto idx = GetIndex(x, y);
            if (score_area[idx] == kBlack) {
                ++black_score_lead;
            } else if (score_area[idx] == kWhite) {
                --black_score_lead;
            }
        }
    }
    return black_score_lead;
}

float Board::ComputeFinalScore(float komi) const {
    return static_cast<float>(ComputeScoreOnBoard()) - komi;
}

void Board::ComputeScoreArea(std::vector<int> &result) const {
    for (int idx = 0; idx < num_intersections_; ++idx) {
        const int x = idx % board_size_;
        const int y = idx / board_size_;
        const int vtx = GetVertex(x,y);

        result[idx] = state_[vtx];
    }
}
