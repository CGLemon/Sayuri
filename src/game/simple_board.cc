#include <sstream>
#include <queue>

#include "game/simple_board.h"
#include "game/symmetry.h"

void SimpleBoard::Reset(const int boardsize) {
    SetBoardSize(boardsize);

    ResetBoard();
    ResetBasicData();
}

void SimpleBoard::ResetBoard() {
    const auto num_vertices = GetNumVertices();
    const auto boardsize = GetBoardSize();
    for (int vtx = 0; vtx < num_vertices; ++vtx) {
        state_[vtx] = kInvalid;
        neighbours_[vtx] = 0;
    }

    empty_cnt_ = 0;

    for (int y = 0; y < boardsize; ++y) {
        for (int x = 0; x < boardsize; ++x) {
            const auto vtx = GetVertex(x, y);
            state_[vtx] = kEmpty;
            empty_idx_[vtx] = empty_cnt_;
            empty_[empty_cnt_++] = vtx;

            if (x == 0 || x == (boardsize - 1)) {
                neighbours_[vtx] += ((1ULL << kBlackNeighborShift) |
                                    (1ULL << kWhiteNeighborShift) |
                                    (1ULL << kEmptyNeighborShift));
            } else {
                neighbours_[vtx] += (2ULL << kEmptyNeighborShift);
            }

            if (y == 0 || y == (boardsize - 1)) {
                neighbours_[vtx] += ((1ULL << kBlackNeighborShift) |
                                    (1ULL << kWhiteNeighborShift) |
                                    (1ULL << kEmptyNeighborShift));
            } else {
                neighbours_[vtx] += (2ULL << kEmptyNeighborShift);
            }
        }
    }
}

void SimpleBoard::ResetBasicData() {
    prisoners_[kBlack] = 0;
    prisoners_[kWhite] = 0;

    ko_move_ = kNullVertex;
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

    strings_.Reset();

    hash_ = ComputeHash(GetKoMove());
    ko_hash_ = ComputeKoHash();
}

bool SimpleBoard::IsStar(const int x, const int y) const {
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

std::string SimpleBoard::GetStateString(const VertexType color, bool is_star) const {
    auto res = std::ostringstream{};

    color == kBlack ? res << 'x' :
        color == kWhite   ? res << 'o' : 
        is_star == true   ? res << '+' :
        color == kEmpty   ? res << '.' :
        color == kInvalid ? res << '-' : res << "error";
 
    return res.str();
}

std::string SimpleBoard::GetSpcacesString(const int times) const {
    auto res = std::ostringstream{};
    for (int i = 0; i < times; ++i) {
        res << ' ';
    }
    return res.str();
}

std::string SimpleBoard::GetColumnsString(const int bsize) const {
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

std::string SimpleBoard::GetPrisonersString() const {
    auto out = std::ostringstream{};
    out << "BLACK (X) has captured ";
    out << std::to_string(GetPrisoner(kBlack));
    out << " stones" << std::endl;
    out << "WHITE (O) has captured ";
    out << std::to_string(GetPrisoner(kWhite));
    out << " stones" << std::endl;
    return out.str();
}

std::string SimpleBoard::GetBoardString(const int last_move, bool is_sgf) const {
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
    out << GetPrisonersString();
    return out.str();
}

bool SimpleBoard::IsLegalMove(const int vtx, const int color) const {
    return IsLegalMove(vtx, color, [](int /* vtx */, int /* color */) { return false; });
}

bool SimpleBoard::IsLegalMove(const int vtx, const int color,
                              std::function<bool(int, int)> AvoidToMove) const {
    if (vtx == kPass || vtx == kResign) {
        return true;
    }

    if (state_[vtx] != kEmpty) {
        return false;
    }

    if (AvoidToMove(vtx, color)) {
        return false;
    }

    if (IsSuicide(vtx, color)) {
        return false;
    }

    if (vtx == ko_move_) {
        return false;
    }

    return true;

}

void SimpleBoard::SetMoveNumber(int number) {
    move_number_ = number;
}

void SimpleBoard::SetBoardSize(int boardsize) {
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

void SimpleBoard::SetLastMove(int vertex) {
    last_move_ = vertex;
}

int SimpleBoard::ComputeReachColor(int color) const {

    auto buf = std::vector<bool>(num_vertices_, false);
    auto PeekState = [&](int vtx) -> int {
        return state_[vtx];
    };

    return ComputeReachColor(color, kEmpty, buf, PeekState);
}


int SimpleBoard::ComputeReachColor(int color, int spread_color,
                                   std::vector<bool> &buf,
                                   std::function<int(int)> Peek) const {
    if (buf.size() != (size_t)num_vertices_) {
        buf.resize(num_vertices_);
    }

    int reachable = 0;
    auto open = std::queue<int>();
    for (int y = 0; y < board_size_; ++y) {
        for (int x = 0; x < board_size_; ++x) {
            const auto vertex = GetVertex(x, y);
            const auto peek = Peek(vertex);

            if (peek == color) {
                reachable++;
                buf[vertex] = true;
                open.emplace(vertex);
            } else {
                buf[vertex] = false;
            }
        }
    }
    while (!open.empty()) {
        const auto vertex = open.front();
        open.pop();

        for (int k = 0; k < 4; ++k) {
            const auto neighbor = vertex + directions_[k];
            const auto peek = Peek(neighbor);

            if (!buf[neighbor] && peek == spread_color) {
                reachable++;
                buf[neighbor] = true;
                open.emplace(neighbor);
            }
        }
    }
    return reachable;
}


std::uint64_t SimpleBoard::ComputeHash(int komove) const {
    return ComputeHash(komove, [](const auto vertex) { return vertex; });
}

std::uint64_t SimpleBoard::ComputeSymmetryHash(int komove, int symmetry) const {
    return ComputeHash(komove, [this, symmetry](const auto vertex) {
        return Symmetry::Get().TransformVertex(symmetry, vertex);
    });
}

std::uint64_t SimpleBoard::ComputeKoHash() const {
    return ComputeKoHash([](const auto vertex) { return vertex; });
}

std::uint64_t SimpleBoard::ComputeHash(int komove, std::function<int(int)> transform) const {
    auto res = ComputeKoHash(transform);

    if (to_move_ == kBlack) {
        res ^= Zobrist::kBlackToMove;
    }
    res ^= Zobrist::kPrisoner[kBlack][GetPrisoner(kBlack)];
    res ^= Zobrist::kPrisoner[kWhite][GetPrisoner(kWhite)];
    res ^= Zobrist::KPass[GetPasses()];
    res ^= Zobrist::kKoMove[transform(komove)];

    return res;
}

std::uint64_t SimpleBoard::ComputeKoHash(std::function<int(int)> transform) const {
    auto res = Zobrist::kEmpty;
    for (int v = 0; v < num_vertices_; ++v) {
        if (state_[v] != kInvalid) {
            res ^= Zobrist::kState[state_[v]][transform(v)];
        }
    }
    return res;
}

int SimpleBoard::CountPliberties(const int vtx) const {
    return (neighbours_[vtx] >> (kEmptyNeighborShift)) & kNeighborMask;
}

bool SimpleBoard::IsSimpleEye(const int vtx, const int color) const {
    return neighbours_[vtx] & kEyeMask[color];
}

bool SimpleBoard::IsSuicide(const int vtx, const int color) const {
    if (CountPliberties(vtx)) {
        return false;
    }

    for (auto k = 0; k < 4; ++k) {
        const auto avtx = vtx + directions_[k];
        const auto libs =  strings_.GetLiberty(strings_.GetParent(avtx));
        const auto state = GetState(avtx);
        if (state == color && libs > 1) {
            // Be sure that the string at least is one liberty.
            return false;
        } else if (state == (!color) && libs <= 1) {
            // We can capture opponent's stone.
            return false;
        }
    }

    return true;
}

void SimpleBoard::SetToMove(int color) {
    assert(color == kBlack || color == kWhite);
    UpdateZobristToMove(color, to_move_);
    to_move_ = color;
}

void SimpleBoard::ExchangeToMove() {
    to_move_ = !(to_move_);
    UpdateZobristToMove(kBlack, kWhite);
}

void SimpleBoard::AddStone(const int vtx, const int color) {
    assert(color == kBlack || color == kWhite);
    assert(state_[vtx] == kEmpty);

    int nbr_pars[4];
    int nbr_par_cnt = 0;

    // Set board content.
    state_[vtx] = static_cast<VertexType>(color);

    // Updata zobrist key.
    UpdateZobrist(vtx, color, kEmpty);

    for (int k = 0; k < 4; ++k) {
        const auto avtx = vtx + directions_[k];
        neighbours_[avtx] += ((1ULL << (kNeighborShift * color))
                                 - (1ULL << kEmptyNeighborShift));

        bool found = false;
        const auto ip = strings_.GetParent(avtx);
        for (int i = 0; i < nbr_par_cnt; ++i) {
            if (nbr_pars[i] == ip) {
                found = true;
                break;
            }
        }
        if (!found) {
            strings_.liberties_[ip]--;
            nbr_pars[nbr_par_cnt++] = ip;
        }
    }
}

void SimpleBoard::RemoveStone(const int vtx, const int color) {
    assert(color == kBlack || color == kWhite);
    assert(state_[vtx] != kEmpty);

    int nbr_pars[4];
    int nbr_par_cnt = 0;

    // Set board content.
    state_[vtx] = kEmpty;

    // Updata zobrist key.
    UpdateZobrist(vtx, kEmpty, color);

    for (int k = 0; k < 4; ++k) {
        const int avtx = vtx + directions_[k];
        neighbours_[avtx] += ((1ULL << kEmptyNeighborShift)
                                 - (1ULL << (kNeighborShift * color)));

        bool found = false;
        const int ip = strings_.GetParent(avtx);
        for (int i = 0; i < nbr_par_cnt; i++) {
            if (nbr_pars[i] == ip) {
                found = true;
                break;
            }
        }
        if (!found) {
            strings_.liberties_[ip]++;
            nbr_pars[nbr_par_cnt++] = ip;
        }
    }
}

void SimpleBoard::MergeStrings(const int ip, const int aip) {
    assert(ip != kNumVertices && aip != kNumVertices);
    assert(strings_.GetStone(ip) >= strings_.GetStone(aip));

    strings_.stones_[ip] += strings_.GetStone(aip);
    int next_pos = aip;

    do {
        for (int k = 0; k < 4; k++) {
            const int apos = next_pos + directions_[k];
            if (state_[apos] == kEmpty) {
                bool found = false;
                for (int kk = 0; kk < 4; kk++) {
                    const int aapos = apos + directions_[kk];
                    if (strings_.GetParent(aapos) == ip) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    strings_.liberties_[ip]++;
                }
            }
        }

        strings_.parent_[next_pos] = ip;
        next_pos = strings_.GetNext(next_pos);
    } while (next_pos != aip);

    std::swap(strings_.next_[aip], strings_.next_[ip]);
}

int SimpleBoard::RemoveString(const int ip) {
    int pos = ip;
    int removed = 0;
    int color = state_[ip];

    assert(color != kEmpty);

    do {
        RemoveStone(pos, color);
        strings_.parent_[pos] = kNumVertices;

        empty_idx_[pos] = empty_cnt_;
        empty_[empty_cnt_] = pos;
        empty_cnt_++;

        removed++;

        pos = strings_.GetNext(pos);
    } while (pos != ip);

    return removed;
}


int SimpleBoard::UpdateBoard(const int vtx, const int color) {
    assert(vtx != kPass && vtx != kResign);

    AddStone(vtx, color);
    strings_.AddStone(vtx, CountPliberties(vtx));

    bool is_eyeplay = IsSimpleEye(vtx, !color);

    int captured_stones = 0;
    int captured_vtx = kNullVertex;

    for (int k = 0; k < 4; ++k) {
        const auto avtx = vtx + directions_[k];
        const auto aip = strings_.GetParent(avtx);
        const auto state = GetState(avtx);

        if (state == !color) {
            if (strings_.GetLiberty(aip) <= 0) {
                const auto this_captured = RemoveString(avtx);
                captured_vtx = avtx;
                captured_stones += this_captured;
            }
        } else if (state == color) {
            const auto ip = strings_.GetParent(vtx);
            if (ip != aip) {
                if (strings_.GetStone(ip) >= strings_.GetStone(aip)) {
                    MergeStrings(ip, aip);
                } else {
                    MergeStrings(aip, ip);
                }
            }
        }
    }

    if (captured_stones != 0) {
        const int old_prisoners = prisoners_[color];
        const int new_prisoners = old_prisoners + captured_stones;
        prisoners_[color] = new_prisoners;
        UpdateZobristPrisoner(color, new_prisoners, old_prisoners);
    }

    int lastvertex = empty_[--empty_cnt_];
    empty_idx_[lastvertex] = empty_idx_[vtx];
    empty_[empty_idx_[vtx]] = lastvertex;

    if (strings_.GetLiberty(strings_.GetParent(vtx)) == 0) {
        assert(captured_stones == 0);
        RemoveString(vtx);
    }

    if (captured_stones == 1 && is_eyeplay) {
        assert(state_[captured_vtx] == kEmpty && !IsSuicide(captured_vtx, !color));
        return captured_vtx;
    }

    return kNullVertex;
}

void SimpleBoard::SetPasses(int val) {
    if (val > 4) {
        val = 4;
     }
     UpdateZobristPass(val, passes_);
     passes_ = val;
}

void SimpleBoard::IncrementPasses() {
    int old_passes = passes_;
    passes_++;
    if (passes_ > 4) {
        passes_ = 4;
    }
    UpdateZobristPass(passes_, old_passes);
}

void SimpleBoard::PlayMoveAssumeLegal(const int vtx, const int color) {
    assert(vtx != kResign);

    SetToMove(color);
    const int old_ko_move = ko_move_;

    if (vtx == kPass) {
        IncrementPasses();
        ko_move_ = kNullVertex;
    } else {
        if (GetPasses() != 0) {
            SetPasses(0);
        }
        ko_move_ = UpdateBoard(vtx, color);
    }

    if (ko_move_ != old_ko_move) {
        UpdateZobristKo(ko_move_, old_ko_move);
    }
    last_move_ = vtx;
    move_number_++;

    ExchangeToMove();
}

int SimpleBoard::GetPrisoner(const int color) const {
    return prisoners_[color];
}

int SimpleBoard::GetMoveNumber() const {
    return move_number_;
}

int SimpleBoard::GetBoardSize() const {
    return board_size_;
}

int SimpleBoard::GetLetterBoxSize() const {
    return letter_box_size_;
}

int SimpleBoard::GetNumVertices() const {
    return num_vertices_;
}

int SimpleBoard::GetNumIntersections() const {
    return num_intersections_;
}

int SimpleBoard::GetToMove() const {
    return to_move_;
}

int SimpleBoard::GetLastMove() const {
    return last_move_;
}

int SimpleBoard::GetKoMove() const {
    return ko_move_;
}

int SimpleBoard::GetPasses() const {
    return passes_;
}

int SimpleBoard::GetKoHash() const {
    return ko_hash_;
}

int SimpleBoard::GetHash() const {
    return hash_;
}

int SimpleBoard::GetState(const int vtx) const {
    return state_[vtx];
}

int SimpleBoard::GetState(const int x, const int y) const {
    return GetState(GetVertex(x,y));
}

void SimpleBoard::UpdateZobrist(const int vtx,
                                const int new_color,
                                const int old_color) {
    hash_ ^= Zobrist::kState[old_color][vtx];
    hash_ ^= Zobrist::kState[new_color][vtx];
    ko_hash_ ^= Zobrist::kState[old_color][vtx];
    ko_hash_ ^= Zobrist::kState[new_color][vtx];
}

void SimpleBoard::UpdateZobristPrisoner(const int color,
                                        const int new_pris,
                                        const int old_pris) {
    hash_ ^= Zobrist::kPrisoner[color][old_pris];
    hash_ ^= Zobrist::kPrisoner[color][new_pris];
}

void SimpleBoard::UpdateZobristToMove(const int new_color,
                                      const int old_color) {
    if (old_color != new_color) {
        hash_ ^= Zobrist::kBlackToMove;
    }
}

void SimpleBoard::UpdateZobristKo(const int new_komove,
                                  const int old_komove) {
    hash_ ^= Zobrist::kKoMove[old_komove];
    hash_ ^= Zobrist::kKoMove[new_komove];
}

void SimpleBoard::UpdateZobristPass(const int new_pass,
                                    const int old_pass) {
    hash_ ^= Zobrist::KPass[old_pass];
    hash_ ^= Zobrist::KPass[new_pass];
}
