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

void Board::ResetBasicData() {
    prisoners_[kBlack] = 0;
    prisoners_[kWhite] = 0;

    ko_move_ = kNullVertex;
    last_move_ = kNullVertex;
    last_move_2_ = kNullVertex;
    to_move_ = kBlack;
    passes_ = 0;

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

std::string Board::GetSpacesString(const int times) const {
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

std::string Board::GetHashingString() const {
    auto out = std::ostringstream{};
    out << std::hex << std::uppercase
            << "Hash: " << GetHash() << " | "
            << "Ko Hash: " << GetKoHash()
            << std::dec // cancel hex
            << std::endl;
    return out.str();
}

std::string Board::GetPrisonersString() const {
    auto out = std::ostringstream{};
    out << "BLACK (X) has captured "
            << std::to_string(GetPrisoner(kBlack))
            << " stones" << std::endl
            << "WHITE (O) has captured "
            << std::to_string(GetPrisoner(kWhite))
            << " stones" << std::endl;
    return out.str();
}

std::string Board::GetBoardString(const int last_move, bool y_invert) const {
    auto out = std::ostringstream{};
    auto boardsize = GetBoardSize();
    boardsize > 9 ? (out << GetSpacesString(3))
                  : (out << GetSpacesString(2));
    out << GetColumnsString(boardsize);

    for (int y = 0; y < boardsize; y++) {
        const auto row = y_invert ? y : boardsize - y - 1;

        out << std::to_string(row + 1);
        if (row < 9 && boardsize > 9) {
            out << GetSpacesString(1);
        }
        if (last_move == GetVertex(0, row)) {
            out << '(';
        } else {
            out << GetSpacesString(1);
        }

        for (int x = 0; x < boardsize; x++) {
            const auto vtx = GetVertex(x, row);
            const auto state = GetState(vtx);
            out << GetStateString(
                       static_cast<VertexType>(state), IsStar(x, row));

            if (last_move == GetVertex(x, row)) {
                out << ')';
            } else if (x != boardsize - 1 && last_move == GetVertex(x, row) + 1) {
                out << '(';
            } else {
                out << GetSpacesString(1);
            }
        }
        out << std::to_string(row + 1);
        out << std::endl;
    }
    boardsize > 9 ? (out << GetSpacesString(3))
                  : (out << GetSpacesString(2));
    out << GetColumnsString(boardsize);
    out << GetPrisonersString();
    out << GetHashingString();

    return out.str();
}

bool Board::IsLegalMove(const int vtx, const int color) const {
    return IsLegalMove(vtx, color, [](int /* vtx */, int /* color */) { return false; });
}

bool Board::IsLegalMove(const int vtx, const int color,
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

void Board::SetBoardSize(int boardsize) {
    if (boardsize > kBoardSize) {
        boardsize = kBoardSize;
    } else if (boardsize < kMinGTPBoardSize) {
        boardsize = kMinGTPBoardSize;
    }

    board_size_ = boardsize;
    letter_box_size_ = board_size_ + 2;
    num_vertices_ = letter_box_size_ * letter_box_size_;
    num_intersections_ = board_size_ * board_size_;
}

void Board::SetLastMove(int first_vtx, int second_vtx) {
    last_move_ = first_vtx;
    last_move_2_ = second_vtx;
}

void Board::RemoveMarkedStrings(std::vector<int> &marked) {
    int removed_stones[2] = {0, 0};
    for (auto &vtx : marked) {
        auto color = GetState(vtx);
        if (color == kBlack || color == kWhite) {
            removed_stones[color] += RemoveString(vtx);
        }
    }

    IncreasePrisoner(kBlack, removed_stones[kWhite]);
    IncreasePrisoner(kWhite, removed_stones[kBlack]);
}

int Board::ComputeReachGroup(int start_vertex, int spread_color, std::vector<bool> &buf) const {
    auto PeekState = [&](int vtx) -> int {
        return state_[vtx];
    };

    return ComputeReachGroup(start_vertex, spread_color, buf, PeekState);
}

int Board::ComputeReachGroup(int start_vertex, int spread_color,
                             std::vector<bool> &buf, std::function<int(int)> Peek) const {
    if (buf.size() != (size_t)num_vertices_) {
        buf.resize(num_vertices_);
    }
    int reachable = 0;
    auto open = std::queue<int>();

    buf[start_vertex] = true;
    open.emplace(start_vertex);
    ++reachable;

    while (!open.empty()) {
        const auto vertex = open.front();
        open.pop();

        for (int k = 0; k < 4; ++k) {
            const auto neighbor = vertex + directions_[k];
            const auto peek = Peek(neighbor);

            if (!buf[neighbor] && peek == spread_color) {
                ++reachable;
                buf[neighbor] = true;
                open.emplace(neighbor);
            }
        }
    }
    return reachable;
}

int Board::ComputeReachColor(int color) const {
    auto buf = std::vector<bool>(num_vertices_, false);
    auto PeekState = [&](int vtx) -> int {
        return state_[vtx];
    };

    return ComputeReachColor(color, kEmpty, buf, PeekState);
}

int Board::ComputeReachColor(int color, int spread_color,
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
                ++reachable;
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
                ++reachable;
                buf[neighbor] = true;
                open.emplace(neighbor);
            }
        }
    }
    return reachable;
}

std::uint64_t Board::ComputeHash(int komove) const {
    return ComputeHash(komove, [](const auto vertex) { return vertex; });
}

std::uint64_t Board::ComputeSymmetryHash(int komove, int symmetry) const {
    return ComputeHash(komove, [this, symmetry](const auto vertex) {
        return Symmetry::Get().TransformVertex(board_size_, symmetry, vertex);
    });
}

std::uint64_t Board::ComputeKoHash() const {
    return ComputeKoHash([](const auto vertex) { return vertex; });
}

std::uint64_t Board::ComputeKoHash(int symmetry) const {
    return ComputeKoHash([this, symmetry](const auto vertex) {
        return Symmetry::Get().TransformVertex(board_size_, symmetry, vertex);
    });
}

std::uint64_t Board::ComputeHash(int komove, std::function<int(int)> transform) const {
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

std::uint64_t Board::ComputeKoHash(std::function<int(int)> transform) const {
    auto res = Zobrist::kEmpty;
    for (int v = 0; v < num_vertices_; ++v) {
        if (state_[v] != kInvalid) {
            res ^= Zobrist::kState[state_[v]][transform(v)];
        }
    }
    return res;
}

int Board::CountPliberties(const int vtx) const {
    return (neighbours_[vtx] >> kEmptyNeighborShift) & kNeighborMask;
}

void Board::FindStringSurround(const int vtx,
                               const int color,
                               std::vector<int>& lib_buf,
                               std::vector<int>& index_buf) const {
    const auto set_insert = [](std::vector<int> &buf, int element){
        auto begin = std::begin(buf);
        auto end = std::end(buf);
        auto res = std::find(begin, end, element);
        if (res == end) {
            buf.emplace_back(element);
        }
    };

    assert(GetState(vtx) == color);
    int next = vtx;

    do {
        for(int k = 0; k < 4; ++k) {
            const auto avtx = next + directions_[k];
            const auto state = GetState(avtx);
            if (state == kEmpty) {
                set_insert(lib_buf, avtx);
            } else if (state == !color) {
                set_insert(index_buf, strings_.GetParent(avtx));
            }
        }
        next = strings_.GetNext(next);
    } while (next != vtx);
}

int Board::FindStringLiberties(const int vtx,
                                   std::vector<int>& buf) const {
    auto num_found = size_t{0};
    auto next = vtx;
    do {
        for(int k = 0; k < 4; ++k) {
            const auto avtx = next + directions_[k];
            if (GetState(avtx) == kEmpty) {
                auto begin = std::begin(buf);
                auto end = std::end(buf);
                auto res = std::find(begin, end, avtx);
                if (res == end) {
                    buf.emplace_back(avtx);
                    num_found++;
                }
            }
        }
        next = strings_.GetNext(next);
    } while (next != vtx);

    return num_found;
}

int Board::FindStringLibertiesGainingCaptures(const int vtx,
                                                  std::vector<int>& buf) const {
    const int color = GetState(vtx);
    const int opp = !(color);

    assert(color == kBlack || color == kWhite);

    auto strings_buf = std::vector<int>{};
    int num_found = 0;
    int next = vtx;

    do {
        for(int k = 0; k < 4; ++k) {
            const int avtx = next + directions_[k];
            if(GetState(avtx) == opp) {
                const int aip = strings_.GetParent(avtx);
                if(strings_.GetLiberty(aip) == 1) {
                    auto begin = std::begin(strings_buf);
                    auto end = std::end(strings_buf);
                    auto res = std::find(begin, end, avtx);
                    if (res == end) {
                        num_found += FindStringLiberties(avtx, buf);
                    } else {
                        strings_buf.emplace_back(avtx);
                    }
                }
            }
        }
        next = strings_.GetNext(next);
    } while (next != vtx);

    return num_found;
}

std::pair<int, int> Board::GetLadderLiberties(const int vtx, const int color) const {
    const int stone_libs = CountPliberties(vtx);
    const int opp = (!color);

    int num_captures = 0;                 // Number of adjacent directions in which we will capture.
    int potential_libs_from_captures = 0; // Total number of stones we're capturing (possibly with multiplicity).
    int num_connection_libs = 0;          // Sum over friendly groups connected to of their libs-1.
    int max_connection_libs = stone_libs; // Max over friendly groups connected to of their libs-1.

    for (int k = 0; k < 4; ++k) {
        const auto avtx = vtx + directions_[k];
        const auto acolor = GetState(avtx);

        if (acolor == color) {
            const int alibs = strings_.GetLiberty(strings_.GetParent(avtx)) - 1;
            num_connection_libs += alibs;

            if(alibs > max_connection_libs) {
                max_connection_libs = alibs;
            }
        } else if (acolor == opp) {
            const int aip = strings_.GetParent(avtx);
            const int alibs = strings_.GetLiberty(aip);
            if (alibs == 1) {
                num_captures++;
                potential_libs_from_captures += strings_.GetStones(aip);
            }
        }
    }
    const int lower_bound =
        num_captures + max_connection_libs;
    const int upper_bound =
        stone_libs + potential_libs_from_captures + num_connection_libs;

    return std::make_pair(lower_bound, upper_bound);
}

LadderType Board::PreySelections(const int prey_color,
                                 const int ladder_vtx,
                                 std::vector<int>& selections, bool think_ko) const {
    assert(selections.empty());

    const int libs = strings_.GetLiberty(strings_.GetParent(ladder_vtx));
    if (libs >= 2 || (ko_move_ != kNullVertex && think_ko)) {
        // If we are the prey and the hunter left a simple ko point, assume we already win
        // because we don't want to say yes on ladders that depend on kos
        // This should also hopefully prevent any possible infinite loops - I don't know of any infinite loop
        // that would come up in a continuous atari sequence that doesn't ever leave a simple ko point.

        return LadderType::kGoodForPrey;
    }

    int num_move = FindStringLiberties(ladder_vtx, selections);

    assert(libs == 1);
    assert(num_move == libs);
    const int not_cap_move = selections[0];

    num_move += FindStringLibertiesGainingCaptures(ladder_vtx, selections);

    // The moves must be the legal.
    selections.erase(
        std::remove_if(std::begin(selections), std::end(selections),
            [&](int v) { return !IsLegalMove(v, prey_color);
        }),
        std::end(selections)
    );

    num_move = selections.size();

    // If there is no legal move, the ladder string dies.
    if (num_move == 0) {
        return LadderType::kGoodForHunter;
    }

    auto bgn = std::begin(selections);
    auto end = std::end(selections);

    if (std::find(bgn, end, not_cap_move) != end) {
        auto bound = GetLadderLiberties(not_cap_move, prey_color);
        const auto lower_bound = bound.first;
        const auto upper_bound = bound.second;
        if (lower_bound >= 3) {
            return LadderType::kGoodForPrey;
        }
        if (num_move == 1  && upper_bound == 1) {
            return LadderType::kGoodForHunter;
        }
    }

    return LadderType::kGoodForNeither; // keep running
}

LadderType Board::HunterSelections(const int prey_color,
                                   const int ladder_vtx,
                                   std::vector<int>& selections) const {
    assert(selections.empty());

    const int libs = strings_.GetLiberty(strings_.GetParent(ladder_vtx));

    if (libs >= 3) {
        // It is not a ladder.
        return LadderType::kGoodForPrey;
    } else if (libs <= 1) {
        // The ladder string will be captured next move.
        return LadderType::kGoodForHunter;
    }

    assert(libs == 2);

    auto buf = std::vector<int>{};
    int num_libs = FindStringLiberties(ladder_vtx, buf);

    assert(num_libs == libs);
#ifdef NDEBUG
    (void) num_libs;
#endif

    const int move_1 = buf[0];
    const int move_2 = buf[1];
    //TODO: Avoid double-ko death.

    if (!IsNeighbor(move_1, move_2)) {
        const int hunter_color = (!prey_color);
        const int libs_1 = CountPliberties(move_1);
        const int libs_2 = CountPliberties(move_2);

        if (libs_1 >= 3 && libs_2 >= 3) {
            // A ladder string must be only two liberties. The prey
            // is not a ladder
            return LadderType::kGoodForPrey;
        } else if (libs_1 >= 3) {
            // If the prey play the move 1, it is not a ladder. The
            // move 1 is the only move for hunter.
            if (IsLegalMove(move_1, hunter_color)) {
                selections.emplace_back(move_1);
            }
        } else if (libs_2 >= 3) {
            // If the prey play the move 2, it is not a ladder. The
            // move 2 is the only move for hunter.
            if (IsLegalMove(move_2, hunter_color)) {
                selections.emplace_back(move_2);
            }
        } else {
            if (IsLegalMove(move_1, hunter_color)) {
                selections.emplace_back(move_1);
            }
            if (IsLegalMove(move_2, hunter_color)) {
                selections.emplace_back(move_2);
            }
        }
    } else {
        // At least one liberty, Is is always legal move.
        selections.emplace_back(move_1);
        selections.emplace_back(move_2);
    }

    if (selections.empty()) {
        // The hunter has no atari move.
        return LadderType::kGoodForPrey;
    }

    return LadderType::kGoodForNeither; // keep running
}

LadderType Board::PreyMove(Board* board,
                           const int hunter_vtx, const int prey_color,
                           const int ladder_vtx, size_t& ladder_nodes, bool fork) const {

    if ((++ladder_nodes) >= kMaxLadderNodes) {
        // If hit the limit, assume prey have escaped.
        return LadderType::kGoodForPrey;
    }

    Board* ladder_board;
    if (fork) {
        // Need to delete it before the return.
        ladder_board = new Board(*board);
    } else {
        ladder_board = board;
    }

    if (hunter_vtx != kNullVertex) {
        // Hunter play move first.
        ladder_board->PlayMoveAssumeLegal(hunter_vtx, !prey_color);
    }
    // Search possible move(s) for prey.
    auto selections = std::vector<int>{};
    auto res = ladder_board->PreySelections(prey_color, ladder_vtx, selections, hunter_vtx != kNullVertex);

    if (res != LadderType::kGoodForNeither) {
        if (fork) {
            delete ladder_board;
        }
        return res;
    }

    bool next_fork = true;
    const size_t selection_size = selections.size();
    if (selection_size == 1) {
        // Only one move. We don't need to save the pre-board.
        next_fork = false;
    }

    auto best = LadderType::kGoodForNeither;

    for (auto i = size_t{0}; i < selection_size; ++i) {
        const int vtx = selections[i];
        auto next_res = HunterMove(ladder_board, vtx,
                                   prey_color, ladder_vtx,
                                   ladder_nodes, next_fork);

        assert(next_res != LadderType::kGoodForNeither);

        best = next_res;
        if (next_res == LadderType::kGoodForPrey) {
            break;
        }
    }

    if (fork) {
        delete ladder_board;
    }
    return best;
}

LadderType Board::HunterMove(Board* board,
                             const int prey_vtx, const int prey_color,
                             const int ladder_vtx, size_t& ladder_nodes, bool fork) const {
    if ((++ladder_nodes) >= kMaxLadderNodes) {
        // If hit the limit, assume prey have escaped.
        return LadderType::kGoodForPrey;
    }

    Board* ladder_board;
    if (fork) {
        // Need to delete it before the return.
        ladder_board = new Board(*board);
    } else {
        ladder_board = board;
    }

    if (prey_vtx != kNullVertex) {
        // Prey play move first.
        ladder_board->PlayMoveAssumeLegal(prey_vtx, prey_color);
    }

    // Search possible move(s) for hunter.
    auto selections = std::vector<int>{};
    auto res = ladder_board->HunterSelections(prey_color, ladder_vtx, selections);

    if (res != LadderType::kGoodForNeither) {
        if (fork) {
            delete ladder_board;
        }
        return res;
    }

    bool next_fork = true;
    const auto selection_size = selections.size();
    if (selection_size == 1) {
        // Only one move. We don't need to save the pre-board.
        next_fork = false;
    }

    auto best = LadderType::kGoodForNeither;

    for (auto i = size_t{0}; i < selection_size; ++i) {
        const int vtx = selections[i];
        auto next_res = PreyMove(ladder_board, vtx,
                                 prey_color, ladder_vtx,
                                 ladder_nodes, next_fork);

        assert(next_res != LadderType::kGoodForNeither);

        best = next_res;
        if (next_res == LadderType::kGoodForHunter) {
            break;
        }
    }

    if (fork) {
        delete ladder_board;
    }
    return best;
}

bool Board::IsLadder(const int vtx, std::vector<int> &vital_moves) const {
    if (vtx == kPass) {
        return false;
    }

    const int prey_color = GetState(vtx);
    if (prey_color == kEmpty || prey_color == kInvalid) {
        return false;
    }

    vital_moves.clear();

    auto buf = std::vector<int>{};
    const int libs = FindStringLiberties(vtx, buf);
    const int ladder_vtx = vtx;
    size_t searched_nodes = 0;
    auto res = LadderType::kGoodForNeither;

    if (libs == 1) {
        auto ladder_board = new Board(*this);
        res = PreyMove(ladder_board,
                       kNullVertex, prey_color,
                       ladder_vtx, searched_nodes, false);

        if (res == LadderType::kGoodForHunter) {
            vital_moves.emplace_back(buf[0]);
        }
        delete ladder_board;
    } else if (libs == 2) {
        for (auto vvtx: buf) {
            auto ladder_board = new Board(*this);
            if (ladder_board->IsLegalMove(vvtx, !prey_color)) {

                // force the hunter do atari move first
                res = PreyMove(ladder_board,
                                 vvtx, prey_color,
                                 ladder_vtx, searched_nodes, false);
                if (res == LadderType::kGoodForHunter) {
                    vital_moves.emplace_back(vvtx);
                }
            }
            delete ladder_board;
        }
    } else if (libs >= 3) {
        res = LadderType::kGoodForPrey;
    }

    assert(res != LadderType::kGoodForNeither);
    return !vital_moves.empty();
}

bool Board::IsSelfAtariMove(const int vtx, const int color) const {
    int self_libs = CountPliberties(vtx);
    auto potential_libs_buf = std::vector<int>({vtx});
    auto my_parent_strings = std::vector<int>{};

    for (int k = 0; k < 4; ++k) {
        const auto avtx = vtx + directions_[k];
        const auto aip = strings_.GetParent(avtx);
        const auto libs =  strings_.GetLiberty(aip);
        const auto state = GetState(avtx);

        if (state == color) {
            // Connect with my string.
            FindStringLiberties(avtx, potential_libs_buf);
            my_parent_strings.emplace_back(aip);
        } else if (state == (!color) && libs <= 1) {
            // We can capture opponent's string.
            self_libs += 1;

            // TODO: Fully implement it here. Find gaining liberties by
            //       capturing.
        }
    }

    int potential_libs = potential_libs_buf.size() - 1;

    return (potential_libs + self_libs) == 1;
}

bool Board::IsAtariMove(const int vtx, const int color) const {
    if (IsSuicide(vtx, color)) {
        return false;
    }
    const int opp_color = (!color);

    for (int k = 0; k < 4; ++k) {
        const auto avtx = vtx + directions_[k];
        if (GetState(avtx) == opp_color) {
            const auto libs = strings_.GetLiberty(strings_.GetParent(avtx));
            if (libs == 2) {
                return true;
            }
        }
    }

    return false;
}

bool Board::IsCaptureMove(const int vtx, const int color) const {
    const int opp_color = (!color);
    for (int k = 0; k < 4; ++k) {
        const auto avtx = vtx + directions_[k];
        if (GetState(avtx) == opp_color) {
            const auto libs = strings_.GetLiberty(strings_.GetParent(avtx));
            if (libs == 1) {
                return true;
            }
        }
    }
    return false;
}

bool Board::IsEscapeMove(const int vtx, const int color) const {
    if (IsSuicide(vtx, color)) {
        return false;
    }

    return IsCaptureMove(vtx, !color);
}

bool Board::IsNeighbor(const int vtx, const int avtx) const {
    for (int k = 0; k < 4; ++k) {
        if ((vtx + directions_[k]) == avtx) {
            return true;
        }
    }
    return false;
}

bool Board::IsSimpleEye(const int vtx, const int color) const {
    return neighbours_[vtx] & kEyeMask[color];
}

bool Board::IsRealEye(const int vtx, const int color) const {
    if (state_[vtx] != kEmpty) {
        return false;
    }

    if (!IsSimpleEye(vtx, color)) {
        return false;
    }

    std::array<int, 4> color_count;

    color_count[kBlack] = 0;
    color_count[kWhite] = 0;
    // color_count[kEmpty] = 0; unused
    color_count[kInvalid] = 0;

    for (int k = 4; k < 8; ++k) {
        const auto avtx = vtx + directions_[k];
        color_count[state_[avtx]]++;
    }

    if (color_count[kInvalid] == 0) {
        // The eye is not at side or corner.
        if (color_count[!color] > 1) {
            return false;
        }
    } else {
        if (color_count[!color] > 0) {
            return false;
        }
    }

    return true;
}

bool Board::IsSuicide(const int vtx, const int color) const {
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

bool Board::IsSeki(const int vtx) const {
    if (state_[vtx] != kEmpty) {
        // Not empty point in the seki.
        return false;
    }

    int string_parent[2] = {kNullVertex, kNullVertex};

    for (auto k = 0; k < 4; ++k) {
        const auto avtx = vtx + directions_[k];
        const auto ip = strings_.GetParent(avtx);
        const auto state = state_[avtx];

        if (state == kEmpty) {
            // do nothing...
        }

        if (state == kBlack || state == kWhite) {
            if (string_parent[state] == kNullVertex) {
                string_parent[state] = ip;
            } else if (string_parent[state] != ip) {
                // Must be only one string for each color.
                return false;
            }
        }
    }

    for (auto c = 0; c < 2; ++c) {
        const auto ip = string_parent[c];
        if (ip == kNullVertex) {
            // There must be two strings.
            return false;
        }

        if (strings_.GetLiberty(ip) != 2) {
            // Two strings must have two liberties.
            return false;
        }
    }

    auto lib_buf = std::vector<int>{};
    auto black_sur_idx_buf = std::vector<int>{};
    auto white_sur_idx_buf = std::vector<int>{};

    FindStringSurround(string_parent[kBlack], kBlack, lib_buf, black_sur_idx_buf);
    FindStringSurround(string_parent[kWhite], kWhite, lib_buf, white_sur_idx_buf);

    assert(lib_buf.size() == 2 || lib_buf.size() == 3);

    if (lib_buf.size() == 3) {
        // We simply think that it is seki in this case. It includes false-seki. The false-seki
        // string is not alive. but in the most case, we don't need to play the move in the false-seki
        // point to kill it.
        //
        // .x.ox..
        // oxoox..
        // .ooxx..
        // ooxx...
        // xxx....
        return true;
    }

    auto inner_color = kInvalid;
    if (black_sur_idx_buf.size() == 1) {
        inner_color = kBlack;
    } else if (white_sur_idx_buf.size() == 1) {
        inner_color = kWhite;
    }

    // The 'inner' means the potential seki string surrounded by the other string(s). The
    // black is inner color in the above case.
    //
    // .x.ox..
    // oxoox..
    // oooxx..
    // xxxx...
    // .......

    if (inner_color == kInvalid) {
        // It is the simple seki (no eyes) case.
        //
        // x.ox...
        // x.ox...
        // xxox...
        // ooxx...
        // .oo....
        return true;
    }

    // TODO: How about the seki with double-ko case? We should conside it.
    //       The others are seki with partly filled eye space case.

    auto eye_next = std::vector<int>(num_vertices_, kNullVertex);
    auto eye_size = 1;

    auto next_pos = string_parent[inner_color];
    int pos;

    // Mark the eye region as a string.
    do {
        pos = next_pos;
        next_pos = strings_.GetNext(next_pos);

        eye_next[pos] = next_pos;
        ++eye_size;
    } while (next_pos != string_parent[inner_color]);

    eye_next[pos] = vtx;
    eye_next[vtx] = next_pos;

    return !IsKillableSekiEyeShape(vtx, eye_size, eye_next);
}

bool Board::IsBorder(const int vtx) const {
    return IsNeighborColor(vtx, kInvalid);
}

bool Board::IsNeighborColor(const int vtx, const int color) const {
    for (int k = 0; k < 4; ++k) {
        if (state_[vtx + directions_[k]] == color) return true;
    }
    return false;
}

bool Board::IsKillableSekiEyeShape(const int vtx,
                                       const int eye_size,
                                       const std::vector<int> &eye_next) const {
    if (eye_size <= 3) {
        // We can always kill it.
        return true;
    } else if (eye_size >= 7) {
        // We simply think that it is enogh space to live (include seki);
        return false;
    }

    auto eye_region = std::vector<bool>(num_vertices_, false);
    int boundary_cnt = 0;
    int pos = vtx;

    // Mark the eye shape region.
    do {
        eye_region[pos] = true;
        if (IsBorder(pos)) {
            ++boundary_cnt;
        }
        pos = eye_next[pos];
    } while (pos != vtx);

    auto nakade_vtx = std::vector<int>{};
    auto potential_eyes = std::vector<std::vector<int>>{};
    pos = vtx;

    // Mark the nakade moves and its potential eyes.
    do {
        int influence_cnt = 0;
        auto p_eyes = std::vector<int>{};

        for (int k = 0; k < 8; ++k) {
            const auto apos = pos + directions_[k];
            if (eye_region[apos]) {
                ++influence_cnt;
                if (k >= 4) {
                    // The potential eyes are in the diagonal.
                    p_eyes.emplace_back(apos);
                }
            }
        }

        // The current position influences all empty points. It is nakade. Play the move
        // in it may kill the string. We will do search later.
        if (influence_cnt+1 == eye_size && !p_eyes.empty()) {
            nakade_vtx.emplace_back(pos);
            potential_eyes.emplace_back(p_eyes);
        }

        pos = eye_next[pos];
    } while (pos != vtx);

    const int nakade_cnt = nakade_vtx.size();
    if (nakade_cnt == 0) {
        // No nakade move. We simply thing that It is alive.
        //
        // .....ox.
        // .ooooox.
        // ooxxxxx.
        // xxx.....
        // ........

        return false;
    }

    for (const auto &e: potential_eyes) {
        // No potential eyes. We can kill it.
        if (e.empty()) return true;
    }

    const auto GetEmptySideCount = [this](const int eye_vtx,
                                          std::vector<bool> &eye_region) {
        int side_cnt = 0;
        for (int k = 0; k < 4; ++k) {
            if (eye_region[eye_vtx + directions_[k]]) {
                ++side_cnt;
            }
        }
        return side_cnt;
    };

    // Possible eye shape is here: https://senseis.xmp.net/?EyeShape
    if (eye_size == 4) {
        assert(nakade_cnt == 1);
        // Only bent four, Dogleg four and Squared four cases are here.

        if (boundary_cnt == 4) {
            // Bent four in the corner, we can kill it.
            //
            // ...ox..
            // .ooox..
            // ooxxx..
            // xxx....
            // .......

            return true;
        }

        const auto eye_cnt = potential_eyes[0].size();
        const auto eye_vtx = potential_eyes[0][0];
        if (eye_cnt == 1 && GetEmptySideCount(eye_vtx, eye_region) == 2) {
            // Squared four, obviously we can kill it.
            //
            // ..ox...
            // ..ox...
            // ooox...
            // xxxx...
            // .......

            return true;
        }

        // It is Dogleg four case and is also a killable eye shape. But it doesn't exsit.
        // We don't need to conside it.
        //
        // x..xo..
        // ..xxo..
        // xxxoo..
        // oooo...

        // Other bent four and Dogleg four are always alive.
    } else if (eye_size == 5) {
        // Should notice that crossed five case is not here.

        assert(nakade_cnt == 1);
        const auto eye_cnt = potential_eyes[0].size();
        const auto eye_vtx = potential_eyes[0][0];

        if (eye_cnt == 1 && GetEmptySideCount(eye_vtx, eye_region) == 2) {
            // Bulky Five, we can kill it.
            //
            // ...ox..
            // ..oox..
            // oooxx..
            // xxxx...
            // .......

            return true;
        }
    } else if (eye_size == 6) {
        assert(nakade_cnt == 1 || nakade_cnt == 2);
        if (nakade_cnt == 1) {
            const auto eye_cnt = potential_eyes[0].size();
            const auto eye_vtx = potential_eyes[0][0];

            if (eye_cnt == 1 && GetEmptySideCount(eye_vtx, eye_region) == 2) {
                // Rabbitty six, we can kill it.
                //
                // ..oox..
                // ...ox..
                // o.oox..
                // oooxx..
                // xxxx...
                // .......

                return true;
            }
        } else if (nakade_cnt == 2) {
            if (boundary_cnt == 4) {
                // Rectangular six in the corner, we can kill it.
                //
                // ...ox..
                // ...ox..
                // oooox..
                // xxxxx..
                // .......

                return true;
            }
        }
    }

    // It is impossible to go to here...

    return false;
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

void Board::AddStone(const int vtx, const int color) {
    assert(color == kBlack || color == kWhite);
    assert(state_[vtx] == kEmpty);

    int nbr_pars[4];
    int nbr_par_cnt = 0;

    // Set board content.
    state_[vtx] = static_cast<VertexType>(color);

    // Update zobrist key.
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

void Board::RemoveStone(const int vtx, const int color) {
    assert(color == kBlack || color == kWhite);
    assert(state_[vtx] != kEmpty);

    int nbr_pars[4];
    int nbr_par_cnt = 0;

    // Set board content.
    state_[vtx] = kEmpty;

    // Update zobrist key.
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

void Board::MergeStrings(const int ip, const int aip) {
    assert(ip != kNumVertices && aip != kNumVertices);
    assert(strings_.GetStones(ip) >= strings_.GetStones(aip));

    strings_.stones_[ip] += strings_.GetStones(aip);
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

int Board::RemoveString(const int ip) {
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

void Board::IncreasePrisoner(const int color, const int val) {
    const int old_prisoners = prisoners_[color];
    const int new_prisoners = old_prisoners + val;
    prisoners_[color] = new_prisoners;
    UpdateZobristPrisoner(color, new_prisoners, old_prisoners);
}

int Board::UpdateBoard(const int vtx, const int color) {
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
                if (strings_.GetStones(ip) >= strings_.GetStones(aip)) {
                    MergeStrings(ip, aip);
                } else {
                    MergeStrings(aip, ip);
                }
            }
        }
    }

    if (strings_.GetLiberty(strings_.GetParent(vtx)) == 0) {
        // Suicide move, this move is illegal in general rule.
        assert(captured_stones == 0);
        const int sucide_stones = RemoveString(vtx);

        IncreasePrisoner(!color, sucide_stones);
    }

    if (captured_stones != 0) {
        IncreasePrisoner(color, captured_stones);
    }

    // move last vertex in list to our position
    int lastvertex = empty_[--empty_cnt_];
    empty_idx_[lastvertex] = empty_idx_[vtx];
    empty_[empty_idx_[vtx]] = lastvertex;

    if (captured_stones == 1 && is_eyeplay) {
        // Make a ko.
        assert(state_[captured_vtx] == kEmpty && !IsSuicide(captured_vtx, !color));
        return captured_vtx;
    }

    return kNullVertex;
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
    last_move_2_ = last_move_;
    last_move_ = vtx;

    ExchangeToMove();
}

std::vector<int> Board::GetStringList(const int vtx) const {
    auto result = std::vector<int>{};

    auto start = strings_.GetParent(vtx);
    auto newpos = start;

    do {
        result.emplace_back(newpos);
        newpos = strings_.GetNext(newpos);
    } while (newpos != start);

    assert(!result.empty());

    return result;
}

int Board::ComputeScoreOnBoard(const int color, const int scoring,
                               const std::vector<int> &territory_helper) const {
    int black_score_lead = 0;

    auto score_area = std::vector<int>(num_intersections_, kInvalid);
    ComputeScoreArea(score_area, scoring, territory_helper);

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

    if (scoring == kTerritory) {
        for (int y = 0; y < board_size_; ++y) {
            for (int x = 0; x < board_size_; ++x) {
                const auto vtx = GetVertex(x, y);
                const auto idx = GetIndex(x, y);
                if (score_area[idx] == kBlack ||
                        score_area[idx] == kWhite) {
                    if (GetState(vtx) == kBlack) {
                       --black_score_lead;
                    } else if (GetState(vtx) == kWhite) {
                        ++black_score_lead;
                    }
                }
            }
        }
        black_score_lead += prisoners_[kBlack];
        black_score_lead -= prisoners_[kWhite];
    }
    return color == kBlack ? black_score_lead : -black_score_lead;
}

void Board::ComputeReachArea(std::vector<int> &result) const {
    if (result.size() != (size_t) num_intersections_) {
        result.resize(num_intersections_);
    }
    auto black = std::vector<bool>(num_intersections_, false);
    auto white = std::vector<bool>(num_intersections_, false);

    auto PeekState = [&](int vtx) -> int {
        return state_[vtx];
    };

    // Compute black area.
    ComputeReachColor(kBlack, kEmpty, black, PeekState);

    // Compute white area.
    ComputeReachColor(kWhite, kEmpty, white, PeekState);

    for (int y = 0; y < board_size_; ++y) {
        for (int x = 0; x < board_size_; ++x) {
            const auto idx = GetIndex(x, y);
            const auto vtx = GetVertex(x, y);

            if (black[vtx] && !white[vtx]) {
                // The point is black.
                result[idx] = kBlack;
            } else if (white[vtx] && !black[vtx]) {
                // The white is white.
                result[idx] = kWhite;
            } else {
                //The point belongs to both.
                result[idx] = kEmpty;
            }
        }
    }
}

void Board::ComputeScoreArea(std::vector<int> &result,
                             const int scoring,
                             const std::vector<int> &territory_helper) const {
    if (scoring == kTerritory) {
        auto fork_board = new Board(*this);
        auto dead_list = std::vector<int>{};
        for (int y = 0; y < board_size_; ++y) {
            for (int x = 0; x < board_size_; ++x) {
                const auto vtx = GetVertex(x, y);
                const auto idx = GetIndex(x, y);
                if ((territory_helper[idx] == kBlack && GetState(vtx) == kWhite) ||
                        (territory_helper[idx] == kWhite && GetState(vtx) == kBlack)) {
                    dead_list.emplace_back(vtx);
                }
            }
        }
        fork_board->RemoveMarkedStrings(dead_list);
        fork_board->ComputeScoreArea(
            result, kArea, territory_helper);
        delete fork_board;
        return;
    }
    ComputeReachArea(result);
    auto pass_alive = std::vector<bool>(num_intersections_);

    for (int c: {kBlack, kWhite}) {

        std::fill(std::begin(pass_alive), std::end(pass_alive), false);
        ComputePassAliveArea(pass_alive, c, true, true);

        for (int i = 0; i < num_intersections_; ++i) {
            if (pass_alive[i]) {
                result[i] = c;
            }
        }
    }
}

std::vector<LadderType> Board::GetLadderMap() const {
    auto result = std::vector<LadderType>(num_intersections_, LadderType::kNotLadder);
    auto ladder = std::vector<int>{};
    auto not_ladder = std::vector<int>{};

    const auto VectorFind = [](std::vector<int> &arr, int element) -> bool {
        auto begin = std::begin(arr);
        auto end = std::end(arr);
        return std::find(begin, end, element) != end;
    };

    for (int y = 0; y < board_size_; ++y) {
        for (int x = 0; x < board_size_; ++x) {
            const auto idx = GetIndex(x, y);
            const auto vtx = GetVertex(x, y);

            if (state_[vtx] == kEmpty) {
                // It is not a string.
                 continue;
            }

            auto first_found = false;
            auto vital_moves = std::vector<int>{};
            int libs = 0;
            auto parent = strings_.GetParent(vtx);

            if (VectorFind(ladder, parent)) {
                // Be found! It is a ladder.
                libs = strings_.GetLiberty(parent);
            } else if (!VectorFind(not_ladder, parent)) {
                // Not be found! Now Search it.
                if (IsLadder(vtx, vital_moves)) {
                    // It is a ladder.
                    ladder.emplace_back(parent);
                    first_found = true;
                    libs = strings_.GetLiberty(parent);
                } else {
                    // It is not a ladder.
                    not_ladder.emplace_back(parent);
                    continue;
                }
            } else {
                // It is not a ladder.
                continue;
            }

            assert(libs == 1 || libs == 2);
            if (libs == 1) {
                // The ladder string is already death.
                result[idx] = LadderType::kLadderDeath;
            } else {
                // The ladder string has a chance to escape.
                result[idx] = LadderType::kLadderEscapable;
            }

            if (first_found) {
                for (const auto &v : vital_moves) {
                    const auto ax = GetX(v);
                    const auto ay = GetY(v);
                    const auto aidx = GetIndex(ax, ay);
                    if (libs == 1) {
                        // Someone can capture this ladder string.
                        result[aidx] = LadderType::kLadderTake;
                    } else {
                        // Someone can atari this ladder string.
                        result[aidx] = LadderType::kLadderAtari;
                    }
                }
            }
        }
    }

    return result;
}

void Board::ComputeSekiPoints(std::vector<bool> &result) const {
    for (int y = 0; y < board_size_; ++y) {
        for (int x = 0; x < board_size_; ++x) {
            const auto idx = GetIndex(x, y);
            const auto vtx = GetVertex(x, y);

            if (IsSeki(vtx)) {
                result[idx] = true;
            }
        }
    }
}

void Board::ComputeSafeArea(std::vector<bool> &result, bool mark_seki) const {
    if (result.size() != (size_t) num_intersections_) {
        result.resize(num_intersections_);
    }

    std::fill(std::begin(result), std::end(result), false);

    ComputePassAliveArea(result, kBlack, true, true);
    ComputePassAliveArea(result, kWhite, true, true);
    if (mark_seki) {
        ComputeSekiPoints(result);
    }
}

void Board::ComputePassAliveArea(std::vector<bool> &result,
                                 const int color,
                                 bool mark_vitals,
                                 bool mark_pass_dead) const {
    auto ocupied = std::vector<int>(num_vertices_, kInvalid);

    // Mark the color.
    for (int y = 0; y < board_size_; ++y) {
        for (int x = 0; x < board_size_; ++x) {
            const auto vtx = GetVertex(x, y);
            const auto state = GetState(vtx);
            if (state == color) {
                ocupied[vtx] = color;
            } else {
                ocupied[vtx] = kEmpty;
            }
        }
    }

    // empty regions list
    auto regions_index = std::vector<int>(num_vertices_, -1);
    auto regions_next = std::vector<int>(num_vertices_, kNullVertex);
    auto regions_head = ClassifyGroups(kEmpty, ocupied, regions_index, regions_next);
    auto vitals = std::vector<bool>(num_vertices_, false);

    // TODO: Do we need to think about sucide move?
    constexpr bool allow_sucide = false;

    // TODO: Do we need to compute potential vital regions here?

    // Compute the potential vital regions. That means that the region is
    // possible to becomes vital area for any adjacent strings.
    for (int vtx : regions_head) {
        bool success = true;
        int pos = vtx;
        do {
            bool is_vital = false;
            int state = allow_sucide == true ? ocupied[pos] : GetState(pos);

            assert(state != color);

            if (state == kEmpty) {
                for (int k = 0; k < 4; ++k) {
                    const auto apos = directions_[k] + pos;

                    // Empty point must be adjacent my string if the it is vital, Otherwise
                    // the point opp's potential eye.
                    if (ocupied[apos] == color) {
                        is_vital = true;
                        break;
                    }
                }
            } else if (state == !color) {
                // Opp's stone can not become opp's eye if we forbid the sucide move.
                is_vital = true;
            }

            if (!is_vital) {
                success = false;
                break;
            }
            pos = regions_next[pos];
        } while(pos != vtx);


        if (success) {
            int pos = vtx;
            do {
                vitals[pos] = true;
                pos = regions_next[pos];
            } while(pos != vtx);
        }
    }

    // my strings list
    auto strings_index = std::vector<int>(num_vertices_, -1);
    auto strings_next = std::vector<int>(num_vertices_, kNullVertex);
    auto strings_head = ClassifyGroups(color, ocupied, strings_index, strings_next);

    int group_cnt = strings_head.size();

    // Start the Benson's algorithm.
    // https://senseis.xmp.net/?BensonsAlgorithm
    while(true) {
        auto change = false;

        for (int i = 0; i < group_cnt; ++i) {
            const auto vtx = strings_head[i];

            if (!IsPassAliveString(vtx, allow_sucide, vitals, ocupied,
                                       regions_index, regions_next, strings_index, strings_next)) {
                // The string is not pass-alive. Remove the uncertainty life string.

                int pos = vtx;
                do {
                    strings_index[pos] = 0;
                    ocupied[pos] = kEmpty;
                    pos = strings_next[pos];

                    // The adjacent empty regions of removed string are not
                    // vital any more. Remove they.
                    for (int k = 0; k < 4; ++k) {
                        const auto apos = directions_[k] + pos;
                        if (vitals[apos]) {
                            int rpos = apos;
                            do {
                                vitals[rpos] = false;
                                rpos = regions_next[rpos];
                            } while(rpos != apos);
                        }
                    }
                } while(pos != vtx);

                // Remove the linking.
                std::remove(std::begin(strings_head),
                                std::end(strings_head), vtx);
                group_cnt -= 1;
                change = true;
                break;
            }
        }

        // The algorithm is over if there is no removed string.
        if (!change) break;
    }

    // Fill the pass-alive groups.
    for (int i = 0; i < group_cnt; ++i) {
        const auto vtx = strings_head[i];
        int pos = vtx;
        do {
            auto x = GetX(pos);
            auto y = GetY(pos);
            auto index = GetIndex(x, y);
            result[index] = true;

            pos = strings_next[pos];
        } while(pos != vtx);
    }

    // Fill the pass-alive vitals.
    if (mark_vitals) {
        for (int vtx : regions_head) {
            int pos = vtx;
            do {
                if (vitals[pos]) {
                    auto x = GetX(pos);
                    auto y = GetY(pos);
                    auto index = GetIndex(x, y);
                    result[index] = true;
                    ocupied[pos] = color;
                }
                pos = regions_next[pos];
            } while(pos != vtx);
        }
    }

    if (mark_pass_dead) {
        // Re-compute the regions for scanning pass-dead regions.
        regions_head = ClassifyGroups(kEmpty, ocupied, regions_index, regions_next);

        // Fill the pass dead regions.
        for (int vtx : regions_head) {
            int pos = vtx;
            if (IsPassDeadRegion(pos, !color, allow_sucide, ocupied, regions_next)) {
                do {
                    auto x = GetX(pos);
                    auto y = GetY(pos);
                    auto index = GetIndex(x, y);
                    result[index] = true;

                    pos = regions_next[pos];
                } while(pos != vtx);
            }
        }
    }
}

bool Board::IsPassAliveString(const int vtx,
                              bool allow_sucide,
                              const std::vector<bool> &vitals,
                              const std::vector<int> &features,
                              const std::vector<int> &regions_index,
                              const std::vector<int> &regions_next,
                              const std::vector<int> &strings_index,
                              const std::vector<int> &strings_next) const {
    auto vitals_list = std::set<int>{};
    int my_index = strings_index[vtx];
    int pos = vtx;
    do {
        for (int k = 0; k < 4; ++k) {
            const auto apos = directions_[k] + pos;
            if (vitals[apos]) {
                // This region is potential vital region for my string. Check it.
                int rpos = apos;
                bool success = true;
                do {
                    bool is_adjacent = false;
                    int state = allow_sucide == true ? features[rpos] : GetState(rpos);
                    if (state == kEmpty) {
                        for (int k = 0; k < 4; ++k) {
                            // Check that points of adjacent are empty.
                            const auto aapos = directions_[k] + rpos;
                            if(strings_index[aapos] == my_index) {
                                is_adjacent = true;
                                break;
                            }
                        }
                    } else {
                        is_adjacent = true;
                    }
                    if (!is_adjacent) {
                        // Not every empty points are adjacent to my string. The region
                        // is not vital.
                        success = false;
                        break;
                    }
                    rpos = regions_next[rpos];
                } while(rpos != apos);

                if (success) vitals_list.insert(regions_index[apos]);
            }
        }
        pos = strings_next[pos];
    } while(pos != vtx);

    // We say a string is pass-alive. There must be two or more
    // vitals adjacent to it.
    return vitals_list.size() >= 2;
}

bool Board::IsPassDeadRegion(const int vtx,
                             const int color,
                             bool allow_sucide,
                             std::vector<int> &features,
                             const std::vector<int> &regions_next) const {
    const auto IsPotentialEye = [this](const int vertex,
                                       const int color,
                                       bool allow_sucide,
                                       std::vector<int> &features,
                                       std::vector<bool> &inner_regions) {
        // This is greedy algorithm, we only promise that the position is not
        // potential eye if it returns false. It is possible that the position
        // is fake eye even if it returns true.

        if (!allow_sucide && GetState(vertex) == color) {
            // My stone can not become to my eye if we forbid the sucide move.
            return false;
        }

        // The potential eye is possible to become the real eye in the postion if
        // four adjacent point is mine or empty.
        std::array<int, 4> side_count = {0, 0, 0, 0};

        for (int k = 0; k < 4; ++k) {
            const auto avtx = vertex + directions_[k];
            side_count[features[avtx]]++;
        }

        if (side_count[!color] != 0) {
            return false;
        }

        // The potential eye is possible to become the real eye in the postion if
        // three adjacent corner is mine or empty or out of border.
        std::array<int, 4> corner_count = {0, 0, 0, 0};

        for (int k = 4; k < 8; ++k) {
            const auto avtx = vertex + directions_[k];
            if (inner_regions[avtx]) { // The inner region corner is mine.
                corner_count[color]++;
            } else {
                corner_count[features[avtx]]++;
            }
        }
        if (corner_count[kInvalid] == 0) {
            if (corner_count[!color] > 1) {
                return false;
            }
        } else {
            if (corner_count[!color] > 0) {
                return false;
            }
        }
        return true;
    };

    // The inner region is a region surrounded by a string. On the other head,
    // it does not' touch' the border of the board.
    auto inner_regions = std::vector<bool>(features.size(), false);

    // The inner region may cause the false-eye life (Two-Headed Dragon). The
    // false eyes will become the potential eyes in this condition.
    //
    // false-eye life: https://senseis.xmp.net/?TwoHeadedDragon
    ComputeInnerRegions(vtx, color, regions_next, inner_regions);

    auto potential_eyes = std::vector<int>{};
    int pos = vtx;
    do {
        // Search all potential eyes in this region.
        if (IsPotentialEye(pos, color, allow_sucide, features, inner_regions)) {
            potential_eyes.emplace_back(pos);
        }
        pos = regions_next[pos];
    } while(pos != vtx);

    int eyes_count = potential_eyes.size();

    if (eyes_count == 2) {
        // It is possible to be pass-dead if there are only two potential eyes. The
        // case is two eyes are adjacent to each others.
        //
        // ..ox..
        // ooox..  // two potential eyes are adjacent to each others,
        // xxxx..  // white string is only one eye.
        // ......

        if (IsNeighbor(potential_eyes[0], potential_eyes[1])) {
            eyes_count -= 1;
        }
    }

    // We say a string is pass-dead if the maximum potential eye is lower than 2.
    return eyes_count < 2;
}

void Board::ComputeInnerRegions(const int vtx,
                                const int color,
                                const std::vector<int> &regions_next,
                                std::vector<bool> &inner_regions) const {
    auto surround = std::vector<int>(num_vertices_, kInvalid);

    std::fill(std::begin(inner_regions), std::end(inner_regions), false);

    for (int y = 0; y < board_size_; ++y) {
        for (int x = 0; x < board_size_; ++x) {
            surround[GetVertex(x, y)] = kEmpty;
        }
    }

    int pos = vtx;
    do {
        surround[pos] = !color;
        pos = regions_next[pos];
    } while(pos != vtx);

    auto epmty_index = std::vector<int>(num_vertices_, -1);
    auto epmty_next = std::vector<int>(num_vertices_, kNullVertex);
    auto epmty_head = ClassifyGroups(kEmpty, surround, epmty_index, epmty_next);

    int cnt = epmty_head.size();
    for (int i = 0 ; i < cnt; ++i) {
        int v = epmty_head[i];
        pos = v;
        do {
            bool success = false;
            for (int k = 0; k < 4; ++k) {
                const auto apos = pos + directions_[k];
                if (surround[apos] == kInvalid) {
                    success = true;
                    break;
                }
            }
            if (success) {
                std::remove(std::begin(epmty_head),
                                std::end(epmty_head), v);
                cnt -= 1;
                break;
            }
            pos = epmty_next[pos];
        } while(pos != v);
    }

    for (int i = 0 ; i < cnt; ++i) {
        int v = epmty_head[i];
        pos = v;
        do {
            inner_regions[pos] = true;
            pos = epmty_next[pos];
        } while(pos != v);
    }
}

std::vector<int> Board::ClassifyGroups(const int target,
                                       std::vector<int> &features,
                                       std::vector<int> &regions_index,
                                       std::vector<int> &regions_next) const {
    // Set out of border area as -1.
    std::fill(std::begin(regions_index), std::end(regions_index), -1);

    // Set out of border area as kNullVertex.
    std::fill(std::begin(regions_next), std::end(regions_next), kNullVertex);

    // All invalid strings (groups) 's index are 0.
    for (int y = 0; y < board_size_; ++y) {
        for (int x = 0; x < board_size_; ++x) {
            const auto vtx = GetVertex(x, y);
            regions_index[vtx] = 0;
            regions_next[vtx] = vtx;
        }
    }

    auto head_list = std::vector<int>{}; // all string heads vertex postion
    auto marked = std::vector<bool>(num_vertices_, false); // true if the vertex is usesd
    auto groups_index = 1; // valid index is from 1.

    for (int y = 0; y < board_size_; ++y) {
        for (int x = 0; x < board_size_; ++x) {
            const auto vtx = GetVertex(x, y);

            if (!marked[vtx] && features[vtx] == target) {
                auto buf = std::vector<bool>(num_vertices_, false);

                // Gather all vertices which connect with head vertex.
                ComputeReachGroup(vtx, target, buf, [&](int v){ return features[v]; });

                auto vertices = GatherVertices(buf);
                auto next_vertex = kNullVertex;

                // Link this string.
                for (const auto v : vertices) {
                    regions_next[v] = next_vertex;
                    regions_index[v] = groups_index;
                    marked[v] = true;
                    next_vertex = v;
                }
                if (!vertices.empty()) {
                    regions_next[vertices[0]] = next_vertex;
                }

                // Gather this string head.
                groups_index += 1;
                head_list.emplace_back(vtx);
            }
        }
    }
    return head_list;
}

std::vector<int> Board::GatherVertices(std::vector<bool> &buf) const {
    auto result = std::vector<int>{};

    for (auto vtx = size_t{0}; vtx < buf.size(); ++vtx) {
        if (buf[vtx]) {
            result.emplace_back(vtx);
        }
    }

    return result;
}

void Board::GenerateCandidateMoves(std::vector<int> &moves_set, int color) const {
    auto buf = std::vector<int>{};

    for (const auto vtx : {last_move_, last_move_2_}) {
        if (vtx != kPass && vtx != kNullVertex) {
            const auto center_color = state_[vtx];

            if (center_color != kEmpty &&
                    GetLiberties(vtx) <= 2) {
                FindStringLiberties(vtx, buf);
            }
            for (int k = 0; k < 8; ++k) {
                const auto avtx = vtx + directions_[k];

                if (state_[avtx] == kEmpty) {
                    buf.emplace_back(avtx);
                } else if (center_color != kEmpty &&
                               state_[avtx] == !center_color) {
                    if (GetLiberties(avtx) <= 2) {
                        FindStringLiberties(avtx, buf);
                    }
                }
            }
        }
    }

    // Remove the repetition vertices.
    std::sort(std::begin(buf), std::end(buf));
    buf.erase(std::unique(std::begin(buf), std::end(buf)),
                  std::end(buf));

    for (const auto vtx : buf) {
        if (IsLegalMove(vtx, color) &&
                !(IsSimpleEye(vtx, color) &&
                     !IsCaptureMove(vtx, color)&&
                     !IsEscapeMove(vtx, color))) {
            moves_set.emplace_back(vtx);
        }
    }

    // TODO: Append others heuristic moves.
}

std::string Board::GetMoveTypesString(int vtx, int color) const {
    auto out = std::ostringstream{};
    out << '{';
    int i = 0;

    if (IsCaptureMove(vtx, color)) {
        if (i++ != 0) out << ", ";
        out << "Capture";
    }
    if (IsEscapeMove(vtx, color)) {
        if (i++ != 0) out << ", ";
        out << "Escape";
    }
    if (IsRealEye(vtx, color)) {
        if (i++ != 0) out << ", ";
        out << "Real Eye";
    }
    if (IsSimpleEye(vtx, color)) {
        if (i++ != 0) out << ", ";
        out << "Eye Shape";
    }
    if (IsSeki(vtx)) {
        if (i++ != 0) out << ", ";
        out << "Seki";
    }
    if (IsAtariMove(vtx, color)) {
        if (i++ != 0) out << ", ";
        out << "Atari";
    }
    if (IsSelfAtariMove(vtx, color)) {
        if (i++ != 0) out << ", ";
        out << "Self Atari";
    }

    for (int k = 0; k < 4; ++k) {
        auto vital_moves = std::vector<int>{};
        const auto avtx = vtx + directions_[k];

        if (IsLadder(avtx, vital_moves)) {
            auto libs = GetLiberties(avtx);
            if (libs == 1 && state_[avtx] == color) {
                if (i++ != 0) out << ", ";
                out << "Ladder Dead";
            } else if (libs == 1 && state_[avtx] != color) {
                if (i++ != 0) out << ", ";
                out << "Ladder Capture";
            } else if (libs == 2 && state_[avtx] == color) {
                if (i++ != 0) out << ", ";
                out << "Ladder Atari";
            } else if (libs == 2 && state_[avtx] != color) {
                if (i++ != 0) out << ", ";
                out << "Ladder Escape";
            }
        }
    }

    out << '}';
    return out.str();
}
