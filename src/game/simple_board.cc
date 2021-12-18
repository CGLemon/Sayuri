#include <sstream>
#include <queue>
#include <algorithm>

#include "game/simple_board.h"
#include "game/symmetry.h"

void SimpleBoard::Reset(const int boardsize) {
    SetBoardSize(boardsize);

    ResetBoard();
    ResetBasicData();
}

void SimpleBoard::ResetBoard() {
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

std::string SimpleBoard::GetHashingString() const {
    auto out = std::ostringstream{};
    out << std::hex;
    out << "Hash: " << GetHash() << " | ";
    out << "Ko Hash: " << GetKoHash();
    out << std::dec << std::endl;
    return out.str();
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
    out << GetHashingString();

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

void SimpleBoard::RemoveMarkedStrings(std::vector<int> &marked) {
    int removed_stones[2] = {0, 0};
    for (auto &vtx : marked) {
        auto color = GetState(vtx);
        if (color == kBlack || color == kWhite) {
            removed_stones[color] += RemoveString(vtx);
        }
    }

    IncrementPrisoner(kBlack, removed_stones[kBlack]);
    IncrementPrisoner(kWhite, removed_stones[kWhite]);
}

int SimpleBoard::ComputeReachGroup(int start_vertex, int spread_color, std::vector<bool> &buf) const {
    auto PeekState = [&](int vtx) -> int {
        return state_[vtx];
    };

    return ComputeReachColor(start_vertex, spread_color, buf, PeekState);
}

int SimpleBoard::ComputeReachGroup(int start_vertex, int spread_color,
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

void SimpleBoard::FindStringSurround(const int vtx,
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

    assert(GetState(avtx) == color);
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

int SimpleBoard::FindStringLiberties(const int vtx,
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

int SimpleBoard::FindStringLibertiesGainingCaptures(const int vtx,
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

std::pair<int, int> SimpleBoard::GetLadderLiberties(const int vtx, const int color) const {
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

LadderType SimpleBoard::PreySelections(const int prey_color,
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
    const int move = selections[0];

    num_move += FindStringLibertiesGainingCaptures(ladder_vtx, selections);

    // The moves must be the legal.
    selections.erase(
        std::remove_if(std::begin(selections), std::end(selections),
            [&](int v) { return !IsLegalMove(v, prey_color); 
        }),
        std::end(selections)
    );

    num_move = selections.size();

    // If there is no legal move, the ladder string die.
    if (num_move == 0) {
        return LadderType::kGoodForHunter; 
    }

    if (selections[0] == move) {
        auto bound = GetLadderLiberties(move, prey_color);
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

LadderType SimpleBoard::HunterSelections(const int prey_color,
                                         const int ladder_vtx, std::vector<int>& selections) const {
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
        size_t size = 0;
        const int hunter_color = (!prey_color);
        const int libs_1 = CountPliberties(move_1); 
        const int libs_2 = CountPliberties(move_2); 

        if (libs_1 >= 3 && libs_2 >= 3) {
            // A ladder string must be only two liberty.
            return LadderType::kGoodForPrey;
        } else if (libs_1 >= 3) {
            if (IsLegalMove(move_1, hunter_color)) {
                selections.emplace_back(move_1);
                size++;
            }
        } else if (libs_2 >= 3) {
            if (IsLegalMove(move_2, hunter_color)) {
                selections.emplace_back(move_2);
                size++;
            }
        } else {
            if (IsLegalMove(move_1, hunter_color)) {
                selections.emplace_back(move_1);
                size++;
            }
            if (IsLegalMove(move_2, hunter_color)) {
                selections.emplace_back(move_2);
                size++;
            }
        }
    }

    if (selections.empty()) {
        // The hunter has no atari move.
        return LadderType::kGoodForPrey;
    }

    return LadderType::kGoodForNeither; // keep running
}

LadderType SimpleBoard::PreyMove(std::shared_ptr<SimpleBoard> board,
                                 const int hunter_vtx, const int prey_color,
                                 const int ladder_vtx, size_t& ladder_nodes, bool fork) const {

    if ((++ladder_nodes) >= kMaxLadderNodes) {
        // If hit the limit, assume prey have escaped. 
        return LadderType::kGoodForPrey;
    }

    std::shared_ptr<SimpleBoard> ladder_board;
    if (fork) {
        ladder_board = std::make_shared<SimpleBoard>(*board);
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

    return best;
}

LadderType SimpleBoard::HunterMove(std::shared_ptr<SimpleBoard> board,
                                   const int prey_vtx, const int prey_color,
                                   const int ladder_vtx, size_t& ladder_nodes, bool fork) const {
    if ((++ladder_nodes) >= kMaxLadderNodes) {
        // If hit the limit, assume prey have escaped. 
        return LadderType::kGoodForPrey;
    }

    std::shared_ptr<SimpleBoard> ladder_board;
    if (fork) {
        ladder_board = std::make_shared<SimpleBoard>(*board);
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

    return best;
}

bool SimpleBoard::IsLadder(const int vtx) const {
    if (vtx == kPass) {
        return false;
    }

    const int prey_color = GetState(vtx);
    if (prey_color == kEmpty || prey_color == kInvalid) {
        return false;
    }

    const int libs = strings_.GetLiberty(strings_.GetParent(vtx));
    const int ladder_vtx = vtx;
    size_t searched_nodes = 0;
    auto res = LadderType::kGoodForNeither;
    if (libs == 1) {
        auto ladder_board = std::make_shared<SimpleBoard>(*this);
        res = PreyMove(ladder_board,
                       kNullVertex, prey_color,
                       ladder_vtx, searched_nodes, false);
    } else if (libs == 2) {
        auto ladder_board = std::make_shared<SimpleBoard>(*this);
        res = HunterMove(ladder_board,
                         kNullVertex, prey_color,
                         ladder_vtx, searched_nodes, false);
    } else if (libs >= 3) {
        res = LadderType::kGoodForPrey;
    }

    assert(res != LadderType::kGoodForNeither);
    return res == LadderType::kGoodForHunter;
}

bool SimpleBoard::IsAtariMove(const int vtx, const int color) const {
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

bool SimpleBoard::IsCaptureMove(const int vtx, const int color) const {
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

bool SimpleBoard::IsEscapeMove(const int vtx, const int color) const {
    if (IsSuicide(vtx, color)) {
        return false;
    }

    return IsCaptureMove(vtx, !color);
}

bool SimpleBoard::IsNeighbor(const int vtx, const int avtx) const {
    for (int k = 0; k < 4; ++k) {
        if ((vtx + directions_[k]) == avtx) {
             return true;
        }
    }
    return false;
}

bool SimpleBoard::IsSimpleEye(const int vtx, const int color) const {
    return neighbours_[vtx] & kEyeMask[color];
}

bool SimpleBoard::IsRealEye(const int vtx, const int color) const {
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

bool SimpleBoard::IsSeki(const int vtx) const {
    // This function is not always correct. Some cases, like Hanezeki, will be
    // missed. You can watch the webside to get the more seki information.
    // https://senseis.xmp.net/?Seki

    if (state_[vtx] != kEmpty) {
        return false;
    }

    int string_parent[2] = {kNullVertex, kNullVertex}; 

    for (auto k = 0; k < 4; ++k) {
        const auto avtx = vtx + directions_[k];
        const auto ip = strings_.GetParent(avtx);
        const auto state = state_[avtx];

        if (state == kEmpty) {
            return false;
        }

        if (state != kInvalid) {
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
            // Must be two strings.
            return false;
        }

        if (strings_.GetLiberty(ip) != 2) {
            // Two strings must be 2 liberties.
            return false;
        }
    }

    auto lib_buf = std::vector<int>{};
    auto black_idx_buf = std::vector<int>{};
    auto white_idx_buf = std::vector<int>{};

    FindStringSurround(string_parent[kBlack], kBlack, lib_buf, black_idx_buf);
    FindStringSurround(string_parent[kWhite], kWhite, lib_buf, white_idx_buf);
 
    assert(lib_buf.size() == 2 || lib_buf.size() == 3);

    if (lib_buf.size() == 3) {
        // We simply think that it is seki in this case. It includes false-seki. The false-seki
        // string is not alive, but in the most case, we don't need to move the false-seki point
        // to kill it.
        //
        // .x.ox..
        // oxoox..
        // .ooxx..
        // ooxx...
        // xxx....
        return true;
    }

    auto inner_color = kInvalid;
    if (black_idx_buf.size() == 1) {
        inner_color = kBlack;
    } else if (white_idx_buf.size() == 1) {
        inner_color = kWhite;
    }

    // The inner mean the potential seki string surrounded by the other string. The
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
    // The others are seki with partly filled eye space case.

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

bool SimpleBoard::IsBoundary(const int vtx) const {
    for (int k = 0; k < 4; ++k) {
        if (state_[vtx + directions_[k]] == kInvalid) return true;
    }
    return false;
}

bool SimpleBoard::IsKillableSekiEyeShape(const int vtx,
                                             const int eye_size,
                                             const std::vector<int> &eye_next) const {
    if (eye_size <= 3) {
        // It always died.
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
        if (IsBoundary(pos)) {
            ++boundary_cnt;
        }
        pos = eye_next[pos];
    } while (pos != vtx);

    auto nakade_vtx = std::vector<int>{};
    auto potential_eyes = std::vector<std::vector<int>>{};
    pos = vtx;

    // Mark the nakade moves and its potential eyes.
    do {
        int cnt = 0;
        auto p_eyes = std::vector<int>{};

        for (int k = 0; k < 8; ++k) {
            const auto apos = pos + directions_[k];
            if (eye_region[apos]) {
                ++cnt;
                if (k >= 4) {
                    // The potential eyes are in the diagonal.
                    p_eyes.emplace_back(apos);
                }
            }
        }

        // Nakade move influences all empty point, try to avoid the points to become
        // the eyes.
        if (cnt+1 == eye_size && !p_eyes.empty()) {
            nakade_vtx.emplace_back(pos);
            potential_eyes.emplace_back(p_eyes);
        }

        pos = eye_next[pos];
    } while (pos != vtx);

    const int nakade_cnt = nakade_vtx.size();
    if (nakade_cnt == 0) {
        // No nakade move. It is alive.
        return false;
    }

    for (const auto &e: potential_eyes) {
        // No potential eyes. It died.
        if (e.empty()) return true;
    }

    const auto GetEmptySideCount = [this](const int eye_vtx,
                                              std::vector<bool> &eye_region) {
        int side_cnt = 0;
        for (int k = 0; k < 4; ++k) {
            if (eye_region[eye_vtx + directions_[k]])
                ++side_cnt;
        }
        return side_cnt;
    };

    // Possible eye shape is here: https://senseis.xmp.net/?EyeShape
    if (eye_size == 4) {
        assert(nakade_cnt == 1);
        // Only bent four, Dogleg four and Squared four cases are here.

        if (boundary_cnt == 4) {
            // Bent four in the corner. It died.
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
            // Squared four. Obviously It died.
            //
            // ..ox...
            // ..ox...
            // ooox...
            // xxxx...
            // .......

            return true;
        }

        // It is Dogleg four case and is also killable eye shape. But it doesn't exsit.
        // We donl't need to conside it.
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
            // Bulky Five. It died.
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
                // Rabbitty six. It died.
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
                // Rectangular six in the corner. It died.
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

    return false;
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

void SimpleBoard::RemoveStone(const int vtx, const int color) {
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

void SimpleBoard::MergeStrings(const int ip, const int aip) {
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

void SimpleBoard::IncrementPrisoner(const int color, const int val) {
    const int old_prisoners = prisoners_[color];
    const int new_prisoners = old_prisoners + val;
    prisoners_[color] = new_prisoners;
    UpdateZobristPrisoner(color, new_prisoners, old_prisoners);
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

        IncrementPrisoner(!color, sucide_stones);
    }

    if (captured_stones != 0) {
        IncrementPrisoner(color, captured_stones);
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

std::uint64_t SimpleBoard::GetKoHash() const {
    return ko_hash_;
}

std::uint64_t SimpleBoard::GetHash() const {
    return hash_;
}

int SimpleBoard::GetState(const int vtx) const {
    return state_[vtx];
}

int SimpleBoard::GetState(const int x, const int y) const {
    return GetState(GetVertex(x,y));
}

int SimpleBoard::GetLiberties(const int vtx) const {
    return strings_.GetLiberty(strings_.GetParent(vtx));
}

int SimpleBoard::GetStones(const int vtx) const {
    return strings_.GetStones(strings_.GetParent(vtx));
}

int SimpleBoard::GetEmptyCount() const {
    return empty_cnt_;
}

int SimpleBoard::GetEmpty(const int idx) const {
    return empty_[idx];
}

std::vector<int> SimpleBoard::GetStringList(const int vtx) const {
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
