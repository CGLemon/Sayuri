#include "game/game_state.h"
#include "game/types.h"
#include "utils/parser.h"
#include "utils/log.h"
#include "utils/random.h"
#include "utils/komi.h"
#include "pattern/pattern.h"
#include "pattern/gammas_dict.h"

#include <random>

void GameState::Reset(const int boardsize, const float komi) {
    board_.Reset(boardsize);
    SetKomi(komi);
    ko_hash_history_.clear();
    game_history_.clear();
    append_moves_.clear();

    ko_hash_history_.emplace_back(GetKoHash());
    game_history_.emplace_back(std::make_shared<Board>(board_));

    winner_ = kUndecide;
    handicap_ = 0;
    move_number_ = 0;
}

void GameState::ClearBoard() {
    Reset(GetBoardSize(), GetKomi());
}

void GameState::PlayMoveFast(const int vtx, const int color) {
    if (vtx != kResign) {
        board_.PlayMoveAssumeLegal(vtx, color);
        move_number_++;
    }
}

bool GameState::AppendMove(const int vtx, const int color) {
    if (vtx == kResign || vtx == kPass) {
        return false;
    }
    if (move_number_ != 0) {
        ClearBoard(); // Besure sure that it is the first move.
    }

    if (IsLegalMove(vtx, color)) {
        board_.PlayMoveAssumeLegal(vtx, color);
        board_.SetToMove(kBlack);
        board_.SetLastMove(kNullVertex, kNullVertex);

        move_number_ = 0;
        ko_hash_history_.clear();
        game_history_.clear();

        ko_hash_history_.emplace_back(GetKoHash());
        game_history_.emplace_back(std::make_shared<Board>(board_));
        append_moves_.emplace_back(vtx, color);

        return true;
    }
    return false;
}

bool GameState::PlayMove(const int vtx) {
    return PlayMove(vtx, GetToMove());
}

bool GameState::PlayMove(const int vtx, const int color) {
    if (vtx == kResign) {
        if (color == kBlack) {
            winner_ = kWhiteWon;
        } else {
            winner_ = kBlackWon;
        }
        return true;
    }

    if (IsLegalMove(vtx, color)) {
        board_.PlayMoveAssumeLegal(vtx, color);
        move_number_++;

        // Cut off unused history.
        ko_hash_history_.resize(move_number_);
        game_history_.resize(move_number_);

        ko_hash_history_.emplace_back(GetKoHash());
        game_history_.emplace_back(std::make_shared<Board>(board_));

        return true;
    }
    return false;
}

bool GameState::UndoMove() {
    if (move_number_ >= 1) {
        // Cut off unused history.
        ko_hash_history_.resize(move_number_);
        game_history_.resize(move_number_);

        board_ = *game_history_[move_number_-1];

        winner_ = kUndecide;
        move_number_--;

        return true;
    }
    return false;
}

int GameState::TextToVertex(std::string text) const {
    if (text.size() < 2) {
        return kNullVertex;
    }

    if (text == "PASS" || text == "pass") {
        return kPass;
    } else if (text == "RESIGN" || text == "resign") {
        return kResign;
    }

    int x = -1;
    int y = -1;
    if (text[0] >= 'a' && text[0] <= 'z') {
        x = text[0] - 'a';
        if (text[0] >= 'i')
            x--;
    } else if (text[0] >= 'A' && text[0] <= 'Z') {
        x = text[0] - 'A';
        if (text[0] >= 'I')
            x--;
    }
    auto y_str = std::string{};
    auto skip = bool{false};
    std::for_each(std::next(std::begin(text), 1), std::end(text),
                      [&](const auto in) -> void {
                          if (skip) {
                              return;
                          }
                          if (in >= '0' && in <= '9') {
                              y_str += in;
                          } else {
                              y_str = std::string{};
                              skip = true;
                          }
                      });
    if (!y_str.empty()) {
        y = std::stoi(y_str) - 1;
    }

    if (x == -1 || y == -1) {
        return kNullVertex;
    }
    return board_.GetVertex(x, y);
}

int GameState::TextToColor(std::string text) const {
    auto lower = text;
    for (auto & c: lower) {
        c = std::tolower(c);
    }

    if (lower == "b" || lower == "black") {
        return kBlack;
    } else if (lower == "w" || lower == "white") {
        return kWhite;
    }
    return kInvalid;
}

std::string GameState::VertexToSgf(const int vtx) const {
    assert(vtx != kNullVertex);

    if (vtx == kPass || vtx == kResign) {
        return std::string{};
    }

    auto out = std::ostringstream{};    
    const auto x = GetX(vtx);
    const auto y = GetY(vtx);

    if (x >= 26) {
        out << static_cast<char>(x - 26 + 'A');
    } else {
        out << static_cast<char>(x + 'a');
    }

    if (y >= 26) {
        out << static_cast<char>(y - 26 + 'A');
    } else {
        out << static_cast<char>(y + 'a');
    }

    return out.str();
}

std::string GameState::VertexToText(const int vtx) const {
    assert(vtx != kNullVertex);

    if (vtx == kPass) {
        return "pass";
    }
    if (vtx == kResign) {
        return "resign";
    }

    auto out = std::ostringstream{};    
    const auto x = GetX(vtx);
    const auto y = GetY(vtx);

    auto offset = 0;
    if (static_cast<char>(x + 'A') >= 'I') {
        offset = 1;
    }

    out << static_cast<char>(x + offset + 'A');
    out << y+1;

    return out.str();
}

bool GameState::PlayTextMove(std::string input) {
    int color = kInvalid;
    int vertex = kNullVertex;

    auto parser = CommandParser(input);

    if (parser.GetCount() == 2) {
        const auto color_str = parser.GetCommand(0)->Get<std::string>();
        const auto vtx_str = parser.GetCommand(1)->Get<std::string>();
        color = TextToColor(color_str);
        vertex = TextToVertex(vtx_str);
    } else if (parser.GetCount() == 1) {
        const auto vtx_str = parser.GetCommand(0)->Get<std::string>();
        color = board_.GetToMove();
        vertex = TextToVertex(vtx_str);
    }

    if (color == kInvalid || vertex == kNullVertex) {
        return false;
    }

    return PlayMove(vertex, color);
}

void GameState::ShowMoveTypes(int vtx, int color) const {
    if (color == kBlack) {
        LOGGING << "Black ";
    } else {
        LOGGING << "White ";
    }
    LOGGING << VertexToText(vtx) << ' ' << board_.GetMoveTypesString(vtx, color) << '\n';
}

std::string GameState::GetStateString() const {
    auto out = std::ostringstream{};
    out << "{";
    out << "Next Player: ";
    if (GetToMove() == kBlack) {
        out << "Black";
    } else if (GetToMove() == kWhite) {
        out << "White";
    } else {
        out << "Error";
    }
    out << ", ";
    out << "Move Number: " << move_number_ << ", ";
    out << "Komi: " << GetKomi() << ", ";
    out << "Board Size: " << GetBoardSize() << ", ";
    out << "Handicap: " << GetHandicap();

    out << "}" << std::endl;
    return out.str();
}

void GameState::ShowBoard() const {
    LOGGING << board_.GetBoardString(board_.GetLastMove(), true);
    LOGGING << GetStateString();
}

void GameState::SetWinner(GameResult result) {
    winner_ = result;
}

void GameState::SetKomi(float komi) {
    bool negative = komi < 0.f;
    if (negative) {
        komi = -komi;
    }

    int integer_part = static_cast<int>(komi);
    float float_part = komi - static_cast<float>(integer_part);

    if (IsSameKomi(float_part, 0.f)) {
        komi_half_ = false;
    } else if (IsSameKomi(float_part, 0.5f)) {
        komi_half_ = true;
    } else {
        LOGGING << "Only accept for integer komi or half komi." << std::endl;
        return;
    }

    komi_negative_ = negative;
    komi_integer_ = integer_part;

    komi_hash_ = Zobrist::kKomi[komi_integer_];
    if (komi_negative_) {
        komi_hash_ ^= Zobrist::kNegativeKomi;
    }
    if (komi_half_) {
        komi_hash_ ^= Zobrist::kHalfKomi;
    }
}

void GameState::SetToMove(const int color) {
    board_.SetToMove(color);
}

void GameState::SetHandicap(int handicap) {
    handicap_ = handicap;
}

bool GameState::IsGameOver() const {
    return winner_ != kUndecide || GetPasses() >= 2;
}

bool GameState::IsSuperko() const {
    auto first = std::crbegin(ko_hash_history_);
    auto last = std::crend(ko_hash_history_);

    auto res = std::find(++first, last, GetKoHash());

    return res != last;
}

bool GameState::IsLegalMove(const int vertex) const {
    return board_.IsLegalMove(vertex, GetToMove());
}

bool GameState::IsLegalMove(const int vertex, const int color) const {
    return board_.IsLegalMove(vertex, color);
}

bool GameState::IsLegalMove(const int vertex, const int color,
                            std::function<bool(int, int)> AvoidToMove) const {
    return board_.IsLegalMove(vertex, color, AvoidToMove);
}

bool GameState::SetFixdHandicap(int handicap) {
    const auto ValidHandicap = [](int bsize, int handicap) {
        if (handicap < 2 || handicap > 9) {
            return false;
        }
        if (bsize % 2 == 0 && handicap > 4) {
            return false;
        }
        if (bsize == 7 && handicap > 4) {
            return false;
        }
        if (bsize < 7 && handicap > 0) {
            return false;
        }
        return true;
    };

    const int board_size = GetBoardSize();
    const int high = board_size >= 13 ? 3 : 2;
    const int mid = board_size / 2;
    const int low = board_size - 1 - high;

    if (!ValidHandicap(board_size, handicap)) {
        return false;
    }

    if (handicap >= 2) {
        AppendMove(GetVertex(low, low),  kBlack);
        AppendMove(GetVertex(high, high),  kBlack);
    }

    if (handicap >= 3) {
        AppendMove(GetVertex(high, low), kBlack);
    }

    if (handicap >= 4) {
        AppendMove(GetVertex(low, high), kBlack);
    }

    if (handicap >= 5 && handicap % 2 == 1) {
        AppendMove(GetVertex(mid, mid), kBlack);
    }

    if (handicap >= 6) {
        AppendMove(GetVertex(low, mid), kBlack);
        AppendMove(GetVertex(high, mid), kBlack);
    }

    if (handicap >= 8) {
        AppendMove(GetVertex(mid, low), kBlack);
        AppendMove(GetVertex(mid, high), kBlack);
    }

    SetHandicap(handicap);
    SetToMove(kWhite);

    return true;
}

bool GameState::SetFreeHandicap(std::vector<std::string> movelist) {
    auto movelist_vertex = std::vector<int>(movelist.size());
    std::transform(std::begin(movelist), std::end(movelist), std::begin(movelist_vertex),
                       [this](auto text){
                           return TextToVertex(text);
                       }
                   );

    auto fork_state = *this;

    for (const auto vtx : movelist_vertex) {
        if (fork_state.IsLegalMove(vtx, kBlack)) {
            fork_state.AppendMove(vtx, kBlack);
        } else {
            return false;
        }
    }

    *this = fork_state;

    SetHandicap(movelist.size());
    SetToMove(kWhite);

    return true;
}

std::vector<int> GameState::PlaceFreeHandicap(int handicap) {
    auto stone_list = std::vector<int>{};
    if (SetFixdHandicap(handicap)) {
        for (auto m : append_moves_) {
            stone_list.emplace_back(m.first);
        }
    }
    return stone_list;
}

std::vector<int> GameState::GetOwnership() const {
    auto res = std::vector<int>(GetNumIntersections(), kInvalid);

    board_.ComputeScoreArea(res);

    return res;
}

void GameState::FillRandomMove() {
    const int color = GetToMove();
    const int empty_cnt = board_.GetEmptyCount();
    const int rand = Random<kXoroShiro128Plus>::Get().Generate() % empty_cnt;
    int select_move = kPass;

    auto filled_area = std::vector<int>(GetNumIntersections(), kInvalid);
    auto safe_area = std::vector<bool>(GetNumIntersections(), false);

    board_.ComputeScoreArea(filled_area);
    board_.ComputeSafeArea(safe_area, true);

    for (int i = 0; i < empty_cnt; ++i) {
        const auto rand_pick = (rand + i) % empty_cnt;
        const auto vtx = board_.GetEmpty(rand_pick);

        if (!IsLegalMove(vtx, color)) {
            continue;
        }
        auto x = GetX(vtx);
        auto y = GetY(vtx);

        if (safe_area[GetIndex(x, y)]) {
            continue;
        }

        if (board_.IsCaptureMove(vtx, color)) {
            select_move = vtx;
            break;
        }
    }

    for (int i = 0; i < empty_cnt; ++i) {
        if (select_move != kPass) break;

        const auto rand_pick = (rand + i) % empty_cnt;
        const auto vtx = board_.GetEmpty(rand_pick);

        if (!IsLegalMove(vtx, color)) {
            continue;
        }

        if (board_.IsRealEye(vtx, color)) {
            continue;
        }

        auto x = GetX(vtx);
        auto y = GetY(vtx);

        if (safe_area[GetIndex(x, y)]) {
            continue;
        }

        if (board_.IsSimpleEye(vtx, color) &&
                !board_.IsCaptureMove(vtx, color) &&
                !board_.IsEscapeMove(vtx, color)) {
            continue;
        }

        if (filled_area[GetIndex(x, y)] != kEmpty) {
            continue;
        }

        select_move = vtx;
    }

    PlayMoveFast(select_move, color);
}

void GameState::PlayRandomMove() {
    auto candidate_moves = std::vector<int>{};
    const int color = GetToMove();

    board_.GenerateCandidateMoves(candidate_moves, color);
    std::shuffle(std::begin(candidate_moves),
                     std::end(candidate_moves),
                     Random<kXoroShiro128Plus>::Get());

    //TODO: Use the switch-case to wrap it.

    if (Random<kXoroShiro128Plus>::Get().Roulette<10000>(0.90f)) {
        // ~90%: capture
        for (const auto vtx : candidate_moves) {
            if (board_.IsCaptureMove(vtx, color)) {
                PlayMoveFast(vtx, color);
                return;
            }
        }
    }
    if (Random<kXoroShiro128Plus>::Get().Roulette<10000>(0.95f)) {
        // ~95%: pattern3
        for (const auto vtx : candidate_moves) {
            if (board_.MatchPattern3(vtx, color) &&
                    !board_.IsSelfAtariMove(vtx, color)) {
                PlayMoveFast(vtx, color);
                return;
            }
        }
    }
    if (Random<kXoroShiro128Plus>::Get().Roulette<10000>(0.90f)) {
        // ~90%: atari
        for (const auto vtx : candidate_moves) {
            if (board_.IsAtariMove(vtx, color) &&
                    !board_.IsSelfAtariMove(vtx, color)) {
                PlayMoveFast(vtx, color);
                return;
            }
        }
    }
    if (Random<kXoroShiro128Plus>::Get().Roulette<10000>(0.90f)) {
        // ~90%: escape
        for (const auto vtx : candidate_moves) {
            if (board_.IsEscapeMove(vtx, color) &&
                    !board_.IsSelfAtariMove(vtx, color)) {
                PlayMoveFast(vtx, color);
                return;
            }
        }
    }

    // gather legal moves
    const int empty_cnt = board_.GetEmptyCount();
    auto legal_moves = std::vector<int>{};

    for (int i = 0; i < empty_cnt; ++i) {
        const auto vtx = board_.GetEmpty(i);

        if (IsLegalMove(vtx, color) &&
                !(board_.IsSimpleEye(vtx, color) &&
                     !board_.IsCaptureMove(vtx, color)&&
                     !board_.IsEscapeMove(vtx, color))) {
            legal_moves.emplace_back(vtx);
        }
    }

    //there is no legal moves
    if (legal_moves.empty()) {
        PlayMoveFast(kPass, color);
        return;
    }

    std::shuffle(std::begin(legal_moves),
                     std::end(legal_moves),
                     Random<kXoroShiro128Plus>::Get());

    // play first random move
    PlayMoveFast(legal_moves[0], color);
}

float GameState::GetGammaValue(const int vtx, const int color) const {
    if (board_.GetState(vtx) != kEmpty) {
        return 0.f;
    }

    float val = 1.f;

    std::uint64_t hash = 0ULL;
    float gamma = 0.f;

    for (int d = 2; d < kMaxPatternDist+1; ++d) {
        hash = board_.GetSurroundPatternHash(hash, vtx, color, d);

        if (GammasDict::Get().ProbePattern(hash, gamma)) {
            val *= gamma;
        }
    }

    for (int i = 0; i < Board::GetMaxFeatures(); ++i) {
        if (board_.GetFeatureWrapper(i, vtx, hash)) {
            if (GammasDict::Get().ProbeFeature(hash, gamma)) {
                val *= gamma;
            }
        }
    }

    return val;
}

std::vector<float> GameState::GetGammasPolicy(const int color) const {
    auto num_intersections = GetNumIntersections();
    auto board_size = GetBoardSize();

    auto policy = std::vector<float>(num_intersections, 0);
    auto acc = 0.f;

    for (int idx = 0; idx < num_intersections; ++idx) {
        const auto x = idx % board_size;
        const auto y = idx / board_size;
        const auto vtx = GetVertex(x,y);

        const auto gval = GetGammaValue(vtx, color);
        policy[idx] = gval;
        acc += gval;
    }

    for (int idx = 0; idx < num_intersections; ++idx) {
       policy[idx] /= acc;
    } 

    return policy;
}

std::vector<int> GameState::GetOwnershipAndRemovedDeadStrings(int playouts) const {
    auto fork_state = *this;
    fork_state.RemoveDeadStrings(playouts);
    return fork_state.GetOwnership();
}

std::vector<int> GameState::MarKDeadStrings(int playouts) const {
    auto num_intersections = GetNumIntersections();
    auto buffer = std::vector<int>(num_intersections, 0);

    static constexpr int kMaxPlayoutsCount = 32 * 16384;

    playouts = std::min(playouts, kMaxPlayoutsCount);
    bool already_removed = false;

    for (int p = 0; p < playouts; ++p) {
        int moves = 0;
        auto fork_state = *this;
        if (p%2==0) {
            fork_state.board_.SetToMove(!GetToMove());
        }
        while(true) {
            fork_state.FillRandomMove();

            if (p == 0 &&
                    moves == 0 &&
                    fork_state.GetLastMove() == kPass) {
                // The first move is pass. That means all dead strings
                // are removed.
                p = kMaxPlayoutsCount+1; // stop the playouts
                already_removed = true;
                break;
            }

            if (fork_state.GetPasses() >= 4) {
                break;
            }

            if (moves++ >= 2 * num_intersections) {
                // too many moves
                break;
            }
        }

        auto final_ownership = fork_state.GetOwnership();

        for (int idx = 0; idx < num_intersections; ++idx) {
            auto owner = final_ownership[idx];
            if (owner == kBlack) {
                buffer[idx] += 1;
            } else if (owner == kWhite) {
                buffer[idx] -= 1;
            }
        }
    }

    if (already_removed) {
        playouts = 1; // in order to resize the thes
    }

    const auto board_size = GetBoardSize();
    const auto thes = (int)(0.7 * playouts);
    auto dead = std::vector<int>{};

    for (int idx = 0; idx < num_intersections; ++idx) {
        const auto x = idx % board_size;
        const auto y = idx / board_size;
        const auto state = GetState(x, y);
        if (buffer[idx] >= thes) {
            // It means that this point belongs to black.
            if (state == kWhite) dead.emplace_back(GetVertex(x, y));
        } else if (buffer[idx] <= -thes) {
            // It means that this point belongs to white.
            if (state == kBlack) dead.emplace_back(GetVertex(x, y));
        } 
    }

    return dead;
}

void GameState::RemoveDeadStrings(int playouts) {
    auto dead = MarKDeadStrings(playouts);
    board_.RemoveMarkedStrings(dead);
}

void GameState::RemoveDeadStrings(std::vector<int> &dead_list) {
    board_.RemoveMarkedStrings(dead_list);
}

float GameState::GetFinalScore(float bonus) const {
    return board_.ComputeFinalScore(GetKomi() - bonus);
}

int GameState::GetVertex(const int x, const int y) const {
    return board_.GetVertex(x, y);
}

int GameState::GetIndex(const int x, const int y) const {
    return board_.GetIndex(x, y);
}

int GameState::GetX(const int vtx) const {
    return board_.GetX(vtx);
}

int GameState::GetY(const int vtx) const {
    return board_.GetY(vtx);
}

float GameState::GetKomi() const {
    float komi = static_cast<float>(komi_integer_) +
                     static_cast<float>(komi_half_) * 0.5f;
    if (komi_negative_) {
        komi = -komi;
    }
    return komi;
}

int GameState::GetWinner() const {
    return winner_;
}

int GameState::GetHandicap() const {
    return handicap_;
}

int GameState::GetPrisoner(const int color) const {
    return board_.GetPrisoner(color);
}

int GameState::GetMoveNumber() const {
    return move_number_;
}

int GameState::GetBoardSize() const {
    return board_.GetBoardSize();
}

int GameState::GetNumIntersections() const {
    return board_.GetNumIntersections();
}

int GameState::GetToMove() const {
    return board_.GetToMove();
}

int GameState::GetLastMove() const {
    return board_.GetLastMove();
}

int GameState::GetKoMove() const {
    return board_.GetKoMove();
}

int GameState::GetPasses() const {
    return board_.GetPasses();
}

std::uint64_t GameState::GetKoHash() const {
    return board_.GetKoHash();
}

std::uint64_t GameState::GetHash() const {
    return board_.GetHash() ^ komi_hash_;
}

std::uint64_t GameState::GetMoveHash(const int vtx, const int color) const {
    return board_.GetMoveHash(vtx, color);
}

int GameState::GetState(const int vtx) const {
    return board_.GetState(vtx);
}

int GameState::GetState(const int x, const int y) const {
    return board_.GetState(x, y);
}

int GameState::GetLiberties(const int vtx) const {
    return board_.GetLiberties(vtx);
}

std::vector<int> GameState::GetAppendMoves(int color) const {
    auto move_list = std::vector<int>{};
    for (const auto &m : append_moves_) {
        if (m.second == color) move_list.emplace_back(m.first);
    }
    return move_list;
}

std::shared_ptr<const Board> GameState::GetPastBoard(unsigned int p) const {
    assert(p <= (unsigned)move_number_);
    return game_history_[(unsigned)move_number_ - p];
}

const std::vector<std::shared_ptr<const Board>>& GameState::GetHistory() const {
    return game_history_;
}

std::vector<int> GameState::GetStringList(const int vtx) const {
    return board_.GetStringList(vtx);
}

std::vector<bool> GameState::GetStrictSafeArea() const {
    auto result = std::vector<bool>(GetNumIntersections(), false);
    board_.ComputeSafeArea(result, false);
    return result;
}

int GameState::GetFirstPassColor() const {
    for (auto &board : game_history_) {
        if (board->GetLastMove() == kPass) {
            return !(board->GetToMove());
        }
    }
    return kInvalid;
}

std::uint64_t GameState::ComputeSymmetryHash(const int symm) const {
    return board_.ComputeSymmetryHash(board_.GetKoMove(), symm) ^ komi_hash_;
}

std::uint64_t GameState::ComputeSymmetryKoHash(const int symm) const {
    return board_.ComputeKoHash(symm);
}
