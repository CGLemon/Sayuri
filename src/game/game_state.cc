#include "game/game_state.h"
#include "game/types.h"
#include "utils/splitter.h"
#include "utils/log.h"
#include "utils/random.h"
#include "utils/komi.h"
#include "utils/logits.h"
#include "pattern/pattern.h"
#include "pattern/gammas_dict.h"

#include <random>
#include <cmath>

void GameState::Reset(const int boardsize,
                      const float komi,
                      const int scoring) {
    board_.Reset(boardsize);
    SetKomi(komi);
    SetRule(scoring);
    ko_hash_history_.clear();
    game_history_.clear();
    append_moves_.clear();
    comments_.clear();

    ko_hash_history_.emplace_back(GetKoHash());
    game_history_.emplace_back(std::make_shared<Board>(board_));
    comments_.emplace_back(std::string{});

    winner_ = kUndecide;
    handicap_ = 0;
    move_number_ = 0;
    last_comment_.clear();
    territory_helper_ = std::vector<int>(GetNumIntersections(), kEmpty);
}

void GameState::SetBoardSize(const int boardsize) {
    int scoring = GetScoringRule();
    float komi = GetKomi();

    Reset(boardsize, komi, scoring);
}

void GameState::ClearBoard() {
    int scoring = GetScoringRule();
    int boardsize = GetBoardSize();
    float komi = GetKomi();

    Reset(boardsize, komi, scoring);
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

        last_comment_.clear();
        move_number_ = 0;

        ko_hash_history_.clear();
        game_history_.clear();
        comments_.clear();

        ko_hash_history_.emplace_back(GetKoHash());
        game_history_.emplace_back(std::make_shared<Board>(board_));
        append_moves_.emplace_back(vtx, color);
        comments_.emplace_back(std::string{});

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
        comments_.resize(move_number_);

        ko_hash_history_.emplace_back(GetKoHash());
        game_history_.emplace_back(std::make_shared<Board>(board_));
        PushComment();

        return true;
    }
    return false;
}

bool GameState::UndoMove() {
    if (move_number_ >= 1) {
        // Cut off unused history.
        ko_hash_history_.resize(move_number_);
        game_history_.resize(move_number_);
        comments_.resize(move_number_);

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
    const auto y = GetBoardSize() - GetY(vtx) - 1;

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

    auto spt = Splitter(input);

    if (spt.GetCount() == 2) {
        const auto color_str = spt.GetWord(0)->Get<>();
        const auto vtx_str = spt.GetWord(1)->Get<>();
        color = TextToColor(color_str);
        vertex = TextToVertex(vtx_str);
    } else if (spt.GetCount() == 1) {
        const auto vtx_str = spt.GetWord(0)->Get<>();
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
    LOGGING << VertexToText(vtx) << ' '
                << board_.GetMoveTypesString(vtx, color)
                << std::endl;
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
    out << "Handicap: " << GetHandicap() << ", ";
    out << "Rule: " << GetRuleString();

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

void GameState::SetRule(const int scoring) {
    if (scoring == kArea || scoring == kTerritory) {
        scoring_rule_ = static_cast<ScoringRuleType>(scoring);
        scoring_hash_ = Zobrist::KScoringRule[scoring];
    } else {
        LOGGING << "Only accept for Chinese or Japanese rules." << std::endl;
    }
}

void GameState::SetTerritoryHelper(const std::vector<int> &ownership) {
     territory_helper_ = ownership;
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

    auto movelist_vertex = std::vector<int>{};

    if (handicap >= 2) {
        movelist_vertex.emplace_back(GetVertex(low, low));
        movelist_vertex.emplace_back(GetVertex(high, high));
    }

    if (handicap >= 3) {
        movelist_vertex.emplace_back(GetVertex(high, low));
    }

    if (handicap >= 4) {
        movelist_vertex.emplace_back(GetVertex(low, high));
    }

    if (handicap >= 5 && handicap % 2 == 1) {
        movelist_vertex.emplace_back(GetVertex(mid, mid));
    }

    if (handicap >= 6) {
        movelist_vertex.emplace_back(GetVertex(low, mid));
        movelist_vertex.emplace_back(GetVertex(high, mid));
    }

    if (handicap >= 8) {
        movelist_vertex.emplace_back(GetVertex(mid, low));
        movelist_vertex.emplace_back(GetVertex(mid, high));
    }

    PlayHandicapStones(movelist_vertex, true);

    return true;
}

bool GameState::SetFreeHandicap(std::vector<std::string> movelist) {
    auto movelist_vertex = std::vector<int>(movelist.size());
    std::transform(std::begin(movelist), std::end(movelist), std::begin(movelist_vertex),
                       [this](auto text){
                           return TextToVertex(text);
                       }
                   );
    return PlayHandicapStones(movelist_vertex, true);
}

std::vector<int> GameState::PlaceFreeHandicap(int handicap) {
    auto stones_list = std::vector<int>{};
    auto num_intersections = GetNumIntersections();

    if (SetFixdHandicap(handicap)) {
        for (int idx = 0; idx < num_intersections; ++idx) {
            const auto vtx = IndexToVertex(idx);
            if (GetState(vtx) == kBlack) {
                stones_list.emplace_back(vtx);
            }
        }
    }

    return stones_list;
}

bool GameState::PlayHandicapStones(std::vector<int> movelist_vertex,
                                   bool kata_like_handicap_style) {
    auto fork_state = *this;
    fork_state.ClearBoard();

    const int size = movelist_vertex.size();

    for (int i = 0; i < size; ++i) {
        const auto vtx = movelist_vertex[i];
        if (fork_state.IsLegalMove(vtx, kBlack)) {
            if (i == size-1 && kata_like_handicap_style) {
                // The last handicap move is not appending move
                // in the KataGo's SGF.
                fork_state.PlayMove(vtx, kBlack);
            } else {
                fork_state.AppendMove(vtx, kBlack);
            }
        } else {
            return false;
        }
    }

    // Legal status. Copy the status.
    *this = fork_state;

    SetHandicap(movelist_vertex.size());
    SetToMove(kWhite);

    return true;
}

std::vector<int> GameState::GetOwnership() const {
    auto res = std::vector<int>(GetNumIntersections(), kInvalid);

    board_.ComputeScoreArea(
        res, scoring_rule_, territory_helper_);

    return res;
}

std::vector<int> GameState::GetRawOwnership() const {
    auto res = std::vector<int>(GetNumIntersections(), kInvalid);

    board_.ComputeReachArea(res);

    return res;
}

void GameState::PlayRandomMove() {
    auto candidate_moves = std::vector<int>{};
    const int color = GetToMove();

    board_.GenerateCandidateMoves(candidate_moves, color);
    std::shuffle(std::begin(candidate_moves),
                     std::end(candidate_moves),
                     Random<>::Get());

    if (Random<>::Get().Roulette<10000>(0.90f)) {
        // ~90%: capture
        for (const auto vtx : candidate_moves) {
            if (board_.IsCaptureMove(vtx, color)) {
                PlayMoveFast(vtx, color);
                return;
            }
        }
    }
    if (Random<>::Get().Roulette<10000>(0.95f)) {
        // ~95%: pattern3
        for (const auto vtx : candidate_moves) {
            if (board_.MatchPattern3(vtx) &&
                    !board_.IsSelfAtariMove(vtx, color)) {
                PlayMoveFast(vtx, color);
                return;
            }
        }
    }
    if (Random<>::Get().Roulette<10000>(0.90f)) {
        // ~90%: atari
        for (const auto vtx : candidate_moves) {
            if (board_.IsAtariMove(vtx, color) &&
                    !board_.IsSelfAtariMove(vtx, color)) {
                PlayMoveFast(vtx, color);
                return;
            }
        }
    }
    if (Random<>::Get().Roulette<10000>(0.90f)) {
        // ~90%: escape
        for (const auto vtx : candidate_moves) {
            if (board_.IsEscapeMove(vtx, color) &&
                    !board_.IsSelfAtariMove(vtx, color)) {
                PlayMoveFast(vtx, color);
                return;
            }
        }
    }

    // gather the legal moves
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

    // there is no legal move
    if (legal_moves.empty()) {
        PlayMoveFast(kPass, color);
        return;
    }

    int selected = Random<>::Get().Generate() % legal_moves.size();
    PlayMoveFast(legal_moves[selected], color);
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
        if (board_.GetFeatureWrapper(i, vtx, color, hash)) {
            if (GammasDict::Get().ProbeFeature(hash, gamma)) {
                val *= gamma;
            }
        }
    }

    return val;
}

std::vector<float> GameState::GetGammasPolicy(const int color) const {
    auto num_intersections = GetNumIntersections();
    auto policy = std::vector<float>(num_intersections, 0);

    for (int idx = 0; idx < num_intersections; ++idx) {
        const auto vtx = IndexToVertex(idx);
        const auto gval = GetGammaValue(vtx, color);
        policy[idx] = std::log(gval);
    }

    return Softmax(policy, 1.f);
}

void GameState::RemoveDeadStrings(std::vector<int> &dead_list) {
    board_.RemoveMarkedStrings(dead_list);
}

float GameState::GetFinalScore(const int color) const {
    float komi_with_handicap = GetKomi() + handicap_;
    float black_score = static_cast<float>(
       board_.ComputeScoreOnBoard(kBlack, scoring_rule_, territory_helper_)) - komi_with_handicap;
    return color == kBlack ? black_score : -black_score;
}

float GameState::GetFinalScore(const int color,
                               const std::vector<int> &territory_helper) {
    SetTerritoryHelper(territory_helper);
    return GetFinalScore(color);
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

int GameState::IndexToVertex(int idx) const {
    return board_.IndexToVertex(idx);
}

int GameState::VertexToIndex(int vtx) const {
    return board_.VertexToIndex(vtx);
}

float GameState::GetKomiWithPenalty() const {
    return GetKomi() + GetPenalty();
}

float GameState::GetPenalty() const {
    float penalty = 0.f;

    if (scoring_rule_ == kTerritory) {
        penalty += board_.GetPrisoner(kWhite);
        penalty -= board_.GetPrisoner(kBlack);
    }
    if (scoring_rule_ == kArea) {
        penalty += handicap_;
    }
    return penalty;
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

int GameState::GetNumVertices() const {
    return board_.GetNumVertices();
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
    return board_.GetHash() ^ komi_hash_ ^ scoring_hash_;
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

int GameState::GetScoringRule() const {
    return scoring_rule_;
}

std::vector<bool> GameState::GetStrictSafeArea() const {
    auto result = std::vector<bool>(GetNumIntersections(), false);
    board_.ComputeSafeArea(result, false);
    return result;
}

std::uint64_t GameState::ComputeSymmetryHash(const int symm) const {
    return board_.ComputeSymmetryHash(board_.GetKoMove(), symm) ^ komi_hash_ ^ scoring_hash_;
}

std::uint64_t GameState::ComputeSymmetryKoHash(const int symm) const {
    return board_.ComputeKoHash(symm);
}

void GameState::SetComment(std::string c) {
    last_comment_ = c;
}

void GameState::RewriteComment(std::string c, size_t i) {
    if (i < comments_.size()) {
        comments_[i] = c;
    }
}

std::string GameState::GetComment(size_t i) const {
    if (i >= comments_.size()) {
        return std::string{};
    }
    return comments_[i];
}

void GameState::PushComment() {
    comments_.emplace_back(last_comment_);
    last_comment_.clear();
}

float GameState::GetWave() const {
    if (scoring_rule_ == kTerritory) {
        return 0.f;
    }

    float curr_komi = GetKomiWithPenalty();
    if (GetToMove() == kWhite) {
        curr_komi = 0.f - curr_komi;
    }

    bool is_board_area_even = (GetNumIntersections()) % 2 == 0;

    // Find the difference between the komi viewed from our perspective and
    // the nearest drawable komi below it.
    float komi_floor;
    if (is_board_area_even) {
        komi_floor = std::floor(curr_komi / 2.0f) * 2.0f;
    } else {
        komi_floor = std::floor((curr_komi-1.0f) / 2.0f) * 2.0f + 1.0f;
    }

    // Cap just in case we have floating point weirdness.
    float delta = curr_komi - komi_floor;
    delta = std::max(delta, 0.f);
    delta = std::min(delta, 2.f);

    // Create the triangle wave based on the difference.
    float wave;
    if (delta < 0.5f) {
        wave = delta;
    } else if (delta < 1.5f) {
        wave = 1.f - delta;
    } else {
        wave = delta - 2.f;
    }
    return wave;
}

std::string GameState::GetRuleString() const {
    if (scoring_rule_ == kArea) {
        return "chinese";
    }
    if (scoring_rule_ == kTerritory) {
        return "japanese";
    }
    return "unknown";
}

bool GameState::IsNeighborColor(const int vtx, const int color) const {
    return board_.IsNeighborColor(vtx, color);
}
