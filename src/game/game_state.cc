#include "game/game_state.h"
#include "game/types.h"
#include "utils/parser.h"
#include "utils/log.h"

void GameState::Reset(const int boardsize, const float komi) {
    board_.Reset(boardsize);
    SetKomi(komi);
    ko_hash_history_.clear();
    game_history_.clear();
    ko_hash_history_.emplace_back(GetKoHash());
    game_history_.emplace_back(std::make_shared<Board>(board_));

    winner_ = kUndecide;
}

void GameState::ClearBoard() {
    Reset(GetBoardSize(), GetKomi());
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

        // Cut off unused history.
        ko_hash_history_.resize(GetMoveNumber());
        game_history_.resize(GetMoveNumber());

        ko_hash_history_.emplace_back(GetKoHash());
        game_history_.emplace_back(std::make_shared<Board>(board_));

        return true;
    }
    return false;
}

bool GameState::UndoMove() {
    const auto move_number = GetMoveNumber();
    if (move_number >= 1) {
        // Cut off unused history.
        ko_hash_history_.resize(move_number);
        game_history_.resize(move_number);

        board_ = *game_history_[move_number-1];
        return true;
    }
    return false;
}

int GameState::TextToVertex(std::string text) {
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

std::string GameState::VertexToSgf(const int vtx) {
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

std::string GameState::VertexToText(const int vtx) {
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

        if (color_str == "B" || color_str == "b" || color_str == "black") {
            color = kBlack;
        } else if (color_str == "W" || color_str == "w" || color_str == "white") {
            color = kWhite;
        }
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
    out << "Move Number: " << GetMoveNumber() << ", ";
    out << "Komi: " << GetKomi() << ", ";
    out << "Board Size: " << GetBoardSize() << ", ";
    out << "Handicap: " << GetHandicap() << ", ";
    out << "Result: ";
    if (GetWinner() == kBlackWon) {
        out << "Black Won";
    } else if (GetWinner() == kWhiteWon) {
        out << "White Won";
    } else if (GetWinner() == kDraw) {
        out << "Draw";
    } else if (GetWinner() == kUndecide) {
        out << "None";
    }
    out << "}" << std::endl;
    return out.str();
}


void GameState::ShowBoard() const {
    LOGGING << board_.GetBoardString(board_.GetLastMove(), false);
    LOGGING << GetStateString();
}

void GameState::SetWinner(GameResult result) {
    winner_ = result;
}

void GameState::SetKomi(float komi) {
    komi_integer_ = static_cast<int>(komi);
    komi_float_ = komi - static_cast<float>(komi_integer_);
    if (komi_float_ < 1e-4 && komi_float_ > (-1e-4)) {
        komi_float_ = 0.0f;
    }
}

void GameState::SetColor(const int color) {
    board_.SetToMove(color);
}

void GameState::SetHandicap(int handicap) {
    handicap_ = handicap;
}

bool GameState::IsSuperko() const {
    auto first = std::crbegin(ko_hash_history_);
    auto last = std::crend(ko_hash_history_);

    auto res = std::find(++first, last, GetKoHash());

    return (res != last);
}

bool GameState::IsLegalMove(const int vertex, const int color) const {
    return board_.IsLegalMove(vertex, color);
}

bool GameState::IsLegalMove(const int vertex, const int color,
                            std::function<bool(int, int)> AvoidToMove) const {
    return board_.IsLegalMove(vertex, color, AvoidToMove);
}

bool GameState::SetFixdHandicap(int handicap) {
    if (board_.SetFixdHandicap(handicap)) {
        SetHandicap(handicap);
        return true;
    }
    return false;
}

bool GameState::SetFreeHandicap(std::vector<std::string> movelist) {
    auto movelist_vertex = std::vector<int>(movelist.size());
    std::transform(std::begin(movelist), std::end(movelist), std::begin(movelist_vertex),
                       [this](auto text){
                           return TextToVertex(text);
                       }
                   );

    if (board_.SetFreeHandicap(movelist_vertex)) {
        SetHandicap(movelist.size());
        return true;
    }
    return false;
}

float GameState::GetFinalScore(float bonus) const {
    return board_.ComputeFinalScore(GetKomi() + GetHandicap() + bonus);
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
    return komi_float_ + static_cast<float>(komi_integer_);
}

int GameState::GetWinner() const {
    if (winner_ != kUndecide) {
        return winner_;
    }
    if (GetPasses() >= 2) {
        auto score = GetFinalScore();
        if (score > (-1e-4) && score < 1e-4) {
            return kDraw;
        } else if (score > 0.f){
            return kBlack;
        } else {
            return kWhite;
        }
    }
    return kUndecide;
}

int GameState::GetHandicap() const {
    return handicap_;
}

int GameState::GetPrisoner(const int color) const {
    return board_.GetPrisoner(color);
}

int GameState::GetMoveNumber() const {
    return board_.GetMoveNumber();
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

int GameState::GetKoHash() const {
    return board_.GetKoHash();
}

int GameState::GetHash() const {
    return board_.GetHash();
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

std::shared_ptr<const Board> GameState::GetPastBoard(unsigned int p) const {
    assert(p <= (unsigned)GetMoveNumber());
    return game_history_[(unsigned)GetMoveNumber() - p];  
}

const std::vector<std::shared_ptr<const Board>>& GameState::GetHistory() const {
    return game_history_;
}
