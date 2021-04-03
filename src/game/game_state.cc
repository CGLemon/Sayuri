#include "game/game_state.h"
#include "game/types.h"
#include "utils/parser.h"
#include "utils/log.h"

void GameState::Reset(const int boardsize, const float komi) {
    board_.Reset(boardsize);
    SetKomi(komi);
    game_history_.clear();
    game_history_.emplace_back(std::make_shared<Board>(board_));

    winner_ = kUndecide;
}

void GameState::ClearBoard() {
    Reset(GetBoardSize(), GetKomi());
}

bool GameState::PlayMove(const int vtx, const int color) {
    if (vtx == kResign) {
        if (color == kBlack) {
            winner_ = kBlackWon;
        } else {
            winner_ = kWhiteWon;
        }
        return true;
    }

    if (board_.IsLegalMove(vtx, color)) {
        board_.PlayMoveAssumeLegal(vtx, color);
        game_history_.resize(GetMoveNumber());
        game_history_.emplace_back(std::make_shared<Board>(board_));
        return true;
    }
    return false;
}

bool GameState::UndoMove() {
    const auto move_number = GetMoveNumber();
    if (move_number >= 1) {
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
    out << "Board Size: " << GetBoardSize() << ", ";
    out << "Komi: " << GetKomi() << ", ";
    out << "Handicap: " << GetHandicap() << ", ";

    out << std::hex;
    out << "Hash: " << GetHash() << ", ";
    out << "Ko Hash: " << GetKoHash();
    out << std::dec;

    out << "}" << std::endl;
    return out.str();
}


void GameState::ShowBoard() const {
    LOGGING << board_.GetBoardString(board_.GetLastMove(), false);
    LOGGING << GetStateString();
}

void GameState::SetKomi(float komi) {
    komi_integer_ = static_cast<int>(komi);
    komi_float_ = komi - static_cast<float>(komi_integer_);
    if (komi_float_ < 1e-4 && komi_float_ > (-1e-4)) {
        komi_float_ = 0.0f;
    }
}

void GameState::SetHandicap(int handicap) {
    handicap_ = handicap;
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

std::vector<int> GameState::GetSimpleOwnership() const {
    return board_.GetSimpleOwnership();
}

float GameState::GetFinalScore(float bonus) const {
    return board_.ComputeFinalScore(GetKomi() + GetHandicap() + bonus);
}

float GameState::GetKomi() const {
    return komi_float_ + static_cast<float>(komi_integer_);
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
