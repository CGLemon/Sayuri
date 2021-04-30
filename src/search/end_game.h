#pragma once

#include "game/game_state.h"
#include "game/board.h"

class EndGame {
public:
    static EndGame &Get(GameState &state);

    EndGame(GameState &state) : root_state_(state) {}

    std::vector<int> GetFinalOwnership() const;

private:
    std::vector<int> RandomRollout(Board board) const;

    bool RandomMove(Board &current_board, std::vector<int> &movelist) const;

    GameState &root_state_;
};
