#pragma once

#include "game/game_state.h"

#include <vector>

// The history boards were generated by SGF parser may have some 
// error. The GameStateIterator promise the outputs training data
// is correct.
class GameStateIterator {
public:
    GameStateIterator(GameState &state);

    void Reset();
    bool Next();

    GameState &GetState();
    int GetToMove() const;
    int GetVertex() const;
    int GetNextVertex() const;
    int MaxMoveNumber() const;

private:
    struct ColorVertex {
        int to_move;
        int vertex;
    };

    GameState curr_state_;
    std::vector<ColorVertex> move_history_;
    int curr_idx_;
};
