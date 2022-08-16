#include "game/iterator.h"

#include <algorithm>
#include <iostream>

GameStateIterator::GameStateIterator(GameState &state) {
    curr_state_ = state;
    Reset();
}

void GameStateIterator::Reset() {
    move_history_.clear();
    curr_idx_ = 0;

    const int move_num = curr_state_.GetMoveNumber();
    for (int i = 0; i < move_num; ++i) {
        ColorVertex cv;

        cv.to_move = !curr_state_.GetToMove();
        cv.vertex = curr_state_.GetLastMove();

        curr_state_.UndoMove();
        move_history_.emplace_back(cv);
    }


    if (!move_history_.empty()) {
        std::reverse(std::begin(move_history_), std::end(move_history_));

        int first_to_move = move_history_[0].to_move;
        curr_state_.SetToMove(first_to_move);
    } else {
        curr_state_.SetToMove(kBlack);
    }
}

bool GameStateIterator::Next() {
    if (curr_idx_+1 >= (int)move_history_.size()) {
        return false;
    }

    auto cv = move_history_[curr_idx_++];
    curr_state_.PlayMove(cv.vertex, cv.to_move);

    return true;
}

GameState &GameStateIterator::GetState() {
    return curr_state_;
}

int GameStateIterator::GetToMove() const {
    return move_history_[curr_idx_].to_move;
}

int GameStateIterator::GetVertex() const {
    return move_history_[curr_idx_].vertex;
}

int GameStateIterator::GetNextVertex() const {
    if (curr_idx_+1 < (int)move_history_.size()) {
        return move_history_[curr_idx_+1].vertex;
    }
    return kPass;
}

int GameStateIterator::MaxMoveNumber() const {
    return move_history_.size();
}

void GameStateIterator::RemoveUnusedDoublePass() {
    auto copy_move_history_ = std::vector<ColorVertex>{};

    for (int i = 0; i < (int)move_history_.size(); ++i) {
        bool skip = false;
        const auto cv = move_history_[i];

        if (cv.vertex == kPass) {
            if ((i+1) < (int)move_history_.size()-1 &&
                     move_history_[i+1].vertex == kPass) {
                skip = true;
            }
        }

        if (!skip) {
            copy_move_history_.emplace_back(cv);
        } else {
            i++; // also skip next move
        }
    }
    move_history_ = copy_move_history_;
}
