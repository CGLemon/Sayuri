#pragma once
#include <array>
#include <string>

#include "game/types.h"

struct Strings {
    // The next stone in string.
    std::array<std::uint16_t, kNumVertices+1> next_;

    // The parent node of string.
    std::array<std::uint16_t, kNumVertices+1> parent_;

    // The liberties per string parent.
    std::array<std::uint16_t, kNumVertices+1> liberties_;

    // The stones per string parent.
    std::array<std::uint16_t, kNumVertices+1> stones_;

    int GetNext(int vtx) const;
    int GetParent(int vtx) const;
    int GetLiberty(int vtx) const;
    int GetStones(int vtx) const;

    // Reset all strings status.
    void Reset();

    // Place a stone on the board
    void AddStone(const int vtx, const int lib);

    std::string GetDebugString(const int boardsize) const;
};

inline int Strings::GetNext(int vtx) const {
    return next_[vtx];
}

inline int Strings::GetParent(int vtx) const {
    return parent_[vtx];
}

inline int Strings::GetLiberty(int vtx) const {
    return liberties_[vtx];
}

inline int Strings::GetStones(int vtx) const {
    return stones_[vtx];
}
