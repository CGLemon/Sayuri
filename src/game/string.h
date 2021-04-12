#pragma once
#include <array>
#include <string>

#include "game/types.h"

struct String {
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
    int GetStone(int vtx) const;

    void Reset();
    void AddStone(const int vtx, const int lib);

    std::string DebugString(int boardsize) const;
};
