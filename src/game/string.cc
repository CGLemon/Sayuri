#include <sstream>
#include <iomanip>

#include "game/string.h"

void String::Reset() {
    for (int vtx = 0; vtx < kNumVertices + 1; ++vtx) {
        parent_[vtx] = kNumVertices;
        next_[vtx] = kNumVertices;
        stones_[vtx] = 0;
        liberties_[vtx] = 0;
    }
    liberties_[kNumVertices] = KLibertiesReset;
}

void String::AddStone(const int vtx, const int lib) {
    next_[vtx] = vtx;
    parent_[vtx] = vtx;
    liberties_[vtx] = lib;
    stones_[vtx] = 1;
}

std::string String::DebugString(int boardsize) const {
    auto out = std::ostringstream{};

    out << "Next string:" << std::endl;
    for (int y = 0; y < boardsize; ++y) {
        for (int x = 0; x < boardsize; ++x) {
            const auto vtx = x + 1 + (boardsize + 2) * (y + 1);
            out << std::setw(5) << GetNext(vtx);
        }
        out << std::endl;
    }
    out << "Remaning: " << GetNext(kNumVertices) << std::endl << std::endl;

    out << "Parent string:" << std::endl;
    for (int y = 0; y < boardsize; y++) {
        for (int x = 0; x < boardsize; x++) {
            const int vtx = x + 1 + (boardsize + 2) * (y + 1);
            out << std::setw(5) << GetParent(vtx);
        }
        out << std::endl;
    }
    out << "Remaning: " << GetParent(kNumVertices) << std::endl << std::endl;

    out << "Liberties string:" << std::endl;
    for (int y = 0; y < boardsize; y++) {
        for (int x = 0; x < boardsize; x++) {
            const int vtx = x + 1 + (boardsize + 2) * (y + 1);
            out << std::setw(5) << GetLiberty(vtx);
        }
        out << std::endl;
    }
    out << "Remaning: " << GetLiberty(kNumVertices) << std::endl << std::endl;

    out << "Stones string:" << std::endl;
    for (int y = 0; y < boardsize; y++) {
        for (int x = 0; x < boardsize; x++) {
            const int vtx = x + 1 + (boardsize + 2) * (y + 1);
            out << std::setw(5) << GetStones(vtx);
        }
        out << std::endl;
    }
    out << "Remaning: " << GetStones(kNumVertices) << std::endl << std::endl;


    return out.str();
}
