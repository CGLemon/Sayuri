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
            out << std::setw(5) << GetStone(vtx);
        }
        out << std::endl;
    }
    out << "Remaning: " << GetStone(kNumVertices) << std::endl << std::endl;


    return out.str();
}


int String::GetNext(int vtx) const {
    return next_[vtx];
}

int String::GetParent(int vtx) const {
    return parent_[vtx];
}

int String::GetLiberty(int vtx) const {
    return liberties_[vtx];
}
int String::GetStone(int vtx) const {
    return stones_[vtx];
}
