#include <cassert>
#include <sstream>
#include <iomanip>

#include "game/symmetry.h"

constexpr int Symmetry::kNumSymmetris;
constexpr int Symmetry::kIdentitySymmetry;

Symmetry& Symmetry::Get() {
    static Symmetry symmetry;
    return symmetry;
}

void Symmetry::Initialize() {
    if (initialized_) return;

    for (int bsize = kMinGTPBoardSize; bsize <= kBoardSize; ++bsize) {
        PartInitialize(bsize);
    }
    initialized_ = true;
}

void Symmetry::PartInitialize(int boardsize) {
    assert(boardsize >= kMinGTPBoardSize);
    assert(boardsize <= kBoardSize);

    const auto GetVertex = [boardsize](int x, int y) -> int {
        return (y + 1) * (boardsize+2) + (x + 1);
    };

    const auto GetIndex = [boardsize](int x, int y) -> int {
        return y * boardsize + x;
    };

    int bsize = boardsize;

    for (int symm = 0; symm < kNumSymmetris; ++symm) {
        for (int vtx = 0; vtx < kNumVertices; ++vtx) {
            symmetry_nn_vtx_tables_[bsize][symm][vtx] = 0;
        }
        for (int idx = 0; idx < kNumIntersections; ++idx) {
            symmetry_nn_idx_tables_[bsize][symm][idx] = 0;
        }
    }

    for (int symm = 0; symm < kNumSymmetris; ++symm) {
        for (int y = 0; y < bsize; y++) {
            for (int x = 0; x < bsize; x++) {
                const auto symm_idx = GetSymmetry(x, y, symm, bsize);
                const auto vtx = GetVertex(x, y);
                const auto idx = GetIndex(x, y);

                symmetry_nn_idx_tables_[bsize][symm][idx] =
                    GetIndex(symm_idx.first, symm_idx.second);

                symmetry_nn_vtx_tables_[bsize][symm][vtx] =
                    GetVertex(symm_idx.first, symm_idx.second);
            }
        }
    }
}

std::string Symmetry::GetDebugString(int boardsize) const {
    const auto GetVertex = [boardsize](int x, int y) -> int {
        return (y + 1) * (boardsize+2) + (x + 1);
    };

    const auto GetIndex = [boardsize](int x, int y) -> int {
        return y * boardsize + x;
    };

    auto out = std::ostringstream{};
    for (int symm = 0; symm < kNumSymmetris; ++symm) {
        out << "Vertex Symmetry " << symm+1 << ':' << std::endl;
        for (int y = 0; y < boardsize; y++) {
            for (int x = 0; x < boardsize; x++) {
                const auto vtx = GetVertex(x, y);
                out << std::setw(4) << TransformVertex(boardsize, symm, vtx);
            }
            out << std::endl;
        }
        out << std::endl;
    }

    for (int symm = 0; symm < kNumSymmetris; ++symm) {
        out << "Index Symmetry " << symm+1 << ':' << std::endl;
        for (int y = 0; y < boardsize; y++) {
            for (int x = 0; x < boardsize; x++) {
                const auto idx = GetIndex(x, y);
                out << std::setw(4) << TransformIndex(boardsize, symm, idx);
            }
            out << std::endl;
        }
        out << std::endl;
    }
    return out.str();
}

std::pair<int, int> Symmetry::GetSymmetry(const int x, const int y,
                                          const int symmetry,
                                          const int boardsize)  const {
    assert(x >= 0 && x < boardsize);
    assert(y >= 0 && y < boardsize);
    assert(symmetry >= 0 && symmetry < kNumSymmetris);

    int idx_x = x;
    int idx_y = y;

    if ((symmetry & 4) != 0) {
        std::swap(idx_x, idx_y);
    }

    if ((symmetry & 2) != 0) {
        idx_x = boardsize - idx_x - 1;
    }

    if ((symmetry & 1) != 0) {
        idx_y = boardsize - idx_y - 1;
    }

    assert(idx_x >= 0 && idx_x < boardsize);
    assert(idx_y >= 0 && idx_y < boardsize);
    assert(symmetry != kIdentitySymmetry || (x == idx_x && y == idx_y));

    return {idx_x, idx_y};
}
