#pragma once

#include <string>

#include "game/types.h"

class Symmetry {
public:
    static constexpr int kNumSymmetris = 8;

    static constexpr int kIdentitySymmetry = 0;

    static Symmetry& Get();

    void Initialize();

    int TransformIndex(int boardsize, int symmetry, int idx) const;
    int TransformVertex(int boardsize, int symmetry, int vtx) const;
    std::string GetDebugString(int boardsize) const;

private:
    static constexpr int kTableSize = kBoardSize + 1; // Allocate the big enough tables.

    void PartInitialize(int boardsize);

    std::pair<int, int> GetSymmetry(const int x, const int y,
                                    const int symmetry, const int boardsize) const;

    int symmetry_nn_vtx_tables_[kTableSize][kNumSymmetris][kNumVertices];
    int symmetry_nn_idx_tables_[kTableSize][kNumSymmetris][kNumIntersections];

    bool initialized_{false};
};

inline int Symmetry::TransformIndex(int boardsize, int symmetry, int idx) const {
    return symmetry_nn_idx_tables_[boardsize][symmetry][idx];
}

inline int Symmetry::TransformVertex(int boardsize, int symmetry, int vtx) const {
    return symmetry_nn_vtx_tables_[boardsize][symmetry][vtx];
}
