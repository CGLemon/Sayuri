#pragma once

#include <array>
#include <string>

#include "game/types.h"

class Symmetry {
public:
    static constexpr int kNumSymmetris  = 8;

    static constexpr int kIdentitySymmetry = 0;

    static Symmetry& Get();

    void Initialize(int boardsize);

    int TransformIndex(int symmetry, int idx) const;
    int TransformVertex(int symmetry, int vtx) const;
    std::string GetDebugString() const ;

private:
    void InnerInitialize(int boardsize);

    std::pair<int, int> GetSymmetry(const int x, const int y,
                                    const int symmetry, const int boardsize) const;

    std::array<std::array<int, kNumVertices>, kNumSymmetris> symmetry_nn_vtx_table_;
    std::array<std::array<int, kNumIntersections>, kNumSymmetris> symmetry_nn_idx_table_;

    int board_size_{0};
};
