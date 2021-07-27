#pragma once

#include <array>

#include "game/types.h"

class Zobrist {
private:
    using KEY = std::uint64_t;
    static constexpr auto kZobristSize = kNumVertices;
    static constexpr KEY kInitSeed = 0xabcdabcd12345678;

public:
    static constexpr KEY kEmpty = 0x1234567887654321;
    static constexpr KEY kBlackToMove = 0xabcdabcdabcdabcd;

    static std::array<std::array<KEY, kZobristSize>, 4> kState;
    static std::array<KEY, kZobristSize> kKoMove;
    static std::array<std::array<KEY, kZobristSize * 2>, 2> kPrisoner;
    static std::array<KEY, 5> KPass;
    static std::array<KEY, 4096> kIdentity;

    static void Initialize();
};
