#pragma once

#include <array>

#include "game/types.h"

class Zobrist {
private:
    using KEY = std::uint64_t;

    static constexpr auto kZobristSize = kNumVertices;
    static constexpr KEY kInitSeed = 0xabcdabcd12345678;

public:
    static constexpr KEY kEmpty        = 0x1234567887654321;
    static constexpr KEY kBlackToMove  = 0xabcdabcdabcdabcd;
    static constexpr KEY kHalfKomi     = 0x5678876556788765;
    static constexpr KEY kNegativeKomi = 0x4321432143214321;

    static std::array<std::array<KEY, kZobristSize>, 4> kState;
    static std::array<KEY, kZobristSize> kKoMove;
    static std::array<std::array<KEY, kZobristSize * 2>, 2> kPrisoner;
    static std::array<KEY, 5> KPass;
    static std::array<KEY, kZobristSize> kKomi;

    static void Initialize();
};
