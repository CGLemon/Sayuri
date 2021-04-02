#ifndef GAME_ZOBRIST_H_INCLUDE
#define GAME_ZOBRIST_H_INCLUDE

#include <array>

#include "game/types.h"

class Zobrist {
private:
    using KEY = std::uint64_t;
    static constexpr auto ZOBRIST_SIZE = kNumVertices;
    static constexpr KEY kInitSeed = 0xabcdabcd12345678;

public:
    static constexpr KEY kEmpty = 0x1234567887654321;
    static constexpr KEY kBlackToMove = 0xabcdabcdabcdabcd;

    static std::array<std::array<KEY, ZOBRIST_SIZE>, 4> kState;
    static std::array<KEY, ZOBRIST_SIZE> kKoMove;
    static std::array<std::array<KEY, ZOBRIST_SIZE * 2>, 2> kPrisoner;
    static std::array<KEY, 5> KPass;

    static void Initialize();
};


#endif
