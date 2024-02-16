#include <algorithm>
#include <vector>
#include <cassert>

#include "utils/random.h"
#include "game/zobrist.h"

constexpr Zobrist::KEY Zobrist::kInitSeed;
constexpr Zobrist::KEY Zobrist::kEmpty;
constexpr Zobrist::KEY Zobrist::kBlackToMove;
constexpr Zobrist::KEY Zobrist::kHalfKomi;
constexpr Zobrist::KEY Zobrist::kNegativeKomi;

std::array<std::array<Zobrist::KEY, Zobrist::kZobristSize>, 4> Zobrist::kState;
std::array<std::array<Zobrist::KEY, Zobrist::kZobristSize * 2>, 2> Zobrist::kPrisoner;

std::array<Zobrist::KEY, Zobrist::kZobristSize> Zobrist::kKoMove;
std::array<Zobrist::KEY, 5> Zobrist::KPass;
std::array<Zobrist::KEY, 2> Zobrist::KScoringRule;
std::array<Zobrist::KEY, Zobrist::kZobristSize> Zobrist::kKomi;

template<typename T>
bool Collision(std::vector<T> &array) {
    std::sort(std::begin(array), std::end(array));
    auto ite = std::unique(std::begin(array), std::end(array));

    return ite != std::end(array);
}

void Zobrist::Initialize() {
    Random<kXoroShiro128Plus> rng(kInitSeed);

    while (true) {
        auto buf = std::vector<KEY>{};

        buf.emplace_back(kEmpty);
        buf.emplace_back(kBlackToMove);
        buf.emplace_back(kHalfKomi);
        buf.emplace_back(kNegativeKomi);

        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < kZobristSize; ++j) {
                buf.emplace_back(kState[i][j] = rng.Generate());
            }
        }

        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < kZobristSize * 2; ++j) {
                buf.emplace_back(kPrisoner[i][j] = rng.Generate());
            }
        }

        for (int i = 0; i < kZobristSize; ++i) {
            buf.emplace_back(kKoMove[i] = rng.Generate());
        }

        for (int i = 0; i < 5; ++i) {
            buf.emplace_back(KPass[i] = rng.Generate());
        }

        for (int i = 0; i < 2; ++i) {
            buf.emplace_back(KScoringRule[i] = rng.Generate());
        }

        for (int i = 0; i < kZobristSize; ++i) {
            buf.emplace_back(kKomi[i] = rng.Generate());
        }

        if (!Collision(buf)) {
            break;
        }
    }
}
