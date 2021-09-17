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
std::array<Zobrist::KEY, Zobrist::kZobristSize> Zobrist::kKomi;

template<typename T>
bool Collision(std::vector<T> &array) {
    const auto s = array.size();
    if (s <= 1) {
        return false;
    }

    for (auto i = size_t{0}; i < (s-1); ++i) {
        auto begin = std::cbegin(array);
        auto element = std::next(begin, i);
        auto start = std::next(element, 1);
        auto end = std::cend(array);
        auto res = std::find(start, end, *element);
        if (res != end) {
            return true;
        }
    }
    return false;
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
                Zobrist::kState[i][j] = rng.Generate();
                buf.emplace_back(Zobrist::kState[i][j]);
            }
        }

        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < kZobristSize * 2; ++j) {
                Zobrist::kPrisoner[i][j] = rng.Generate();
                buf.emplace_back(Zobrist::kPrisoner[i][j]);
            }
        }

        for (int i = 0; i < kZobristSize; ++i) {
            Zobrist::kKoMove[i] = rng.Generate();
            buf.emplace_back(Zobrist::kKoMove[i]);
        }

        for (int i = 0; i < 5; ++i) {
            Zobrist::KPass[i] = rng.Generate();
            buf.emplace_back(Zobrist::KPass[i]);
        }

        for (int i = 0; i < kZobristSize; ++i) {
            Zobrist::kKomi[i] = rng.Generate();
            buf.emplace_back(Zobrist::kKomi[i]);
        }

        if (!Collision(buf)) {
            break;
        }
    }
}
