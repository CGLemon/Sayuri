#pragma once

#include <unordered_map>
#include <set>
#include <sstream>
#include <iostream>
#include <vector>

#include "game/game_state.h"


enum FeaturnType : std::uint32_t {
    kNoFeature,
    kSpatial3x3
};

struct Pattern {
    Pattern() = default;
    Pattern(std::uint32_t f, std::uint32_t v) : featurn(f), value(v) {}

    std::uint32_t featurn{kNoFeature};

    std::uint32_t value;

    static inline std::uint64_t Bind(std::uint32_t f, std::uint32_t v) {
        return (std::uint64_t) v | (std::uint64_t) f << 32;
    }

    inline std::uint64_t operator()() {
        return Bind(featurn, value);
    }

    static Pattern FromHash(std::uint64_t hash);

    static Pattern GetSpatial3x3(std::uint32_t v);
};

