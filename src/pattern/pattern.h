#pragma once

#include <unordered_map>
#include <set>
#include <sstream>
#include <vector>

enum FeaturnType : std::uint32_t {
    kNoFeature,
    kSpatial3x3,
    kLiberties,
    kAtari
};

struct LocPattern {
    LocPattern() = default;
    LocPattern(std::uint32_t f, std::uint32_t v) : featurn(f), value(v) {}

    std::uint32_t featurn{kNoFeature};

    std::uint32_t value;

    static inline std::uint64_t Bind(std::uint32_t f, std::uint32_t v) {
        return (std::uint64_t) v | (std::uint64_t) f << 32;
    }

    inline std::uint64_t operator()() {
        return Bind(featurn, value);
    }

    static LocPattern FromHash(std::uint64_t hash);

    static LocPattern GetSpatial3x3(std::uint32_t v);
};
