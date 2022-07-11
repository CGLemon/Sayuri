#pragma once

#include <string>
#include <vector>

struct LocPattern {
    enum FeatureType : std::uint32_t {
        kNoFeature = 0,
        kSpatial3x3 = 1,
        kLiberties = 2,
        kDistToBorder = 3,
        kDistToLastMove = 4,
        kAtari = 5
    };

    LocPattern() = default;
    LocPattern(std::uint32_t f, std::uint32_t v) : feature(f), value(v) {}

    std::uint32_t feature{kNoFeature};

    std::uint32_t value;

    static inline std::uint64_t Bind(std::uint32_t f, std::uint32_t v) {
        return (std::uint64_t) v | (std::uint64_t) f << 32;
    }

    inline std::uint64_t operator()() {
        return Bind(feature, value);
    }

    static LocPattern FromHash(std::uint64_t hash);

    static LocPattern GetNoFeature();
    static LocPattern GetSpatial3x3(std::uint32_t v);
    static LocPattern GetLiberties(std::uint32_t v);
    static LocPattern GetDistToBorder(std::uint32_t v);
    static LocPattern GetDistToLastMove(std::uint32_t v);
    static LocPattern GetAtari(std::uint32_t v);
};

const static std::vector<std::string> kFeaturesNameMap = {
    "NO", // kNoFeature
    "s3", // kSpatial3x3
    "l",  // kLiberties
    "db", // kDistToBorder
    "dm", // kDistToLastMove
    "a"   // kAtari
};
