#include "pattern/pattern.h"

LocPattern LocPattern::GetSpatial3x3(std::uint32_t v) {
    return LocPattern(kSpatial3x3, v);
}

LocPattern LocPattern::GetLiberties(std::uint32_t v) {
    return LocPattern(kLiberties, v);
}

LocPattern LocPattern::GetDistToBorder(std::uint32_t v) {
    return LocPattern(kDistToBorder, v);
}

LocPattern LocPattern::GetDistToLastMove(std::uint32_t v) {
    return LocPattern(kDistToLastMove, v);
}

LocPattern LocPattern::GetAtari(std::uint32_t v) {
    return LocPattern(kAtari, v);
}

LocPattern LocPattern::FromHash(std::uint64_t hash) {
    std::uint32_t f = (std::uint32_t)(hash >> 32);
    std::uint32_t v = (std::uint32_t)hash;

    return LocPattern(f, v);
}
