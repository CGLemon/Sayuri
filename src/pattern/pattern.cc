#include "pattern/pattern.h"

LocPattern LocPattern::GetSpatial3x3(std::uint32_t v) {
    return LocPattern(kSpatial3x3, v);
}

LocPattern LocPattern::FromHash(std::uint64_t hash) {
    std::uint32_t f = (std::uint32_t)(hash >> 32);
    std::uint32_t v = (std::uint32_t)hash;

    return LocPattern(f, v);
}
