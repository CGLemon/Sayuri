#include "pattern/pattern.h"

Pattern Pattern::GetSpatial3x3(std::uint32_t v) {
    return Pattern(kSpatial3x3, v);
}

Pattern Pattern::GetLiberties(std::uint32_t v) {
    return Pattern(kLiberties, v);
}

Pattern Pattern::GetAtari(std::uint32_t v) {
    return Pattern(kAtari, v);
}

Pattern Pattern::FromHash(std::uint64_t hash) {
    std::uint32_t f = (std::uint32_t)(hash >> 32);
    std::uint32_t v = (std::uint32_t)hash;

    return Pattern(f, v);
}
