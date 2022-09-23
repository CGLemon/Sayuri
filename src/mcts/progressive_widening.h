#pragma once

#include <vector>
#include <cstdint>

static const std::vector<std::uint64_t> kProgressiveWidening = {
    0,
    40,
    112,
    241,
    474,
    893,
    1648,
    3008,
    5456,
    9863,
    17797,
    32078,
    57785,
    104058,
    187349,
    337274,
    607139,
    1092897,
    1967261,
    3541117,
    6374058,
    11473352,
    20652082,
    37173796,
    66912881,
    120443234,
    216797870,
    390236216,
    702425239,
    1264365481
};

inline int ComputeWidth(int visits) {
    int size = kProgressiveWidening.size();
    int c;
    for (c = 0; c < size; ++c) {
        if (kProgressiveWidening[c] > std::uint64_t(visits)) {
            return c;
        }
    }
    return c;
}
