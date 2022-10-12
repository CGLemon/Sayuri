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

// constexpr int OWNER_MAX = 11;
// constexpr double OWNER_FACTOR = 3.f;
// constexpr double OWNER_BIAS = 1.8f;
// 
// void gen_uct_owner() {
//     double uct_owner[OWNER_MAX];
//     int center = OWNER_MAX/2;
//     for (int i = 0; i < OWNER_MAX; i++) {
//         uct_owner[i] = std::exp(-std::pow(float(i - center)/OWNER_FACTOR, 2) / OWNER_BIAS);
//     }
// }
static const std::vector<float> kUctOwner = {
    0,
    0.158755,
    0.360059,
    0.567514,
    0.726444,
    0.786306,
    0.726444,
    0.567514,
    0.360059,
    0.158755,
    0 
};

inline int ComputeWidth(int visits) {
    const int size = kProgressiveWidening.size();
    int c;
    for (c = 0; c < size; ++c) {
        if (kProgressiveWidening[c] > std::uint64_t(visits)) {
            return c;
        }
    }
    return c;
}

// The val range is [-1 ~ 1] 
inline int ComputeOwnerPriority(float val) {
    const int size = kUctOwner.size();
    val = (val+1) / 2;

    int i = 0;
    for (; i < size; ++i) {
        if ((i+1) * (1.f / size)  >= val) break;
    }
    if (i >= size) {
        i = size - 1;
    }
    return kUctOwner[i];
}
