#include "utils/komi.h"
#include <cmath>
#include <iostream>

bool EqualToZero(float v) {
    return std::abs(v) < 1e-4f;
}

template <> 
float AdjustKomi<float>(float komi) {
    if (EqualToZero(komi)) {
        return 0;
    }

    bool negtive = komi < 0.0f;
    if (negtive) {
        komi = -komi;
    }

    int integer_part = int(komi);
    float float_part = komi - integer_part;

    if (float_part < 0.25f) {
        float_part = 0.f;
    } else if (float_part < 0.75f) {
        float_part = 0.5f;
    } else {
        float_part = 1.f;
    }

    komi = float_part + integer_part;

    if (negtive && !EqualToZero(komi)) {
        komi = -komi;
    }
    return komi;
}

template <> 
float AdjustKomi<int>(float komi) {
    if (EqualToZero(komi)) {
        return 0;
    }

    bool negtive = komi < 0.0f;
    if (negtive) {
        komi = -komi;
    }

    int integer_part = int(komi);
    float float_part = komi - integer_part;

    if (float_part > 0.5f) {
        integer_part += 1;
    }

    if (negtive && integer_part != 0) {
        integer_part = -integer_part;
    }

    return (float)integer_part;
}
