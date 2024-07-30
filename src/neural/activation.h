#pragma once

#include <string>
#include <stdexcept>
#include <cctype>
#include <cmath>

enum class Activation : int {
    kIdentity = 0,
    kReLU = 1,
    kELU = 2,
    kSELU = 3,
    kGELU = 4,
    kMISH = 5,
    kSwish = 6,
    kHardSwish = 7
};

static inline Activation StringToAct(std::string val) {
    for (char &c: val) {
        c = std::tolower(c);
    }
    if (val == "identity") {
        return Activation::kIdentity;
    } else if (val == "relu") {
        return Activation::kReLU;
    } else if (val == "elu") {
        return Activation::kELU;
    } else if (val == "selu") {
        return Activation::kSELU;
    } else if (val == "gelu") {
        return Activation::kGELU;
    } else if (val == "mish") {
        return Activation::kMISH;
    } else if (val == "swish") {
        return Activation::kSwish;
    } else if (val == "hardswish") {
        return Activation::kHardSwish;
    }
    throw std::runtime_error{"Unknown activation type."};
}

#define ACTIVATION_IDENTITY(x) \
    ;

#define ACTIVATION_RELU(x) \
    x = x > 0.f ? x : 0.f;

#define ACTIVATION_ELU(x) \
    x = x > 0.f ? x : (expf(x) - 1);

#define ACTIVATION_SELU(x) \
    x = x > 0.f ?          \
            (1.05070098f * x) : \
            (1.05070098f * 1.67326324f * (expf(x) - 1.0f));

#define ACTIVATION_GELU(x) \
    x = 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715 * x * x * x)));

#define ACTIVATION_MISH(x) \
    x = x * tanhf(logf(1.0f + expf(x)));

#define ACTIVATION_SWISH(x) \
    x = x / (1.0f + expf(-x));

#define ACTIVATION_HARDSWISH(x) \
    x = x >= 3.f ? x :          \
            x <= -3.f ? 0.f : (x * (x + 3.0f) / 6.0f);

#define ACTIVATION_FUNC(x, type)                                    \
    switch (type) {                                                 \
        case Activation::kIdentity: ACTIVATION_IDENTITY(x) break;   \
        case Activation::kReLU: ACTIVATION_RELU(x) break;           \
        case Activation::kELU: ACTIVATION_ELU(x) break;             \
        case Activation::kSELU: ACTIVATION_SELU(x) break;           \
        case Activation::kGELU: ACTIVATION_GELU(x) break;           \
        case Activation::kMISH: ACTIVATION_MISH(x) break;           \
        case Activation::kSwish: ACTIVATION_SWISH(x) break;         \
        case Activation::kHardSwish: ACTIVATION_HARDSWISH(x) break; \
        default: break;                                             \
    }
