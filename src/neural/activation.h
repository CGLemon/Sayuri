#pragma once

#include <string>
#include <cassert>
#include <cctype>

enum class Activation : int {
    kIdentity = 0,
    kReLU = 1
};

inline Activation StringToAct(std::string val) {
    for (char &c: val) {
        c = std::tolower(c);
    }
    if (val == "relu") {
        return Activation::kReLU;
    }
    return Activation::kIdentity; // identity or unknown
}

#define ACTIVATION_IDENTITY(x) \
    ;

#define ACTIVATION_RELU(x) \
    x = x > 0.f ? x : 0.f;

#define ACTIVATION_FUNC(x, type)                                   \
    switch (type) {                                                \
        case Activation::kIdentity: ACTIVATION_IDENTITY(x) break;  \
        case Activation::kReLU: ACTIVATION_RELU(x) break;          \
        default: ;                                                 \
    }
