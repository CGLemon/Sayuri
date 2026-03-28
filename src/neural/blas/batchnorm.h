#pragma once

#include <cstddef>
#include <vector>

#include "neural/activation.h"

class Batchnorm {
public:
    Batchnorm() = delete;
    static void Forward(const size_t board_size,
                        const size_t channels,
                        std::vector<float>& input,
                        const std::vector<float>& means,
                        const std::vector<float>& stddevs,
                        const float* const eltwise = nullptr,
                        const Activation act = Activation::kIdentity);
};
