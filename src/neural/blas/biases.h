#pragma once

#include "neural/activation.h"

#include <vector>
#include <cstddef>

// step1. add biases
// step3. add residual (optional)
// step2. activation function
class AddSpatialBiases {
public:
    AddSpatialBiases() = delete;
    static void Forward(const size_t board_size,
                        const size_t channels,
                        std::vector<float> &input,
                        const std::vector<float> &biases,
                        const Activation act);
    static void Forward(const size_t board_size,
                        const size_t channels,
                        std::vector<float> &input,
                        const std::vector<float> &biases,
                        const std::vector<float> &residual,
                        const Activation act);
};

// step1. add biases
// step2. activation function
// step3. add residual
class AddSpatialBiasesPost {
public:
    AddSpatialBiasesPost() = delete;
    static void Forward(const size_t board_size,
                        const size_t channels,
                        std::vector<float> &input,
                        const std::vector<float> &biases,
                        const Activation act,
                        const std::vector<float> &residual);
};

// step1. add biases
// step2. activation function
class AddVectorBiases {
public:
    AddVectorBiases() = delete;
    static void Forward(const size_t size,
                        std::vector<float> &input,
                        const std::vector<float> &biases,
                        const Activation act);
};
