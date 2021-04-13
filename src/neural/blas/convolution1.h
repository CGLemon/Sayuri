#pragma once

#include <vector>

class Convolution1 {
public:
    Convolution1() = delete;

    // Batched forward inference.
    static void Forward(const size_t batch_size,
                        const size_t board_size,
                        const size_t input_channels,
                        const size_t output_channels,
                        const std::vector<float> &input,
                        const std::vector<float> &weights,
                        std::vector<float> &output);

};
