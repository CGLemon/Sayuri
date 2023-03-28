#pragma once

#include <vector>
#include <cstddef>

class FullyConnect {
public:
    FullyConnect() = delete;
    static void Forward(const size_t inputs_size,
                        const size_t outputs_size,
                        const std::vector<float> &input,
                        const std::vector<float> &weights,
                        const std::vector<float> &biases,
                        std::vector<float> &output, bool ReLU);

    static std::vector<float> Innerproduct(const size_t inputs_size,
                                           const size_t outputs_size,
                                           const std::vector<float> &input,
                                           const std::vector<float> &weights,
                                           const std::vector<float> &biases,
                                           bool ReLU);
};


