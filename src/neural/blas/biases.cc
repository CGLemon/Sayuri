#include "neural/blas/biases.h"

#include <cassert>

void AddSpatialBiases::Forward(const size_t board_size, 
                               const size_t channels,
                               std::vector<float> &input,
                               const std::vector<float> &biases,
                               bool ReLU) {
    const auto width = board_size;
    const auto height = board_size;
    const auto spatial_size = width * height;

    const auto lambda_ReLU = [ReLU](const auto val) {
        return (val > 0.0f || (!ReLU)) ? val : 0.0f;
    };

    float *input_ptr = input.data();
    for (auto c = size_t{0}; c < channels; ++c) {
        for (auto b = size_t{0}; b < spatial_size; b++) {
            float val = *input_ptr + biases[c];
            *input_ptr = lambda_ReLU(val);
            input_ptr++;
        }
    }
}

void AddSpatialBiases::Forward(const size_t board_size, 
                               const size_t channels,
                               std::vector<float> &input,
                               const std::vector<float> &biases,
                               std::vector<float> &eltwise,
                               bool ReLU) {
    const auto width = board_size;
    const auto height = board_size;
    const auto spatial_size = width * height;

    const auto lambda_ReLU = [ReLU](const auto val) {
        return (val > 0.0f || (!ReLU)) ? val : 0.0f;
    };

    float *input_ptr = input.data();
    float *eltwise_ptr = eltwise.data();
    for (auto c = size_t{0}; c < channels; ++c) {
        for (auto b = size_t{0}; b < spatial_size; b++) {
            float val = *input_ptr + biases[c] + *eltwise_ptr;
            *input_ptr = lambda_ReLU(val);
            input_ptr++;
            eltwise_ptr++;
        }
    }
}


void AddVectorBiases::Forward(const size_t size,
                              std::vector<float> &input,
                              const std::vector<float> &biases,
                              bool ReLU) {
    assert(size == biases.size());

    const auto lambda_ReLU = [ReLU](const auto val) {
        return (val > 0.0f || (!ReLU)) ? val : 0.0f;
    };

    for (auto o = size_t{0}; o < size; ++o) {
        input[o] = lambda_ReLU(biases[o] + input[o]);
    }
}
