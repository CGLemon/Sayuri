#include "neural/blas/biases.h"

#include <cassert>

void AddSpatialBiases::Forward(const size_t board_size,
                               const size_t channels,
                               std::vector<float> &input,
                               const std::vector<float> &biases,
                               bool ReLU) {
    auto zero_vec = std::vector<float>{};
    Forward(
        board_size,
        channels,
        input,
        biases,
        zero_vec,
        ReLU);
}

void AddSpatialBiases::Forward(const size_t board_size,
                               const size_t channels,
                               std::vector<float> &input,
                               const std::vector<float> &biases,
                               const std::vector<float> &residual,
                               bool ReLU) {
    const auto width = board_size;
    const auto height = board_size;
    const auto spatial_size = width * height;

    const auto lambda_ReLU = [ReLU](const auto val) {
        return (val > 0.0f || (!ReLU)) ? val : 0.0f;
    };

    float *input_ptr = input.data();
    const float *biases_ptr = biases.empty() ?
                                  nullptr : biases.data();
    const float *residual_ptr = residual.empty() ?
                                    nullptr : residual.data();
    for (auto c = size_t{0}; c < channels; ++c) {
        float bias = 0.0f;
        if (biases_ptr) {
            bias = biases_ptr[c];
        }
        for (auto b = size_t{0}; b < spatial_size; b++) {
            float val = *input_ptr + bias;
            if (residual_ptr) {
                val += *residual_ptr;
                residual_ptr++;
            }
            *input_ptr = lambda_ReLU(val);
            input_ptr++;
        }
    }
}

void AddSpatialBiasesPost::Forward(const size_t board_size,
                                   const size_t channels,
                                   std::vector<float> &input,
                                   const std::vector<float> &biases,
                                   bool ReLU,
                                   const std::vector<float> &residual) {
    const auto width = board_size;
    const auto height = board_size;
    const auto spatial_size = width * height;

    const auto lambda_ReLU = [ReLU](const auto val) {
        return (val > 0.0f || (!ReLU)) ? val : 0.0f;
    };

    float *input_ptr = input.data();
    const float *biases_ptr = biases.empty() ?
                                  nullptr : biases.data();
    const float *residual_ptr = residual.empty() ?
                                    nullptr : residual.data();
    for (auto c = size_t{0}; c < channels; ++c) {
        float bias = 0.0f;
        if (biases_ptr) {
            bias = biases_ptr[c];
        }
        for (auto b = size_t{0}; b < spatial_size; b++) {
            float val = lambda_ReLU(*input_ptr + bias);
            if (residual_ptr) {
                val += *residual_ptr;
                residual_ptr++;
            }
            *input_ptr = val;
            input_ptr++;
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
