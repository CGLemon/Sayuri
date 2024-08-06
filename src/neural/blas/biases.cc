#include "neural/blas/biases.h"

#include <cassert>

void AddSpatialBiases::Forward(const size_t board_size,
                               const size_t channels,
                               std::vector<float> &input,
                               const std::vector<float> &biases,
                               const Activation act) {
    auto zero_vec = std::vector<float>{};
    Forward(
        board_size,
        channels,
        input,
        biases,
        zero_vec,
        act);
}

void AddSpatialBiases::Forward(const size_t board_size,
                               const size_t channels,
                               std::vector<float> &input,
                               const std::vector<float> &biases,
                               const std::vector<float> &residual,
                               const Activation act) {
    const auto width = board_size;
    const auto height = board_size;
    const auto spatial_size = width * height;

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

            ACTIVATION_FUNC(val, act);

            *input_ptr = val;
            input_ptr++;
        }
    }
}

void AddSpatialBiasesPost::Forward(const size_t board_size,
                                   const size_t channels,
                                   std::vector<float> &input,
                                   const std::vector<float> &biases,
                                   const Activation act,
                                   const std::vector<float> &residual) {
    const auto width = board_size;
    const auto height = board_size;
    const auto spatial_size = width * height;

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
            ACTIVATION_FUNC(val, act);

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
                              const Activation act) {
    assert(size == biases.size());
    for (auto o = size_t{0}; o < size; ++o) {
        float val = biases[o] + input[o];
        ACTIVATION_FUNC(val, act);
        input[o] = val;
    }
}
