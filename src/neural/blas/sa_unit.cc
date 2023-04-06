#include "neural/blas/sa_unit.h"
#include "neural/blas/convolution.h"
#include "neural/blas/biases.h"

#include <cstddef>
#include <cmath>
#include <cassert>

void ChannelPooling::Forward(const size_t board_size,
                             const size_t channels,
                             const std::vector<float> &input,
                             std::vector<float> &output) {
    const auto width = board_size;
    const auto height = board_size;
    const auto spatial_size = width * height;
    const float b_coeff = ((float)board_size - kAvgBSize) / 10.f;
    const float *input_ptr = input.data();

    for (auto b = size_t{0}; b < spatial_size; ++b) {
        float sum = 0.0f;
        float max = 0.0f;
        for (auto c = size_t{0}; c < channels; ++c) {
            float val = input_ptr[b + c * spatial_size];
            sum += val;
            max = std::max(val, max);
        }
        input_ptr++;
        const float mean = sum / (float)spatial_size;

        output[b + 0 * spatial_size] = mean;
        output[b + 1 * spatial_size] = mean * b_coeff;
        output[b + 2 * spatial_size] = max;
    }
}

void SAUnit::Forward(const size_t board_size,
                     const size_t channels,
                     std::vector<float> &input,
                     const std::vector<float> &residual,
                     const std::vector<float> &weights,
                     const std::vector<float> &biases,
                     std::vector<float> &workspace,
                     bool ReLU) {
    using Pooling = ChannelPooling;
    using Convolution7 = Convolution<7>;

    const auto width = board_size;
    const auto height = board_size;
    const auto spatial_size = width * height;
    auto pool = std::vector<float>(3 * spatial_size);
    auto scale = std::vector<float>(1 * spatial_size);

    Pooling::Forward(board_size, channels, input, pool);
    Convolution7::Forward(
        board_size, 3, 1,
        pool, weights,
        workspace, scale);
    AddSpatialBiases::Forward(
        board_size, 1,
        scale,
        biases, false);
    SAProcess(board_size, channels, input, residual, scale, ReLU);
}

void SAUnit::SAProcess(const size_t board_size,
                       const size_t channels,
                       std::vector<float> &input,
                       const std::vector<float> &residual,
                       const std::vector<float> &scale,
                       bool ReLU) {
    const auto width = board_size;
    const auto height = board_size;
    const auto spatial_size = width * height;

    const auto lambda_ReLU = [ReLU](const auto val) {
        return (val > 0.0f || (!ReLU)) ? val : 0.0f;
    };

    const auto lambda_sigmoid = [](const auto val) {
        return 1.0f / (1.0f + std::exp(-val));
    };

    auto input_ptr = input.data();
    const float *residual_ptr = residual.empty() ?
                                    nullptr : residual.data();

    for (auto i = size_t{0}; i < spatial_size; ++i) {
        float gamma = lambda_sigmoid(scale[i]);
        for (auto c = size_t{0}; c < channels; ++c) {
            float val = input_ptr[i + c * spatial_size];
            val *= gamma;
            if (residual_ptr) {
                val += *residual_ptr;
                residual_ptr++;
            }
            input_ptr[i + c * spatial_size] = lambda_ReLU(val);
        }
        input_ptr++;
    }
}
