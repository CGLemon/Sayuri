#include "neural/blas/se_unit.h"
#include "neural/blas/fullyconnect.h"

#include <cmath>
#include <cassert>

template<>
void GlobalPooling<false>::Forward(const size_t board_size,
                                   const size_t channels,
                                   const std::vector<float> &input,
                                   std::vector<float> &output) {
    const auto width = board_size;
    const auto height = board_size;
    const auto spatial_size = width * height;
    const float b_coeff = ((float)board_size - kAvgBSize) / 10.f;
    const float *input_ptr = input.data();

    for (auto c = size_t{0}; c < channels; ++c) {
        float sum = 0.0f;
        float max = -5000.0f; // crazy negative value
        for (auto b = size_t{0}; b < spatial_size; ++b) {
            float val = *input_ptr;

            sum += val;
            max = std::max(val, max);

            input_ptr++;
        }

        const float mean = sum / (float)spatial_size;
        output[c + 0 * channels] = mean;
        output[c + 1 * channels] = mean * b_coeff;
        output[c + 2 * channels] = max;
    }
}

template<>
void GlobalPooling<true>::Forward(const size_t board_size,
                                  const size_t channels,
                                  const std::vector<float> &input,
                                  std::vector<float> &output) {
    const auto width = board_size;
    const auto height = board_size;
    const auto spatial_size = width * height;
    const float *input_ptr = input.data();

    const float b_diff = (float)board_size - kAvgBSize;
    const float b_coeff0 = b_diff / 10.f;
    const float b_coeff1 = b_diff * b_diff / 100.f - kBSizeVaraince;

    for (auto c = size_t{0}; c < channels; ++c) {
        float sum = 0.0f;
        for (auto b = size_t{0}; b < spatial_size; ++b) {
            float val = *input_ptr;

            sum += val;

            input_ptr++;
        }

        const float mean = sum / (float)spatial_size;
        output[c + 0 * channels] = mean;
        output[c + 1 * channels] = mean * b_coeff0;
        output[c + 2 * channels] = mean * b_coeff1;
    }
}

void SEUnit::Forward(const size_t board_size,
                     const size_t channels,
                     const size_t se_size,
                     std::vector<float> &input,
                     const std::vector<float> &residual,
                     const std::vector<float> &weights_w1,
                     const std::vector<float> &weights_b1,
                     const std::vector<float> &weights_w2,
                     const std::vector<float> &weights_b2) {
    using pooling = GlobalPooling<false>;
    auto pool = std::vector<float>(3 * channels);
    auto fc_out = std::vector<float>(se_size);

    pooling::Forward(board_size, channels, input, pool);
    FullyConnect::Forward(3*channels, se_size, pool, weights_w1, weights_b1, fc_out, true);
    FullyConnect::Forward(se_size, 2*channels, fc_out, weights_w2, weights_b2, pool, false);
    SEProcess(board_size, channels, input, residual, pool);
}

void SEUnit::SEProcess(const size_t board_size,
                       const size_t channels,
                       std::vector<float> &input,
                       const std::vector<float> &residual,
                       const std::vector<float> &scale) {
    const auto width = board_size;
    const auto height = board_size;
    const auto spatial_size = width * height;

    const auto lambda_ReLU = [](const auto val) {
        return (val > 0.0f) ? val : 0;
    };

    const auto lambda_sigmoid = [](const auto val) {
        return 1.0f / (1.0f + std::exp(-val));
    };

    auto gamma_ptr = scale.data();
    auto beta_ptr = scale.data() + channels;
    auto input_ptr = input.data();
    auto res_ptr = residual.data();

    for (auto c = size_t{0}; c < channels; ++c) {
        const auto gamma = lambda_sigmoid(*gamma_ptr);
        const auto beta = *beta_ptr;

        gamma_ptr++;
        beta_ptr++;

        for (auto i = size_t{0}; i < spatial_size; ++i) {
            float value = *input_ptr;
            *input_ptr = lambda_ReLU(gamma * value + beta + *res_ptr);
            input_ptr++;
            res_ptr++;
        }
    }
}
