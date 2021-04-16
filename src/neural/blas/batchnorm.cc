#include "neural/blas/batchnorm.h"

#include <cassert>

void Batchnorm::Forward(const size_t board_size,
                        const size_t channels,
                        std::vector<float> &input,
                        const std::vector<float> &means,
                        const std::vector<float> &stddevs,
                        const float *const eltwise,
                        const bool ReLU) {
    const auto width = board_size;
    const auto height = board_size;
    const auto spatial_size = width * height;

    const auto lambda_ReLU = [ReLU](const auto val) {
        return (val > 0.0f || (!ReLU)) ? val : 0.0f;
    };

    float *input_ptr = input.data();
    const float *res = eltwise;
    if (eltwise) {
        for (auto c = size_t{0}; c < channels; ++c) {
            const auto mean = means[c];
            const auto scale_stddev = stddevs[c];

            for (auto b = size_t{0}; b < spatial_size; b++) {
                float value = *input_ptr;
                value = scale_stddev * (value - mean) + *res;
                *input_ptr = lambda_ReLU(value);

                input_ptr++;
                res++;
            }
        }
    } else {
        for (auto c = size_t{0}; c < channels; ++c) {
            const auto mean = means[c];
            const auto scale_stddev = stddevs[c];

            for (auto b = size_t{0}; b < spatial_size; b++) {
                float value = *input_ptr;
                value = scale_stddev * (value - mean);
                *input_ptr = lambda_ReLU(value);
                input_ptr++;
            }
        }
    }
}

