#include "neural/blas/batchnorm.h"

#include <cassert>

void Batchnorm::Forward(const size_t board_size,
                        const size_t channels,
                        std::vector<float> &input,
                        const std::vector<float> &means,
                        const std::vector<float> &stddevs,
                        const float *const eltwise,
                        const Activation act) {
    const auto width = board_size;
    const auto height = board_size;
    const auto spatial_size = width * height;

    float *input_ptr = input.data();
    const float *residual = eltwise;

    for (auto c = size_t{0}; c < channels; ++c) {
        const auto mean = means[c];
        const auto scale_stddev = stddevs[c];

        for (auto b = size_t{0}; b < spatial_size; b++) {
            float val = *input_ptr;
            val = scale_stddev * (val - mean);
            if (residual) {
                val += *residual;
                residual++;
            }

            ACTIVATION_FUNC(val, act);

            *input_ptr = val;
            input_ptr++;
        }
    }
}
