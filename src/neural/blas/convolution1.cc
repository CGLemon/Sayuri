#include "neural/blas/convolution1.h"
#include "neural/blas/blas.h"

void Convolution1::Forward(const size_t batch_size,
                           const size_t board_size,
                           const size_t input_channels,
                           const size_t output_channels,
                           const std::vector<float> &input,
                           const std::vector<float> &weights,
                           std::vector<float> &output) {

    const auto width = board_size;
    const auto height = board_size;
    const auto spatial_size = width * height;

    for (size_t b = 0; b < batch_size; ++b) {
        const auto in_offset = b * spatial_size * input_channels;
        const auto out_offset = b * spatial_size * output_channels;
        Blas::ConvolutionSgemm((int)output_channels,
                               spatial_size,
                               (int)input_channels,
                               1.0f,
                               weights.data(),
                               (int)input_channels,
                               input.data() + in_offset,
                               spatial_size,
                               0.0f,
                               output.data() + out_offset,
                               spatial_size);
    }
}
