#include "neural/blas/convolution.h"

void Convolution1::Forward(const size_t board_size,
                           const size_t input_channels,
                           const size_t output_channels,
                           const std::vector<float> &input,
                           const std::vector<float> &weights,
                           std::vector<float> &/* col */,
                           std::vector<float> &output) {
    const unsigned int width = board_size;
    const unsigned int height = board_size;
    const unsigned int spatial_size = width * height;

    Blas::ConvolutionSgemm((int)output_channels,
                           (int)spatial_size,
                           (int)input_channels,
                           1.0f,
                           weights.data(),
                           (int)input_channels,
                           input.data(),
                           (int)spatial_size,
                           0.0f,
                           output.data(),
                           (int)spatial_size);
}
