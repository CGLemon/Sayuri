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

void DepthwiseConvolution::Forward(const size_t board_size,
                                   const size_t filter_size,
                                   const size_t channels,
                                   const std::vector<float> &input,
                                   const std::vector<float> &weights,
                                   std::vector<float> &output) {
    const unsigned int width = board_size;
    const unsigned int height = board_size;
    const unsigned int spatial_size = width * height;
    const unsigned int filter_dim = filter_size * filter_size;

    const int pad = (filter_size / 2);
    const float *data_in = input.data();
    const float *data_w = weights.data();
    float *data_out = output.data();

    for (int channel = channels; channel--; data_in += spatial_size, data_out += spatial_size, data_w += filter_dim) {
        for (unsigned int row = 0; row < height; row++) {
            for (unsigned int col = 0; col < width; col++) {
                float val = 0.0f;
                for (unsigned int kernel_row = 0; kernel_row < filter_size; kernel_row++) {
                    for (unsigned int kernel_col = 0; kernel_col < filter_size;  kernel_col++) {
                        int input_row = -pad + kernel_row + row;
                        int input_col = -pad + kernel_col + col;
                        if (unsigned(input_row) < height && unsigned(input_col) < width) {
                            val += (data_in[input_row * width + input_col] *
                                        data_w[kernel_row * filter_size + kernel_col]);
                        }
                    }
                }
                data_out[row * width + col] = val;
            }
        }
    }
}
