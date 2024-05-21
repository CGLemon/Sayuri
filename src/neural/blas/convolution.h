#pragma once

#include <vector>
#include <cassert>
#include <cstddef>
#include "neural/blas/blas.h"

class Convolution1 {
public:
    Convolution1() = delete;
    static void Forward(const size_t board_size,
                        const size_t input_channels,
                        const size_t output_channels,
                        const std::vector<float> &input,
                        const std::vector<float> &weights,
                        std::vector<float> &col,
                        std::vector<float> &output);
};

template<unsigned int FILTERS>
class Convolution {
public:
    Convolution() = delete;
    static void Forward(const size_t board_size,
                        const size_t input_channels,
                        const size_t output_channels,
                        const std::vector<float> &input,
                        const std::vector<float> &weights,
                        std::vector<float> &col,
                        std::vector<float> &output);

    static size_t GetWorkspaceSize(const size_t board_size, const size_t input_channels);

private:
    static void Im2col(const size_t board_size,
                       const int channels,
                       const std::vector<float> &input,
                       std::vector<float> &col);
};

template<unsigned int FILTERS>
void Convolution<FILTERS>::Forward(const size_t board_size,
                                   const size_t input_channels,
                                   const size_t output_channels,
                                   const std::vector<float> &input,
                                   const std::vector<float> &weights,
                                   std::vector<float> &col,
                                   std::vector<float> &output) {
    constexpr unsigned int filter_size = FILTERS;
    const unsigned int width = board_size;
    const unsigned int height = board_size;
    const unsigned int spatial_size = width * height;

    constexpr int filter_len = filter_size * filter_size;
    const int filter_dim = filter_len * input_channels;

    // Weight shape (output, input, filter_size, filter_size)
    // 96 18 3 3
    // C←αAB + βC
    // outputs[96,19x19] = weights[96,18x3x3] x col[18x3x3,19x19]
    // M Number of rows in matrices A and C.
    // N Number of columns in matrices B and C.
    // K Number of columns in matrix A; number of rows in matrix B.
    // lda The size of the first dimention of matrix A; if you are
    // passing a matrix A[m][n], the value should be m.
    //    cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
    //                ldb, beta, C, N);

    Im2col(board_size, input_channels, input, col);
    Blas::ConvolutionSgemm((int)output_channels,
                           (int)spatial_size,
                           filter_dim,
                           1.0f,
                           weights.data(),
                           filter_dim,
                           col.data(),
                           (int)spatial_size,
                           0.0f,
                           output.data(),
                           (int)spatial_size);
}

template<unsigned int FILTERS>
void Convolution<FILTERS>::Im2col(const size_t board_size,
                                  const int channels,
                                  const std::vector<float> &input,
                                  std::vector<float> &output) {
    constexpr unsigned int filter_size = FILTERS;
    const unsigned int width = board_size;
    const unsigned int height = board_size;
    const unsigned int spatial_size = width * height;

    constexpr int pad = (filter_size / 2);
    unsigned int output_h = height + 2 * pad - filter_size + 1;
    unsigned int output_w = width + 2 * pad - filter_size + 1;

    const float *data_im = input.data();
    float *data_col = output.data();

    for (int channel = channels; channel--; data_im += spatial_size) {
        for (unsigned int kernel_row = 0; kernel_row < filter_size; kernel_row++) {
            for (unsigned int kernel_col = 0; kernel_col < filter_size;  kernel_col++) {
                int input_row = -pad + kernel_row;
                for (int output_rows = output_h; output_rows; output_rows--) {
                    if (unsigned(input_row) < height) {
                        int input_col = -pad + kernel_col;
                        for (int output_col = output_w; output_col; output_col--) {
                            if (unsigned(input_col) < width) {
                                *(data_col++) = data_im[input_row * width + input_col];
                            } else {
                                *(data_col++) = 0;
                            }
                            input_col++;
                        }
                    } else {
                        for (int output_cols = output_w; output_cols; output_cols--) {
                            *(data_col++) = 0;
                        }
                    }
                    input_row++;
                }
            }
        }
    }
}

template<unsigned int FILTERS>
size_t Convolution<FILTERS>::GetWorkspaceSize(const size_t board_size, const size_t input_channels) {
    const auto width = board_size;
    const auto height = board_size;

    constexpr auto filter_size = FILTERS;
    const auto filter_len = filter_size * filter_size;
    const auto filter_dim = filter_len * input_channels;
    return filter_dim * width * height;
}

class DepthwiseConvolution {
public:
    DepthwiseConvolution() = delete;
    static void Forward(const size_t board_size,
                        const size_t filter_size,
                        const size_t channels,
                        const std::vector<float> &input,
                        const std::vector<float> &weights,
                        std::vector<float> &output);
};
