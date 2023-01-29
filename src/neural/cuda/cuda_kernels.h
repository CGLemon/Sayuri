#pragma once

#ifdef USE_CUDA
#include <cassert>

#include "neural/cuda/cuda_common.h"

namespace CUDA {

void add_vectors(float *c, float *a, float *b,
                 int size, int asize, int bsize, bool relu, cudaStream_t stream);

void add_spatial(float *data, const float *biases,
                 const float *eltwise, const float *mask,
                 int bsize, int batch, int channels, int spatial,
                 bool relu, cudaStream_t stream);

void batchnorm(float *data, const float *means, const float *stddevs,
               const float *eltwise, const float *mask,
               int batch, int channels, int spatial,
               bool relu, cudaStream_t stream);

void im2col(int filter_size, int C, int H, int W,
            float *data_im, float *data_col, cudaStream_t stream);

void im2col_batched(int filter_size, int N, int C, int H, int W,
                    float *data_im, float *data_col, cudaStream_t stream);


void global_pooling(float *input, float *output, const float *mask,
                    int batch, int channels, int spatial, cudaStream_t stream);

void head_global_pooling(float *input, float *output, const float *sqrt_mask,
                         int batch, int channels, int spatial, cudaStream_t stream);

void se_scale(const float *input, const float* se_bias,
              const float *mask, float *output,
              int batch, int channels, int spatial, cudaStream_t stream);

void winograd3_transform_in(const float *in, float *V,
                            int batch, int channels, int board_size, cudaStream_t stream);

void winograd3_transform_out(const float *M, const float *biases,
                             const float *eltwise, const float *mask,
                             float *out,
                             int batch, int channels, int board_size,
                             bool relu, cudaStream_t stream);

void gemm(bool TA, bool TB, int M, int N, int K, float ALPHA,
          const float *A_gpu, int lda, const float *B_gpu, int ldb,
          float BETA, float *C_gpu, int ldc, cublasHandle_t handle, cudaStream_t stream);

void gemm_strided_batched(bool TA, bool TB, int M, int N, int K, float ALPHA,
                          const float *A_gpu, int lda, int strideA, const float *B_gpu, int ldb, int strideB,
                          float BETA, float *C_gpu, int ldc, int strideC, int batchsize, cublasHandle_t handle, cudaStream_t stream);

} // namespace CUDA

#endif
