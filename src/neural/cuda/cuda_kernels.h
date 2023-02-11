#pragma once

#ifdef USE_CUDA
#include <cassert>
#include <type_traits>

#include "neural/cuda/cuda_common.h"

namespace CUDA {

template <typename T>
void add_vectors(T *c, T *a, T *b,
                 int size, int asize, int bsize,
                 bool relu, cudaStream_t stream);

template <typename T>
void add_spatial(T *data, const T *biases,
                 const T *eltwise, const T *mask,
                 int bsize, int batch, int channels, int spatial,
                 bool relu, cudaStream_t stream);

template <typename T>
void batchnorm(T *data, const T *means, const T *stddevs,
               const T *eltwise, const T *mask,
               int batch, int channels, int spatial,
               bool relu, cudaStream_t stream);

template <typename T>
void im2col(int filter_size, int C, int H, int W,
            T *data_im, T *data_col, cudaStream_t stream);

template <typename T>
void im2col_batched(int filter_size, int N, int C, int H, int W,
                    T *data_im, T *data_col, cudaStream_t stream);

template <typename T>
void global_pooling(T *input, T *output, const T *mask,
                    int batch, int channels, int spatial, cudaStream_t stream);

template <typename T>
void head_global_pooling(T *input, T *output, const T *sqrt_mask,
                         int batch, int channels, int spatial, cudaStream_t stream);

template <typename T>
void se_scale(const T *input, const T *se_bias,
              const T *mask, T *output,
              int batch, int channels, int spatial, cudaStream_t stream);

template <typename T>
void winograd3_transform_in(const T *in, T *V,
                            int batch, int channels, int board_size, cudaStream_t stream);

template <typename T>
void winograd3_transform_out(const T *M, const T *biases,
                             const T *eltwise, const T *mask,
                             T *out,
                             int batch, int channels, int board_size,
                             bool relu, cudaStream_t stream);

template <typename T>
void gemm(bool TA, bool TB, int M, int N, int K, T ALPHA,
          const T *A_gpu, int lda, const T *B_gpu, int ldb,
          T BETA, T *C_gpu, int ldc, cublasHandle_t handle, cudaStream_t stream);

template <typename T>
void gemm_strided_batched(bool TA, bool TB, int M, int N, int K, T ALPHA,
                          const T *A_gpu, int lda, int strideA, const T *B_gpu, int ldb, int strideB,
                          T BETA, T *C_gpu, int ldc, int strideC, int batchsize, cublasHandle_t handle, cudaStream_t stream);

} // namespace CUDA

#endif
