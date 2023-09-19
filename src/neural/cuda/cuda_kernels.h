#pragma once

#ifdef USE_CUDA
#include <cassert>
#include <type_traits>

#include "neural/cuda/cuda_common.h"

namespace cuda {

template <typename T>
void add_vectors(T *c, T *a, T *b,
                 int size, int asize, int bsize,
                 bool relu, cudaStream_t stream);

template <typename T>
void add_spatial(T *data, const T *biases,
                 const T *residual, const T *mask,
                 int bsize, int batch, int channels, int spatial,
                 bool relu, cudaStream_t stream);

template <typename T>
void im2col_batched(T *data_col, T *data_im,
                    int filter_size, int batch,
                    int channels, int height, int width,
                    cudaStream_t stream);

template <typename T>
void global_pooling(T *output, T *input, const T *mask,
                    const T *sqrt_mask, int batch, int channels,
                    int spatial, cudaStream_t stream);

template <typename T>
void head_global_pooling(T *input, T *output, const T *sqrt_mask,
                         int batch, int channels, int spatial, cudaStream_t stream);

template <typename T>
void se_scale(T *output, const T *input, const T *residual,
              const T *se_biases, const T *mask, int batch,
              int channels, int spatial, bool relu, cudaStream_t stream);

template <typename T>
void winograd3_transform_in(T *V, const T *in, int batch,
                            int channels, int board_size, cudaStream_t stream);

template <typename T>
void winograd3_transform_out(T *out, const T *M, const T *biases,
                             const T *residual, const T *mask,
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

} // namespace cuda

#endif
