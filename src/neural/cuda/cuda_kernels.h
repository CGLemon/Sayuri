#pragma once

#ifdef USE_CUDA
#include <cassert>

#include "neural/cuda/cuda_common.h"
namespace CUDA {

template <typename T>
void add_vectors(T *c, T *a, T *b, int size, int asize, int bsize, bool relu, cudaStream_t stream);

template <typename T>
void add_spatial(T *a, T *b, T *c,
                 int asize, int bsize, int size,
                 int spatial, bool relu, cudaStream_t stream);

template <typename T>
void batchnorm(T *data, const float *means, const float *stddevs,
               int batch, int channels, int spatial_size,
               const T *eltwise, bool relu, cudaStream_t stream);


template <typename T>
void im2col(int filter_size, int C, int H, int W,
            T *data_im, T *data_col, cudaStream_t stream);

template <typename T>
void im2col_batched(int filter_size, int N, int C, int H, int W,
                    T *data_im, T *data_col, cudaStream_t stream);


template<typename T>
void global_pool(T *input, T *output, T b_coeff, int batch,
                 int channels, int spatial_size, cudaStream_t stream);

template<typename T>
void head_global_pool(T *input, T *output, T b_coeff0, T b_coeff1, int batch,
                      int channels, int spatial_size, cudaStream_t stream);


template<typename T>
void se_scale(const T *input, const T* se_bias, T* data,
              int batch, int channels, int spatial_size, cudaStream_t stream);

void gemm(bool TA, bool TB, int M, int N, int K, float ALPHA,
          const float *A_gpu, int lda, const float *B_gpu, int ldb,
          float BETA, float *C_gpu, int ldc, cublasHandle_t handle, cudaStream_t stream);

void gemm_strided_batched(bool TA, bool TB, int M, int N, int K, float ALPHA,
                              const float *A_gpu, int lda, int strideA, const float *B_gpu, int ldb, int strideB,
                              float BETA, float *C_gpu, int ldc, int strideC, int batchsize, cublasHandle_t handle, cudaStream_t stream);

template<typename T>
void winograd3_transform_in(const T *in, T *V,
                            int batch, int channels, int board_size, cudaStream_t stream);

template<typename T>
void winograd3_transform_out(const T *M, T *out,
                             int batch, int channels, int board_size, cudaStream_t stream);


} // namespace CUDA

#endif
