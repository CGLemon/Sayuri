/*
    This file is part of ElephantArt.
    Copyright (C) 2021 Hung-Zhe Lin

    ElephantArt is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    ElephantArt is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with ElephantArt.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once
#ifdef USE_CUDA
#include <cassert>

#include "neural/cuda/cuda_common.h"
namespace CUDA {

template <typename T>
void add_vectors(T *c, T *a, T *b, int size, int asize, int bsize, bool relu);

template <typename T>
void add_spatial(T *a, T *b, T *c,
                 int asize, int bsize, int size,
                 int spatial, bool relu);

template <typename T>
void batchnorm(T *data, const float *means, const float *stddevs,
               int batch, int channels, int spatial_size,
               const T *eltwise, bool relu);


template <typename T>
void im2col(int filter_size, int channels, int H, int W,
            T *data_im, T *data_col);

template<typename T>
void global_avg_pool(T *input, T *output, int batch, int channels, int spatial_size);

template<typename T>
void se_scale(const T *input, const T* se_bias, T* data,
              int batch, int channels, int spatial_size);

template<typename T>
void input_pool(const T *bias, T *data,
                int batch, int channels, int spatial_size);


void gemm(bool TA, bool TB, int M, int N, int K, float ALPHA,
          const float *A_gpu, int lda, const float *B_gpu, int ldb,
          float BETA, float *C_gpu, int ldc, cublasHandle_t * handle);

template<typename T>
void swap(T *a, T *b, int size);

template<typename T>
void copy(T *a, T *b, int size);

} // namespace CUDA

#endif
