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

#include <cstdio>
#include <cuda_runtime.h>
//#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cuda.h>

#ifdef USE_CUDNN
#include <cudnn.h>
#endif

namespace CUDA {

static constexpr auto MAX_SUPPORT_GPUS = 16;

#define KBLOCKSIZE 256

#ifdef USE_CUDNN
void CudnnError(cudnnStatus_t status);
#define ReportCUDNNErrors(status) CudnnError(status)
cudnnHandle_t cudnn_handle(int n);
#endif
void CublasError(cublasStatus_t status);
void CudaError(cudaError_t status);


#define ReportCUBLASErrors(status) CublasError(status)
#define ReportCUDAErrors(status) CudaError(status)

cublasHandle_t blas_handle(int n);

int get_devicecount();
int get_device(int n /* = 0 */);

inline static int DivUp(int a, int b) { return (a + b - 1) / b; }
bool is_using_cuDNN();


struct CudaHandel {
#ifdef USE_CUDNN
    cudnnHandle_t cudnn_handel;
#endif
    cublasHandle_t cublas_handel;

    void apply(int n);
};

void check_devices();

} // namespace CUDA

#endif
