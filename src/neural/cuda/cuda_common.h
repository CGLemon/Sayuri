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
