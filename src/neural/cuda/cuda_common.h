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

#include <string>
#include <sstream>

namespace CUDA {

static constexpr auto MAX_SUPPORT_GPUS = 64;

#define KBLOCKSIZE 256

#ifdef USE_CUDNN
void CudnnError(cudnnStatus_t status);
#define ReportCUDNNErrors(status) CudnnError(status)
#endif
void CublasError(cublasStatus_t status);
void CudaError(cudaError_t status);


#define ReportCUBLASErrors(status) CublasError(status)
#define ReportCUDAErrors(status) CudaError(status)


int GetDeviceCount();
int GetDevice();
void SetDevice(int n);

inline static int DivUp(int a, int b) { return (a + b - 1) / b; }
bool IsUsingCuDNN();

struct CudaHandel {
#ifdef USE_CUDNN
    cudnnHandle_t cudnn_handel;
#endif
    cublasHandle_t cublas_handel;

    void ApplyOnCurrentDevice();
};

std::string GetDevicesInfo();

} // namespace CUDA

#endif
