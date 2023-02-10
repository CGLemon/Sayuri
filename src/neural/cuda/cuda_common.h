#pragma once

#ifdef USE_CUDA

#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cuda.h>

#ifdef USE_CUDNN
#include <cudnn.h>
#endif

#include <string>
#include <sstream>

namespace CUDA {

static constexpr int kMaxSupportGPUs = 256; // Give it a large value.

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
void WaitToFinish(cudaStream_t s);

inline static int DivUp(int a, int b) { return (a + b - 1) / b; }

struct CudaHandles {
#ifdef USE_CUDNN
    cudnnHandle_t cudnn_handle;
#endif
    cublasHandle_t cublas_handle;

    cudaStream_t stream;

    int gpu_id;

    void ApplyOnCurrentDevice();
    void Release();
};

std::string GetBackendInfo();
std::string GetCurrentDeviceInfo();

} // namespace CUDA

#endif
