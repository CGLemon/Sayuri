#pragma once

#ifdef USE_CUDA

#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <vector>

#ifdef USE_CUDNN
#include <cudnn.h>
#endif

#include <string>
#include <sstream>

namespace cuda {

static constexpr int kMaxSupportGPUs = 256; // Give it a large value.

#define KBLOCKSIZE 256

#ifdef USE_CUDNN
cudnnDataType_t GetCudnnDataType(bool fp16);
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

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

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

size_t GetCudaTypeSize(bool fp16);

void MallocAndCopy(bool fp16, void **cude_op,
                       const std::vector<float> &weights);

void MallocCudaOp(bool fp16, void **cude_op, size_t size);

void CopyToCudaOp(bool fp16, void **cude_op,
                      const std::vector<float> &inputs);

void CopyToHostOp(bool fp16, std::vector<float> &output, void **cude_op);

} // namespace cuda

#endif
