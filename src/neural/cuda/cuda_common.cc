#ifdef USE_CUDA

#include "neural/cuda/cuda_common.h"
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <sstream>

namespace CUDA {

const char* cublasGetErrorString(cublasStatus_t status) {
    switch(status) {
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE"; 
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH"; 
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"; 
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR"; 
    }
    return "unknown error";
}

void CublasError(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        const char *cause = cublasGetErrorString(status);
        auto err = std::ostringstream{};
        err << "CUBLAS error: " << cause;
        throw std::runtime_error(err.str());
    }
}

void CudaError(cudaError_t status) {
  if (status != cudaSuccess) {
        const char *cause = cudaGetErrorString(status);
        auto err = std::ostringstream{};
        err << "CUDA Error: " << cause;
        throw std::runtime_error(err.str());
  }
}

int GetDeviceCount() {
    int n = 0;
    ReportCUDAErrors(cudaGetDeviceCount(&n));
    return n;
}

int GetDevice() {
    int n = 0;
    ReportCUDAErrors(cudaGetDevice(&n));
    return n;
}

void SetDevice(int n) {
    cudaSetDevice(n);
}

#ifdef USE_CUDNN
void CudnnError(cudnnStatus_t status) {
    if (status != CUDNN_STATUS_SUCCESS) {
        const char *s = cudnnGetErrorString(status);
        std::cerr << "CUDA Error: " << s << "\n";
        exit(-1);
    }
}

cudnnHandle_t GetCudnnHandle() {
    static bool init[MAX_SUPPORT_GPUS] = {false};
    static cudnnHandle_t handle[MAX_SUPPORT_GPUS];
    int i = GetDevice();
    if(!init[i]) {
        cudnnCreate(&handle[i]);
        init[i] = true;
    }
    return handle[i];
}
#endif

cublasHandle_t GetBlasHandle() {
    static bool init[MAX_SUPPORT_GPUS] = {false};
    static cublasHandle_t handle[MAX_SUPPORT_GPUS];
    int i = GetDevice();
    if (!init[i]) {
        cublasCreate(&handle[i]);
        init[i] = true;
    }
    return handle[i];
}

void CudaHandel::ApplyOnCurrentDevice() {
#ifdef USE_CUDNN
    cudnn_handel = GetCudnnHandle();
#endif
    cublas_handel = GetBlasHandle();
}

bool IsUsingCuDNN() {
#ifdef USE_CUDNN
    return true;
#else
    return false;
#endif
}

std::string OutputSpec(const cudaDeviceProp &dev_prop) {
    auto out = std::ostringstream{};

    out << "  Device name: "             << dev_prop.name                       << '\n';
    out << "  Device memory(MiB): "      << dev_prop.totalGlobalMem/(1024*1024) << '\n';
    out << "  Memory per-block(KiB): "   << dev_prop.sharedMemPerBlock/1024     << '\n';
    out << "  Register per-block(KiB): " << dev_prop.regsPerBlock/1024          << '\n';
    out << "  Warp size: "               << dev_prop.warpSize                   << '\n';
    out << "  Memory pitch(MiB): "       << dev_prop.memPitch/(1024*1024)       << '\n';
    out << "  Constant Memory(KiB): "    << dev_prop.totalConstMem/1024         << '\n';
    out << "  Max thread per-block: "    << dev_prop.maxThreadsPerBlock         << '\n';
    out << "  Max thread dim: ("
            << dev_prop.maxThreadsDim[0] << ", "
            << dev_prop.maxThreadsDim[1] << ", "
            << dev_prop.maxThreadsDim[2] << ")\n";
    out << "  Max grid size: ("
            << dev_prop.maxGridSize[0] << ", "
            << dev_prop.maxGridSize[1] << ", "
            << dev_prop.maxGridSize[2] << ")\n";
    out << "  Clock: "             << dev_prop.clockRate/1000   << "(kHz)" << '\n';
    out << "  Texture Alignment: " << dev_prop.textureAlignment << '\n';

    return out.str();
}

std::string GetDevicesInfo() {
    auto out = std::stringstream{};

    int devicecount = GetDeviceCount();
    if (devicecount == 0) {
        throw std::runtime_error("No CUDA device");
    }

    int cuda_version;
    cudaDriverGetVersion(&cuda_version);
    {
        const auto major = cuda_version/1000;
        const auto minor = (cuda_version - major * 1000)/10;
        out << "CUDA version: "
                << " Major " << major
                << ", Minor " << minor << '\n';
    }

    out << "Using cuDNN: ";
    if (IsUsingCuDNN()) {
        out << "Yes\n";
#ifdef USE_CUDNN
        const auto cudnn_version = cudnnGetVersion();
        const auto major = cudnn_version/1000;
        const auto minor = (cudnn_version -  major * 1000)/100;
        out << "cuDNN version: "
                << " Major " << major
                << ", Minor " << minor << '\n';
#endif
    } else {
        out << "No\n";
    }

    out << "Number of CUDA devices: " << devicecount << '\n';

    for(int i = 0; i < devicecount; ++i) {
        out << "=== Device " << i <<"===\n";
        cudaDeviceProp device_prop;
        cudaGetDeviceProperties(&device_prop, i);
        out << OutputSpec(device_prop);
    }

    return out.str();
}
} // namespace CUDA

#endif
