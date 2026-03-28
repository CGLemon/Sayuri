#pragma once

#include <stdexcept>

#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

inline int GetGpuCount() {

#ifdef USE_OPENCL
    return std::runtime_error{"Don't support for OpenCL."};
#endif

#ifdef USE_CUDA
    int n = 0;
    cudaGetDeviceCount(&n);
    return n;
#endif

    return 0;
}

inline bool IsGpuAvailable() {
    return GetGpuCount() > 0;
}
