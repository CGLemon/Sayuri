#ifdef USE_CUDA

#include "neural/cuda/cuda_common.h"
#include "utils/half.h"
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <cstring>

namespace cuda {

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
        err << "CUDA error: " << cause;
        throw std::runtime_error(err.str());
  }
}

#ifdef USE_CUDNN
cudnnDataType_t GetCudnnDataType(bool fp16) {
    return fp16 ? CUDNN_DATA_HALF :
                      CUDNN_DATA_FLOAT;
}

void CudnnError(cudnnStatus_t status) {
    if (status != CUDNN_STATUS_SUCCESS) {
        const char *cause = cudnnGetErrorString(status);
        auto err = std::ostringstream{};
        err << "CUDNN error: " << cause;
        throw std::runtime_error(err.str());
    }
}
#endif

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
    ReportCUDAErrors(cudaSetDevice(n));
}

void WaitToFinish(cudaStream_t s) {
    ReportCUDAErrors(cudaStreamSynchronize(s));
}

void CudaHandles::ApplyOnCurrentDevice() {
    if (initialized) {
        return;
    }

    ReportCUBLASErrors(cublasCreate(&cublas_handle));
#ifdef USE_CUDNN
    ReportCUDNNErrors(cudnnCreate(&cudnn_handle));
#endif
    ReportCUDAErrors(cudaStreamCreate(&stream));

    fp16 = has_tensor_cores = false;

#ifdef ENABLE_FP16
    // The supported table is here, https://en.wikipedia.org/wiki/CUDA

    cudaDeviceProp dev_prop = GetDeviceProp();
    if (dev_prop.major >= 7) {
        fp16 = has_tensor_cores = true;
    } else if (dev_prop.major == 6 ||
                   (dev_prop.major == 5 &&
                    dev_prop.minor >= 3)) {
        fp16 = true;
    }
#endif

    gpu_id = GetDevice();
    initialized = true;
}

void CudaHandles::Release() {
    if (initialized) {
        cudaStreamDestroy(stream);
        cublasDestroy(cublas_handle);
#ifdef USE_CUDNN
        cudnnDestroy(cudnn_handle);
#endif
        initialized = false;
    }
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

std::string GetBackendInfo() {
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
        out << "CUDA version:"
                << " Major " << major
                << ", Minor " << minor << '\n';
    }

    {
        out << "Use cuDNN: ";
#ifdef USE_CUDNN
        out << "Yes\n";
        const auto cudnn_version = cudnnGetVersion();
        const auto major = cudnn_version/1000;
        const auto minor = (cudnn_version -  major * 1000)/100;
        out << "cuDNN version:"
                << " Major " << major
                << ", Minor " << minor << '\n';
#else
        out << "No\n";
#endif
    }

    out << "Number of CUDA devices: " << devicecount << '\n';

    return out.str();
}

cudaDeviceProp GetDeviceProp() {
    cudaDeviceProp dev_prop;
    ReportCUDAErrors(cudaGetDeviceProperties(&dev_prop, GetDevice()));
    return dev_prop;
}

std::string GetCurrentDeviceInfo(CudaHandles *handles) {
    auto out = std::stringstream{};

    cudaDeviceProp dev_prop = GetDeviceProp();
    out << "=== Device: " << GetDevice() <<" ===\n";

    out << "  Name: " << dev_prop.name << '\n';
    out << "  Compute capability: "
            << dev_prop.major << "."
            << dev_prop.minor << '\n';
    if (handles->fp16) {
        out << "  Enable the FP16\n";
    } else {
        out << "  Disable the FP16\n";
    }
    if (handles->has_tensor_cores) {
        out << "  Enable the tensor cores\n";
    } else {
        out << "  Disable the tensor cores\n";
    }

    return out.str();
}

size_t GetCudaTypeSize(bool fp16) {
    return fp16 ? sizeof(half_float_t) :
                      sizeof(float);
}

void MallocAndCopy(bool fp16, void **cude_op, const std::vector<float> &weights) {
    size_t wsize = weights.size();
    if (fp16) {
        auto buf = std::vector<half_float_t>(wsize);
        for (size_t i = 0; i < wsize; ++i) {
            buf[i] = GetFp16(weights[i]); // aussume it is the normal number
        }
        size_t op_size = wsize * sizeof(half_float_t);
        ReportCUDAErrors(cudaMalloc(&(*cude_op), op_size));
        ReportCUDAErrors(cudaMemcpy(
            *cude_op, buf.data(), op_size, cudaMemcpyHostToDevice));
    } else {
        size_t op_size = wsize * sizeof(float);
        ReportCUDAErrors(cudaMalloc(&(*cude_op), op_size));
        ReportCUDAErrors(cudaMemcpy(
            *cude_op, weights.data(), op_size, cudaMemcpyHostToDevice));
    }
}

void MallocCudaOp(bool fp16, void **cude_op, size_t size) {
    size_t op_size = size;
    if (fp16) {
        op_size *= sizeof(half_float_t);
    } else {
        op_size *= sizeof(float);
    }
    ReportCUDAErrors(cudaMalloc(&(*cude_op), op_size));
}

void ZeroCopyToCuda(bool fp16, void **host_op,
                    const std::vector<float> &inputs) {
    size_t in_size = inputs.size();
    if (fp16) {
        auto buf = std::vector<half_float_t>(in_size);
        for (size_t i = 0; i < in_size; ++i) {
            buf[i] = GetFp16(inputs[i]); // aussume it is the normal number
        }
        size_t op_size = in_size * sizeof(half_float_t);
        std::memcpy(*host_op, buf.data(), op_size);
    } else {
        size_t op_size = in_size * sizeof(float);
        std::memcpy(*host_op, inputs.data(), op_size);
    }
}

void ZeroCopyToHost(bool fp16, std::vector<float> &outputs, void **host_op) {
    size_t out_size = outputs.size();
    if (fp16) {
        auto buf = std::vector<half_float_t>(out_size);
        size_t op_size = out_size * sizeof(half_float_t);
        std::memcpy(buf.data(), *host_op, op_size);

        for (size_t i = 0; i < out_size; ++i) {
            outputs[i] = GetFp32(buf[i]);
        }
    } else {
        size_t op_size = out_size * sizeof(float);
        std::memcpy(outputs.data(), *host_op, op_size);
    }
}

void CopyToCudaOp(bool fp16, void **cude_op,
                  const std::vector<float> &inputs,
                  void **pinned_op) {
    size_t in_size = inputs.size();
    if (fp16) {
        auto buf = std::vector<half_float_t>(in_size);
        for (size_t i = 0; i < in_size; ++i) {
            buf[i] = GetFp16(inputs[i]); // aussume it is the normal number
        }
        size_t op_size = in_size * sizeof(half_float_t);

        if (pinned_op) {
            std::memcpy(*pinned_op, buf.data(), op_size);
            ReportCUDAErrors(cudaMemcpy(
                *cude_op, *pinned_op, op_size, cudaMemcpyHostToDevice));
        } else {
            ReportCUDAErrors(cudaMemcpy(
                *cude_op, buf.data(), op_size, cudaMemcpyHostToDevice));
        }
    } else {
        size_t op_size = in_size * sizeof(float);
        if (pinned_op) {
            std::memcpy(*pinned_op, inputs.data(), op_size);
            ReportCUDAErrors(cudaMemcpy(
                *cude_op, *pinned_op, op_size, cudaMemcpyHostToDevice));
        } else {
            ReportCUDAErrors(cudaMemcpy(
                *cude_op, inputs.data(), op_size, cudaMemcpyHostToDevice));
        }
    }
}

void CopyToHostOp(bool fp16, std::vector<float> &outputs,
                  void **cude_op, void **pinned_op) {
    size_t out_size = outputs.size();
    if (fp16) {
        size_t op_size = out_size * sizeof(half_float_t);
        auto buf = std::vector<half_float_t>(out_size);
        if (pinned_op) {
            ReportCUDAErrors(cudaMemcpy(
                *pinned_op, *cude_op, op_size, cudaMemcpyDeviceToHost));
            std::memcpy(buf.data(), *pinned_op, op_size);
        } else {
            ReportCUDAErrors(cudaMemcpy(
                buf.data(), *cude_op, op_size, cudaMemcpyDeviceToHost));
        }
        for (size_t i = 0; i < out_size; ++i) {
            outputs[i] = GetFp32(buf[i]);
        }
    } else {
        size_t op_size = out_size * sizeof(float);
        if (pinned_op) {
            ReportCUDAErrors(cudaMemcpy(
                *pinned_op, *cude_op, op_size, cudaMemcpyDeviceToHost));
            std::memcpy(outputs.data(), *pinned_op, op_size);
        } else {
            ReportCUDAErrors(cudaMemcpy(
                outputs.data(), *cude_op, op_size, cudaMemcpyDeviceToHost));
        }
    }
}

} // namespace cuda

#endif
