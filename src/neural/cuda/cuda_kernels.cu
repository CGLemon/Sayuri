#include "neural/cuda/cuda_kernels.h"

#ifdef USE_CUDA

namespace CUDA {

template <typename T>
__global__ void add_vectors_kernel(T *a, T *b, T *c,
                                   int asize, int bsize, int size, bool relu) {

    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < size) {
        float aVal = (float)(a[i % asize]);
        float bVal = (float)(b[i % bsize]);
        float cVal = aVal + bVal;

        if (relu && (cVal < 0)) {
            cVal = 0;
        }
        c[i] = (T)cVal;
    }
}

template <typename T>
void add_vectors(T *a, T *b, T *c, int asize, int bsize,
                 int size, bool relu, cudaStream_t stream) {
    const int kBlockSize = KBLOCKSIZE;
    const int blocks = DivUp(size, kBlockSize);

    add_vectors_kernel<<<blocks, kBlockSize, 0, stream>>>(a, b, c, asize, bsize, size, relu);
    ReportCUDAErrors(cudaGetLastError());
}

template <typename T>
__global__ void add_spatial_kernel(T *a, T *b, T *c,
                                   int asize, int bsize, int size,
                                   int spatial, bool relu) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < size) {
        float aVal = (float)(a[i % asize]);
        float bVal = (float)(b[(i / spatial) % bsize]);

        float cVal = aVal + bVal;
        if (relu && (cVal < 0)) {
            cVal = 0;
        }
        c[i] = (T)cVal;
    }
}

template <typename T>
void add_spatial(T *a, T *b, T *c,
                 int asize, int bsize, int size,
                 int spatial, bool relu, cudaStream_t stream) {
    const int kBlockSize = KBLOCKSIZE;
    const int blocks = DivUp(size, kBlockSize);

    add_spatial_kernel<<<blocks, kBlockSize, 0, stream>>>(a, b, c, asize, bsize, size, spatial, relu);
    ReportCUDAErrors(cudaGetLastError());
}

template <typename T>
__global__ void batchNorm_eltwise_kernel(T *data, const float *means, const float *stddevs,
                                         int N, int C, int spatial, const T *eltwise, bool relu) {

    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int size = N * C * spatial;
    if (index < size) {
        int wIndex = (index / (spatial)) % C;

        float el = data[index];
        float mean = means[wIndex];
        float scale_stddev = stddevs[wIndex];

        el -= mean;
        el *= scale_stddev;
        el += (float)eltwise[index];

        if (relu && el < 0) {
            el = 0;
        }
        data[index] = (T)el;
    }
}


template <typename T>
__global__ void batchNorm_kernel(T *data, const float *means, const float *stddevs,
                                 int N, int C, int spatial, bool relu) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int size = N * C * spatial;
    if (index < size) {
        int wIndex = (index / (spatial)) % C;

        float el = data[index];
        float mean = means[wIndex];
        float scale_stddev = stddevs[wIndex];

        el -= mean;
        el *= scale_stddev;

        if (relu && el < 0) {
            el = 0;
        }
        data[index] = (T)el;
    }
}

template <typename T>
void batchnorm(T *data, const float *means, const float *stddevs,
               int batch, int channels, int spatial_size,
               const T *eltwise, bool relu, cudaStream_t stream) {
    const int total_elements = batch * channels * spatial_size;
    const int kBlockSize = KBLOCKSIZE;
    const int blocks = DivUp(total_elements, kBlockSize);
    if (eltwise) {
        batchNorm_eltwise_kernel<<<blocks, kBlockSize, 0 ,stream>>>(data, means, stddevs, batch,
                                                                    channels, spatial_size, eltwise, relu);
    } else {
        batchNorm_kernel<<<blocks, kBlockSize, 0, stream>>>(data, means, stddevs, batch,
                                                            channels, spatial_size, relu);
    }

    ReportCUDAErrors(cudaGetLastError());
}

template <typename T>
__global__ void im2col_kernel(int filter_size, int pad, int C, int H, int W,
                              int output_h, int output_w, T *data_im,
                              T *data_col) {
    int total_elements = C * H * W;
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index < total_elements) {
        int CWH_size = C * W * H;
        int Windex = index % CWH_size;
        int w_out = Windex % output_w;
        int h_index = Windex / output_w;
        int h_out = h_index % output_h;

        int channel_in = h_index / output_h;
        int channel_out = channel_in * filter_size * filter_size;

        int h_in = h_out - pad;
        int w_in = w_out - pad;

        float *data_col_ptr = data_col;
        data_col_ptr +=
            (channel_out * output_h + h_out) * output_w + w_out;
        const float *data_im_ptr = data_im;
        data_im_ptr += (channel_in * H + h_in) * W + w_in;

        for (int kernel_row = 0; kernel_row < filter_size; ++kernel_row) {
            for (int kernel_col = 0; kernel_col < filter_size; ++kernel_col) {
                int h = h_in + kernel_row;
                int w = w_in + kernel_col;
                *data_col_ptr = (h >= 0 && w >= 0 && h < H && w < W)
                                    ? data_im_ptr[kernel_row * W + kernel_col] : 0;
                data_col_ptr += output_w * output_h;
            }
        }
    }
}

template <typename T>
void im2col(int filter_size, int channels, int H, int W,
            T *input, T *output, cudaStream_t stream) {
    const int total_elements = channels * H * W;
    const int kBlockSize = KBLOCKSIZE;
    const int blocks = DivUp(total_elements, kBlockSize);

    const int pad = (filter_size / 2);
    const int output_h = H + 2 * pad - filter_size + 1;
    const int output_w = W + 2 * pad - filter_size + 1;

    im2col_kernel<<<blocks, kBlockSize, 0, stream>>>(filter_size, pad, channels, H, W,
                                                     output_h, output_w, input, output);
    ReportCUDAErrors(cudaGetLastError());
}

template <typename T>
__global__ void global_pool_kernel(T *input, T *output, T b_coeff, int N, int C, int spatial) {
    int total_elements = N * C;
    int index = threadIdx.x + blockDim.x * blockIdx.x; // index = 0 ~ batch * channels
    if (index < total_elements) {
        float *input_ptr = input + index * spatial;
        float sum = 0;
        float max = -5000.0f; // crazy negative value

        for (int i = 0; i < spatial; ++i) {
            float val = input_ptr[i];
            sum += val;
            if (val > max) {
                max = val;
            }
        }

        float mean = sum / spatial;

        int n = index / C;
        int c = index % C;
        int offset = c + n * 3 * C;

        output[offset + 0 * C] = (T)mean;
        output[offset + 1 * C] = (T)mean * b_coeff;
        output[offset + 2 * C] = (T)max;
    }
}

template <typename T>
void global_pool(T *input, T *output, T b_coeff, int batch, int channels, int spatial_size, cudaStream_t stream) {
    const int size = batch * channels;
    const int gpool_kBlockSize = 64;
    const int blocks = DivUp(size, gpool_kBlockSize);
    global_pool_kernel<<<blocks, gpool_kBlockSize, 0, stream>>>(input, output, b_coeff,
                                                                batch, channels, spatial_size);
    ReportCUDAErrors(cudaGetLastError());
}

template <typename T>
__global__ void head_global_pool_kernel(T *input, T *output, T b_coeff0, T b_coeff1, int N, int C, int spatial) {
    int total_elements = N * C;
    int index = threadIdx.x + blockDim.x * blockIdx.x; // index = 0 ~ batch * channels
    if (index < total_elements) {
        float *input_ptr = input + index * spatial;
        float sum = 0;

        for (int i = 0; i < spatial; ++i) {
            float val = input_ptr[i];
            sum += val;
        }

        float mean = sum / spatial;

        int n = index / C;
        int c = index % C;
        int offset = c + n * 3 * C;

        output[offset + 0 * C] = (T)mean;
        output[offset + 1 * C] = (T)mean * b_coeff0;
        output[offset + 2 * C] = (T)mean * b_coeff1;
    }
}

template <typename T>
void head_global_pool(T *input, T *output, T b_coeff0, T b_coeff1,
                      int batch, int channels, int spatial_size, cudaStream_t stream) {
    const int size = batch * channels;
    const int gpool_kBlockSize = 64;
    const int blocks = DivUp(size, gpool_kBlockSize);
    head_global_pool_kernel<<<blocks, gpool_kBlockSize, 0, stream>>>(input, output, b_coeff0, b_coeff1,
                                                                     batch, channels, spatial_size);
    ReportCUDAErrors(cudaGetLastError());
}

template <typename T>
__global__ void se_scale_kernel(const T *input, const T *se_bias, T *data,
                                int N, int C, int spatial) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int total_elements = N * C * spatial;
    if (index < total_elements) {
    
        int c = (index / spatial) % C;
        int n = (index / spatial) / C;
        int start_idx = n * 2 * C;

        float val = input[index];
        float gamma = se_bias[start_idx + c];
        gamma = 1.0f / (1.0f + exp(-gamma));

        float beta = se_bias[start_idx + c + C];
        float res = data[index];

        float op = gamma * val + beta + res;
        if (op < 0)
            op = 0;
        data[index] = (T)op;
    }
}

template<typename T>
void se_scale(const T *input, const T* se_bias, T* data,
                   int batch, int channels, int spatial_size, cudaStream_t stream) {
    const int total_elements = channels * spatial_size * batch;
    const int kBlockSize = KBLOCKSIZE;
    const int blocks = DivUp(total_elements, kBlockSize);

    se_scale_kernel<<<blocks, kBlockSize, 0, stream>>>(input, se_bias, data,
                                            batch, channels, spatial_size);
    ReportCUDAErrors(cudaGetLastError());
}

void gemm(bool TA, bool TB, int M, int N, int K, float ALPHA,
               const float *A_gpu, int lda, const float *B_gpu, int ldb,
               float BETA, float *C_gpu, int ldc, cublasHandle_t handle, cudaStream_t stream) {
    ReportCUBLASErrors(cublasSetStream(handle,stream));
    ReportCUBLASErrors(cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N),
                                   (TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K,
                                   &ALPHA, B_gpu, ldb, A_gpu, lda, &BETA, C_gpu, ldc));
}

template <typename T>
__global__ void swap_kernel(T *a, T *b, int size) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index < size) {
        T temp_a = a[index];
        T temp_b = b[index];
        a[index] = temp_b;
        b[index] = temp_a;
    }
} 

template <typename T>
void swap(T *a, T *b, int size, cudaStream_t stream) {
    const int kBlockSize = KBLOCKSIZE;
    const int blocks = DivUp(size, kBlockSize);
    swap_kernel<<<blocks, kBlockSize, 0, stream>>>(a, b, size);
}

template <typename T>
__global__ void copy_kernel(T *a, T *b, int size) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index < size) {
        T tmep = b[index];
        a[index] = tmep;
    }
} 

template<typename T>
void copy(T *a, T *b, int size, cudaStream_t stream) {
    const int kBlockSize = KBLOCKSIZE;
    const int blocks = DivUp(size, kBlockSize);
    copy_kernel<<<blocks, kBlockSize, 0, stream>>>(a, b, size);
}

template void batchnorm<float>(float *data, const float *means,
                               const float *stddevs, int N, int channels,
                               int spatial_size, const float *eltwise, bool relu, cudaStream_t stream);

template void add_vectors<float>(float *c, float *a, float *b, int size, int asize, int bsize, bool relu, cudaStream_t stream);

template void add_spatial<float>(float *a, float *b, float *c,
                                 int asize, int bsize, int size,
                                 int spatial, bool relu, cudaStream_t stream);

template void im2col<float>(int filter_size, int C, int H, int W,
                            float *data_im, float *data_col, cudaStream_t stream);

template void global_pool<float>(float *input, float *output, float b_coeff,
                                 int batch, int channels, int spatial_size, cudaStream_t stream);

template void head_global_pool<float>(float *input, float *output, float b_coeff0, float b_coeff1,
                                      int batch, int channels, int spatial_size, cudaStream_t stream);

template void se_scale<float>(const float *input, const float *se_bias, float *data,
                              int batch, int channels, int spatial_size, cudaStream_t stream);


template void swap<float>(float *a, float *b, int size, cudaStream_t stream);

template void copy<float>(float *a, float *b, int size, cudaStream_t stream);

} // namespace CUDA
#endif
