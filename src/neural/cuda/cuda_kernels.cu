#include "neural/cuda/cuda_kernels.h"
#include "neural/winograd_helper.h"

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
__global__ void batchnorm_eltwise_kernel(T *data, const float *means, const float *stddevs,
                                         int N, int C, int spatial, const T *eltwise, bool relu) {

    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int size = N * C * spatial;
    if (index < size) {
        int w_index = (index / (spatial)) % C;

        float el = data[index];
        float mean = means[w_index];
        float scale_stddev = stddevs[w_index];

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
__global__ void batchnorm_kernel(T *data, const float *means, const float *stddevs,
                                 int N, int C, int spatial, bool relu) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int size = N * C * spatial;
    if (index < size) {
        int w_index = (index / (spatial)) % C;

        float el = data[index];
        float mean = means[w_index];
        float scale_stddev = stddevs[w_index];

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
        batchnorm_eltwise_kernel<<<blocks, kBlockSize, 0 ,stream>>>(data, means, stddevs, batch,
                                                                    channels, spatial_size, eltwise, relu);
    } else {
        batchnorm_kernel<<<blocks, kBlockSize, 0, stream>>>(data, means, stddevs, batch,
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
        int w_index = index % CWH_size;
        int w_out = w_index % output_w;
        int h_index = w_index / output_w;
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
__global__ void im2col_batched_kernel(int filter_size, int pad, int N, int C, int H, int W,
                                      int output_h, int output_w, T *data_im,
                                      T *data_col) {
    int total_elements = N * C * H * W;
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index < total_elements) {
        int CWH_size = C * W * H;
        int n_index = index / CWH_size;
        int w_index = index % CWH_size;
        int w_out = w_index % output_w;
        int h_index = w_index / output_w;
        int h_out = h_index % output_h;

        int channel_in = h_index / output_h;
        int channel_out = channel_in * filter_size * filter_size;

        int h_in = h_out - pad;
        int w_in = w_out - pad;

        float *data_col_ptr = data_col;
        data_col_ptr += n_index * (C * filter_size * filter_size * output_h * output_w) +
                            (channel_out * output_h + h_out) * output_w + w_out;
        const float *data_im_ptr = data_im;
        data_im_ptr += n_index * CWH_size +
                          (channel_in * H + h_in) * W + w_in;

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
void im2col_batched(int filter_size, int batch, int channels, int H, int W,
                        T *input, T *output, cudaStream_t stream) {
    const int total_elements = batch * channels * H * W;
    const int kBlockSize = KBLOCKSIZE;
    const int blocks = DivUp(total_elements, kBlockSize);

    const int pad = (filter_size / 2);
    const int output_h = H + 2 * pad - filter_size + 1;
    const int output_w = W + 2 * pad - filter_size + 1;

    im2col_batched_kernel<<<blocks, kBlockSize, 0, stream>>>(filter_size, pad, batch, channels, H, W,
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

        #pragma unroll
        for (int i = 0; i < spatial; ++i) {
            sum += input_ptr[i];
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
    ReportCUBLASErrors(cublasSetStream(handle, stream));
    ReportCUBLASErrors(cublasSgemm(
                           handle,
                           (TB ? CUBLAS_OP_T : CUBLAS_OP_N),
                           (TA ? CUBLAS_OP_T : CUBLAS_OP_N),
                           N, M, K,
                           &ALPHA,
                           B_gpu, ldb,
                           A_gpu, lda,
                           &BETA,
                           C_gpu, ldc));
}

void gemm_strided_batched(bool TA, bool TB, int M, int N, int K, float ALPHA,
                              const float *A_gpu, int lda, int strideA, const float *B_gpu, int ldb, int strideB,
                              float BETA, float *C_gpu, int ldc, int strideC, int batchsize, cublasHandle_t handle, cudaStream_t stream) {
    ReportCUBLASErrors(cublasSetStream(handle, stream));
    ReportCUBLASErrors(cublasSgemmStridedBatched(
                           handle,
                           (TB ? CUBLAS_OP_T : CUBLAS_OP_N),
                           (TA ? CUBLAS_OP_T : CUBLAS_OP_N),
                           N, M, K,
                           &ALPHA,
                           B_gpu, ldb, strideB,
                           A_gpu, lda, strideA,
                           &BETA,
                           C_gpu, ldc, strideC,
                           batchsize));
}

template <typename T>
__device__ __forceinline__ void multiply_bt(
    T * o0, T * o1, T * o2, T * o3, T * o4, T * o5,
    T i0,   T i1,   T i2,   T i3,   T i4,   T i5
) {
    const float SQ2 = kSqrt2;

    T i3m1 = i1 * -SQ2 + i3 * (SQ2 / 2.0f);
    T i4m2 = i2 * -2.0f + i4 * 1.0f;

    *o0 = i0 + i2 * (-5.0f/2.0f) + i4;
    *o1 = i3m1 + i4m2;
    *o2 = -i3m1 + i4m2;

    T i3m1_2 = i3 * (SQ2) + i1 * (-SQ2/2.0f);
    T i4m2_2 = i2 * (-1.0f/2.0f) + i4;

    *o3 = i3m1_2 + i4m2_2;
    *o4 = -i3m1_2 + i4m2_2;

    *o5 = i1 + i3 * (-5.0f/2.0f) + i5;
}

template <typename T>
__device__ __forceinline__ void multiply_atv(
    T * o,
    T i0, T i1, T i2, T i3, T i4, T i5
) {
    const float SQ2 = kSqrt2;

    T t1p2 = (i1 + i2) * (1.0f / 2.0f);
    T t1m2 = (i1 - i2) * (SQ2/4.0f);
    T t3p4 = i3 + i4;
    T t3m4 = (i3 - i4) * (SQ2);

    o[0] = i0 + t1p2 + t1p2 + t3p4;
    o[1] = t1m2 + t1m2 + t3m4;
    o[2] = t1p2 + t3p4 + t3p4;
    o[3] = t1m2 + t3m4 + t3m4 + i5;
}

template <typename T>
__device__ __forceinline__ void multiply_at(
    T * o0, T * o1, T * o2, T * o3,
    T i0,   T i1,   T i2,   T i3, T i4, T i5
) {
    T o[4];
    multiply_atv(o, i0, i1, i2, i3, i4, i5);

    *o0 = o[0];
    *o1 = o[1];
    *o2 = o[2];
    *o3 = o[3];
}

template <typename T>
__global__ void transform_in_kernel(const T *in, T *V,
                                    const int C,
                                    const int Cpad, const int Ppad,
                                    const int board_size, const int batch_size) {
    const int W = board_size;
    const int H = board_size;
    const int WTILES = board_size / kWinogradM + (board_size % kWinogradM != 0);
    const int P = WTILES * WTILES;
    const int CPpad = Ppad * Cpad;

    const int spatial = W * H;

    // const int block = get_global_id(0);
    // const int ch = get_global_id(1);
    const int block = threadIdx.x;
    const int ch = blockIdx.x;

    const int batch = block / P;
    const int block_x = (block - P * batch) % WTILES;
    const int block_y = (block - P * batch) / WTILES;

    // 6x6 tiles overlap by 2
    const int yin = kWinogradM * block_y - 1;
    const int xin = kWinogradM * block_x - 1;

    if (block < batch_size * P && ch < C) {
        // Cache input tile and handle zero padding
        T x[kWinogradAlpha][kWinogradAlpha];
        for (int i = 0; i < kWinogradAlpha; i++) {
            for (int j = 0; j < kWinogradAlpha; j++) {
                int a = xin + j;
                int b = yin + i;
                // x is transposed here for better layout later
                if (b >= 0 && a >= 0 && b < H && a < W) {
                    x[j][i] = in[batch * C * spatial + 
                                     ch * spatial + b * W + a];
                } else {
                    x[j][i] = 0;
                }
            }
        }

        // V dimensions are [36, input_channels, batch_size * tiles].
        // Padded with zeros as necessary for SGEMM
        // = [36, Cpad, Ppad]

        T T1[kWinogradAlpha][kWinogradAlpha];
        T T2[kWinogradAlpha][kWinogradAlpha];

        for (int j = 0; j < kWinogradAlpha; j++) {
            multiply_bt(
                &(T1[0][j]), &(T1[1][j]), &(T1[2][j]), &(T1[3][j]), &(T1[4][j]), &(T1[5][j]),
                x[j][0], x[j][1], x[j][2], x[j][3], x[j][4], x[j][5]
            );
        }

        for (int i = 0; i < kWinogradAlpha; i++){
            multiply_bt(
                &(T2[i][0]),  &(T2[i][1]),  &(T2[i][2]),  &(T2[i][3]),  &(T2[i][4]),  &(T2[i][5]),
                T1[i][0], T1[i][1], T1[i][2], T1[i][3], T1[i][4], T1[i][5]
            );
        }

        const int offset = ch * Ppad + block;

        // Scatter each sub element in tile to separate matrices
        for (int i = 0; i < kWinogradAlpha; i++) {
            for (int j = 0; j < kWinogradAlpha; j++) {
                // vstore_net_t(T2[i][j], (i*kWinogradAlpha + j)*CPpad + offset, V);
                V[(i*kWinogradAlpha + j) * CPpad + offset] = T2[i][j];
            }
        }
    }
}

template <typename T>
__global__ void transform_out_kernel(const T * M,
                                     T * Y,
                                     const int K,
                                     const int Kpad, const int Ppad,
                                     const int board_size, 
                                     const int batch_size) {
    const int W = board_size;
    const int H = board_size;
    const int WTILES = board_size / kWinogradM + (board_size % kWinogradM != 0);
    const int P = WTILES * WTILES;

    const int block = threadIdx.x;
    const int k = blockIdx.x;

    if (k < K && block < batch_size * P) {
        T temp[kWinogradM][kWinogradAlpha];
        T o[kWinogradM][kWinogradM];

        // M dimensions are [36, outputs, batch_size * tiles].
        // Plus zero padding from SGEMM.
        // const int offset = block * Kpad + k;
        const int offset = k * Ppad + block;

        // Calculates transpose(A).temp_m
        for (int xn = 0; xn < kWinogradAlpha; xn++) {
            T temp_m0 = M[(0 * kWinogradAlpha + xn) * Kpad * Ppad + offset];
            T temp_m1 = M[(1 * kWinogradAlpha + xn) * Kpad * Ppad + offset];
            T temp_m2 = M[(2 * kWinogradAlpha + xn) * Kpad * Ppad + offset];
            T temp_m3 = M[(3 * kWinogradAlpha + xn) * Kpad * Ppad + offset];
            T temp_m4 = M[(4 * kWinogradAlpha + xn) * Kpad * Ppad + offset];
            T temp_m5 = M[(5 * kWinogradAlpha + xn) * Kpad * Ppad + offset];
            multiply_at(
                &(temp[0][xn]), &(temp[1][xn]), &(temp[2][xn]), &(temp[3][xn]),
                temp_m0, temp_m1, temp_m2, temp_m3, temp_m4, temp_m5
            );
        }

        // Calculates temp.A
        for (int i = 0; i < kWinogradM; i++){
            T r[4];
            multiply_atv(
                r,
                temp[i][0], temp[i][1], temp[i][2], temp[i][3], temp[i][4], temp[i][5]
            );

            o[i][0] = r[0];
            o[i][1] = r[1];
            o[i][2] = r[2];
            o[i][3] = r[3];
        }

        const int batch = block / P;
        const int block_x = (block - P * batch) % WTILES;
        const int block_y = (block - P * batch) / WTILES;
        const int x = kWinogradM * block_x;
        const int y = kWinogradM * block_y;
        const int spatial = W * H;

        for (int i = 0; i < kWinogradM; i++) {
            for (int j = 0; j < kWinogradM; j++) {
                const int out_idx =
                    batch * K * spatial +
                    k * spatial +
                    (y + i) * W + (x + j);
                if (y + i < H && x + j < W) {
                    const T oval = o[i][j];
                    Y[out_idx] = oval;
                }
            }
        }
    }
}

template<typename T>
void winograd3_transform_in(const T *in, T *V,
                            int batch, int channels, int board_size, cudaStream_t stream) {
    const int ptiles = GetWinogradP(board_size);
    const int blocks_size = batch * ptiles;
    const int blocks = channels;

    const int k_ceil = channels;
    const int n_ceil = batch * ptiles;

    transform_in_kernel<<<blocks, blocks_size, 0, stream>>>(
        in, V, channels, k_ceil, n_ceil, board_size, batch);
}

template<typename T>
void winograd3_transform_out(const T *M, T *out,
                             int batch, int channels, int board_size, cudaStream_t stream) {
    const int ptiles = GetWinogradP(board_size);
    const int blocks_size = batch * ptiles;
    const int blocks = channels;

    const int m_ceil = channels;
    const int n_ceil = batch * ptiles;

    transform_out_kernel<<<blocks, blocks_size, 0, stream>>>(
        M, out, channels, m_ceil, n_ceil, board_size, batch);
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

template void im2col_batched<float>(int filter_size, int N, int C, int H, int W,
                                    float *data_im, float *data_col, cudaStream_t stream);


template void global_pool<float>(float *input, float *output, float b_coeff,
                                 int batch, int channels, int spatial_size, cudaStream_t stream);

template void head_global_pool<float>(float *input, float *output, float b_coeff0, float b_coeff1,
                                      int batch, int channels, int spatial_size, cudaStream_t stream);

template void se_scale<float>(const float *input, const float *se_bias, float *data,
                              int batch, int channels, int spatial_size, cudaStream_t stream);

template void winograd3_transform_in(const float *in, float *V,
                                    int batch, int channels, int board_size, cudaStream_t stream);

template void winograd3_transform_out(const float *M, float *out,
                                     int batch, int channels, int board_size, cudaStream_t stream);

} // namespace CUDA
#endif
