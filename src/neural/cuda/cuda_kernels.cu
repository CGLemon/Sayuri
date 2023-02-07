#include "neural/cuda/cuda_kernels.h"
#include "neural/winograd_helper.h"

#ifdef USE_CUDA

namespace CUDA {

__global__ void add_vectors_kernel(float *a, float *b, float *c,
                                   int asize, int bsize, int size, bool relu) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < size) {
        float aval = a[i % asize];
        float bval = b[i % bsize];
        float cval = aval + bval;

        if (relu && (cval < 0)) {
            cval = 0;
        }
        c[i] = cval;
    }
}

void add_vectors(float *a, float *b, float *c,
                 int asize, int bsize, int size,
                 bool relu, cudaStream_t stream) {
    const int block_size = KBLOCKSIZE;
    const int blocks = DivUp(size, block_size);

    add_vectors_kernel<<<blocks, block_size, 0, stream>>>(
        a, b, c, asize, bsize, size, relu);

    ReportCUDAErrors(cudaGetLastError());
}

__global__ void add_spatial_kernel(float *data, const float *biases,
                                   const float *eltwise, const float *mask,
                                   int bsize, int N, int C, int spatial, bool relu) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int size = N * C * spatial;
    if (index < size) {
        int batch = index / (C * spatial);
        int b_index = (index / spatial) % bsize;
        int s_index = index % spatial;

        float val = data[index];

        if (biases) {
            val += biases[b_index];
        }
        if (eltwise) {
            val += eltwise[index];
        }
        if (mask) {
            val *= mask[batch * spatial + s_index];
        }
        if (relu && val < 0) {
            val = 0;
        }
        data[index] = val;
    }
}

void add_spatial(float *data, const float *biases,
                 const float *eltwise, const float *mask,
                 int bsize, int batch, int channels, int spatial,
                 bool relu, cudaStream_t stream) {
    const int total_elements = batch * channels * spatial;
    const int block_size = KBLOCKSIZE;
    const int blocks = DivUp(total_elements, block_size);

    add_spatial_kernel<<<blocks, block_size, 0, stream>>>(
        data, biases, eltwise, mask, bsize, batch, channels, spatial, relu);

    ReportCUDAErrors(cudaGetLastError());
}

__global__ void batchnorm_kernel(float *data, const float *means, const float *stddevs,
                                 const float *eltwise, const float *mask,
                                 int N, int C, int spatial, bool relu) {

    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int size = N * C * spatial;
    if (index < size) {
        int batch = index / (C * spatial);
        int c_index = (index / spatial) % C;
        int s_index = index % spatial;

        float el = data[index];
        float mean = means[c_index];
        float scale_stddev = stddevs[c_index];

        el -= mean;
        el *= scale_stddev;
        if (eltwise) {
            el += eltwise[index];
        }
        if (mask) {
            el *= mask[batch * spatial + s_index];
        }
        if (relu && el < 0) {
            el = 0;
        }
        data[index] = el;
    }
}

void batchnorm(float *data, const float *means, const float *stddevs,
               const float *eltwise, const float *mask,
               int batch, int channels, int spatial,
               bool relu, cudaStream_t stream) {
    const int total_elements = batch * channels * spatial;
    const int block_size = KBLOCKSIZE;
    const int blocks = DivUp(total_elements, block_size);

    batchnorm_kernel<<<blocks, block_size, 0 ,stream>>>(
        data, means, stddevs, eltwise, mask, batch, channels, spatial, relu);

    ReportCUDAErrors(cudaGetLastError());
}

__global__ void im2col_kernel(int filter_size, int pad, int C, int H, int W,
                              int output_h, int output_w,
                              float *data_im,
                              float *data_col) {
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

void im2col(int filter_size, int channels, int H, int W,
            float *input, float *output, cudaStream_t stream) {
    const int total_elements = channels * H * W;
    const int block_size = KBLOCKSIZE;
    const int blocks = DivUp(total_elements, block_size);

    const int pad = (filter_size / 2);
    const int output_h = H + 2 * pad - filter_size + 1;
    const int output_w = W + 2 * pad - filter_size + 1;

    im2col_kernel<<<blocks, block_size, 0, stream>>>(
        filter_size, pad, channels, H, W, output_h, output_w, input, output);

    ReportCUDAErrors(cudaGetLastError());
}

__global__ void im2col_batched_kernel(int filter_size, int pad, int N, int C, int H, int W,
                                      int output_h, int output_w,
                                      float *data_im,
                                      float *data_col) {
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

void im2col_batched(int filter_size, int batch, int channels, int H, int W,
                    float *input, float *output, cudaStream_t stream) {
    const int total_elements = batch * channels * H * W;
    const int block_size = KBLOCKSIZE;
    const int blocks = DivUp(total_elements, block_size);

    const int pad = (filter_size / 2);
    const int output_h = H + 2 * pad - filter_size + 1;
    const int output_w = W + 2 * pad - filter_size + 1;

    im2col_batched_kernel<<<blocks, block_size, 0, stream>>>(
        filter_size, pad, batch, channels, H, W, output_h, output_w, input, output);

    ReportCUDAErrors(cudaGetLastError());
}

__global__ void global_pooling_kernel(float *input, float *output,
                                      const float *mask,
                                      int N, int C, int spatial) {
    int total_elements = N * C;
    int index = threadIdx.x + blockDim.x * blockIdx.x; // index = [0 ~ batch * channels]
    if (index < total_elements) {
        int n = index / C;
        int c = index % C;

        float *input_ptr = input + index * spatial;
        float vsum = 0;
        float vmax = -5000.0f; // crazy negative value
        float vspat = 0.f;

        #pragma unroll
        for (int i = 0; i < spatial; ++i) {
            float val = input_ptr[i];
            float vmask = 1.0f;

            if (mask) {
                vmask = mask[n * spatial + i];
            }

            vspat += vmask;
            vsum += val;
            if ((1.0f-vmask) * (-5000.0f) + val > vmax) {
                vmax = val;
            }
        }

        float vmean = vsum / vspat;

        int offset = c + n * 3 * C;

        output[offset + 0 * C] = vmean;
        output[offset + 1 * C] = vmean * (sqrt(vspat) - 14.f) * 0.1f;
        output[offset + 2 * C] = vmax;
    }
}

void global_pooling(float *input, float *output, const float *mask,
                    int batch, int channels, int spatial, cudaStream_t stream) {
    const int total_elements = batch * channels;
    const int block_size = KBLOCKSIZE;
    const int blocks = DivUp(total_elements, block_size);

    global_pooling_kernel<<<blocks, block_size, 0, stream>>>(
        input, output, mask, batch, channels, spatial);

    ReportCUDAErrors(cudaGetLastError());
}

__global__ void head_global_pooling_kernel(float *input, float *output,
                                           const float *sqrt_mask,
                                           int N, int C, int spatial) {
    int total_elements = N * C;
    int index = threadIdx.x + blockDim.x * blockIdx.x; // index = [0 ~ batch * channels]
    if (index < total_elements) {
        int n = index / C;
        int c = index % C;

        float *input_ptr = input + index * spatial;
        float vsum = 0;

        float vsqrt = sqrt((float)spatial);
        if (sqrt_mask) {
            vsqrt = sqrt_mask[n];
        }

        #pragma unroll
        for (int i = 0; i < spatial; ++i) {
            vsum += input_ptr[i];
        }

        float vmean = vsum / (vsqrt * vsqrt);

        int offset = c + n * 3 * C;

        output[offset + 0 * C] = vmean;
        output[offset + 1 * C] = vmean * (vsqrt - 14.f) * 0.1f;
        output[offset + 2 * C] = vmean * ((vsqrt - 14.f) * (vsqrt - 14.f) * 0.01f - 0.1f);
    }
}

void head_global_pooling(float *input, float *output,
                         const float *sqrt_mask,
                         int batch, int channels, int spatial, cudaStream_t stream) {
    const int total_elements = batch * channels;
    const int block_size = KBLOCKSIZE;
    const int blocks = DivUp(total_elements, block_size);

    head_global_pooling_kernel<<<blocks, block_size, 0, stream>>>(
        input, output, sqrt_mask, batch, channels, spatial);

    ReportCUDAErrors(cudaGetLastError());
}

__global__ void se_scale_kernel(const float *input,
                                const float *se_bias,
                                const float *mask,
                                float *output,
                                int N, int C, int spatial) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int total_elements = N * C * spatial;
    if (index < total_elements) {
        int c = (index / spatial) % C;
        int n = (index / spatial) / C;
        int s_index = index % spatial;
        int start_idx = n * 2 * C;

        float val = input[index];
        float gamma = se_bias[start_idx + c];
        gamma = 1.0f / (1.0f + exp(-gamma));

        float beta = se_bias[start_idx + c + C];
        float res = output[index];

        val = gamma * val + beta + res;

        if (mask) {
            val *= mask[n * spatial + s_index];
        }
        if (val < 0) {
            val = 0;
        }
        output[index] = val;
    }
}

void se_scale(const float *input, const float* se_bias,
              const float *mask, float* output,
              int batch, int channels, int spatial, cudaStream_t stream) {
    const int total_elements = channels * spatial * batch;
    const int block_size = KBLOCKSIZE;
    const int blocks = DivUp(total_elements, block_size);

    se_scale_kernel<<<blocks, block_size, 0, stream>>>(
        input, se_bias, mask, output, batch, channels, spatial);

    ReportCUDAErrors(cudaGetLastError());
}

__device__ __forceinline__ void multiply_bt(
    float * o0, float * o1, float * o2, float * o3, float * o4, float * o5,
    float i0,   float i1,   float i2,   float i3,   float i4,   float i5
) {
    const float SQ2 = kSqrt2;

    float i3m1 = i1 * -SQ2 + i3 * (SQ2 / 2.0f);
    float i4m2 = i2 * -2.0f + i4 * 1.0f;

    *o0 = i0 + i2 * (-5.0f/2.0f) + i4;
    *o1 = i3m1 + i4m2;
    *o2 = -i3m1 + i4m2;

    float i3m1_2 = i3 * (SQ2) + i1 * (-SQ2/2.0f);
    float i4m2_2 = i2 * (-1.0f/2.0f) + i4;

    *o3 = i3m1_2 + i4m2_2;
    *o4 = -i3m1_2 + i4m2_2;

    *o5 = i1 + i3 * (-5.0f/2.0f) + i5;
}

__device__ __forceinline__ void multiply_atv(
    float * o,
    float i0, float i1, float i2, float i3, float i4, float i5
) {
    const float SQ2 = kSqrt2;

    float t1p2 = (i1 + i2) * (1.0f / 2.0f);
    float t1m2 = (i1 - i2) * (SQ2/4.0f);
    float t3p4 = i3 + i4;
    float t3m4 = (i3 - i4) * (SQ2);

    o[0] = i0 + t1p2 + t1p2 + t3p4;
    o[1] = t1m2 + t1m2 + t3m4;
    o[2] = t1p2 + t3p4 + t3p4;
    o[3] = t1m2 + t3m4 + t3m4 + i5;
}

__device__ __forceinline__ void multiply_at(
    float * o0, float * o1, float * o2, float * o3,
    float i0,   float i1,   float i2,   float i3, float i4, float i5
) {
    float o[4];
    multiply_atv(o, i0, i1, i2, i3, i4, i5);

    *o0 = o[0];
    *o1 = o[1];
    *o2 = o[2];
    *o3 = o[3];
}

__global__ void transform_in_kernel(const float *in, float *V,
                                    const int C,
                                    const int Cpad, const int Ppad,
                                    const int board_size, const int batch_size) {
    const int W = board_size;
    const int H = board_size;
    const int WTILES = board_size / kWinogradM + (board_size % kWinogradM != 0);
    const int P = WTILES * WTILES;
    const int CPpad = Ppad * Cpad;

    const int spatial = W * H;

    const int index = threadIdx.x + blockDim.x * blockIdx.x;
    const int block = index % (batch_size * P);
    const int ch = index / (batch_size * P);

    const int batch = block / P;
    const int block_x = (block - P * batch) % WTILES;
    const int block_y = (block - P * batch) / WTILES;

    // 6x6 tiles overlap by 2
    const int yin = kWinogradM * block_y - 1;
    const int xin = kWinogradM * block_x - 1;

    if (block < batch_size * P && ch < C) {
        // Cache input tile and handle zero padding
        float x[kWinogradAlpha][kWinogradAlpha];

        #pragma unroll
        for (int i = 0; i < kWinogradAlpha; i++) {
            #pragma unroll
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

        float T1[kWinogradAlpha][kWinogradAlpha];
        float T2[kWinogradAlpha][kWinogradAlpha];

        #pragma unroll
        for (int j = 0; j < kWinogradAlpha; j++) {
            multiply_bt(
                &(T1[0][j]), &(T1[1][j]), &(T1[2][j]), &(T1[3][j]), &(T1[4][j]), &(T1[5][j]),
                x[j][0], x[j][1], x[j][2], x[j][3], x[j][4], x[j][5]
            );
        }

        #pragma unroll
        for (int i = 0; i < kWinogradAlpha; i++){
            multiply_bt(
                &(T2[i][0]),  &(T2[i][1]),  &(T2[i][2]),  &(T2[i][3]),  &(T2[i][4]),  &(T2[i][5]),
                T1[i][0], T1[i][1], T1[i][2], T1[i][3], T1[i][4], T1[i][5]
            );
        }

        const int offset = ch * Ppad + block;

        // Scatter each sub element in tile to separate matrices
        #pragma unroll
        for (int i = 0; i < kWinogradAlpha; i++) {
            #pragma unroll
            for (int j = 0; j < kWinogradAlpha; j++) {
                // vstore_net_t(T2[i][j], (i*kWinogradAlpha + j)*CPpad + offset, V);
                V[(i*kWinogradAlpha + j) * CPpad + offset] = T2[i][j];
            }
        }
    }
}

__global__ void transform_out_kernel(const float *M,
                                     const float *biases,
                                     const float *eltwise,
                                     const float *mask,
                                     float *Y,
                                     const int K,
                                     const int Kpad, const int Ppad,
                                     const int board_size, 
                                     const int batch_size, bool relu) {
    const int W = board_size;
    const int H = board_size;
    const int WTILES = board_size / kWinogradM + (board_size % kWinogradM != 0);
    const int P = WTILES * WTILES;

    const int index = threadIdx.x + blockDim.x * blockIdx.x;
    const int block = index % (batch_size * P);
    const int k = index / (batch_size * P);

    if (k < K && block < batch_size * P) {
        float temp[kWinogradM][kWinogradAlpha];
        float o[kWinogradM][kWinogradM];
        float bias = 0.f;
        if (biases) {
            bias = biases[k];
        }

        // M dimensions are [36, outputs, batch_size * tiles].
        // Plus zero padding from SGEMM.
        // const int offset = block * Kpad + k;
        const int offset = k * Ppad + block;

        // Calculates transpose(A).temp_m
        #pragma unroll
        for (int xn = 0; xn < kWinogradAlpha; xn++) {
            float temp_m0 = M[(0 * kWinogradAlpha + xn) * Kpad * Ppad + offset];
            float temp_m1 = M[(1 * kWinogradAlpha + xn) * Kpad * Ppad + offset];
            float temp_m2 = M[(2 * kWinogradAlpha + xn) * Kpad * Ppad + offset];
            float temp_m3 = M[(3 * kWinogradAlpha + xn) * Kpad * Ppad + offset];
            float temp_m4 = M[(4 * kWinogradAlpha + xn) * Kpad * Ppad + offset];
            float temp_m5 = M[(5 * kWinogradAlpha + xn) * Kpad * Ppad + offset];
            multiply_at(
                &(temp[0][xn]), &(temp[1][xn]), &(temp[2][xn]), &(temp[3][xn]),
                temp_m0, temp_m1, temp_m2, temp_m3, temp_m4, temp_m5
            );
        }

        // Calculates temp.A
        #pragma unroll
        for (int i = 0; i < kWinogradM; i++){
            float r[4];
            multiply_atv(
                r,
                temp[i][0], temp[i][1], temp[i][2], temp[i][3], temp[i][4], temp[i][5]
            );

            o[i][0] = r[0] + bias;
            o[i][1] = r[1] + bias;
            o[i][2] = r[2] + bias;
            o[i][3] = r[3] + bias;
        }

        const int batch = block / P;
        const int block_x = (block - P * batch) % WTILES;
        const int block_y = (block - P * batch) / WTILES;
        const int x = kWinogradM * block_x;
        const int y = kWinogradM * block_y;
        const int spatial = W * H;

        #pragma unroll
        for (int i = 0; i < kWinogradM; i++) {
            #pragma unroll
            for (int j = 0; j < kWinogradM; j++) {
                const int out_idx =
                    batch * K * spatial +
                    k * spatial +
                    (y + i) * W + (x + j);
                if (y + i < H && x + j < W) {
                    float val =  o[i][j];
                    if (eltwise) {
                        val += eltwise[out_idx];
                    }
                    if (mask) {
                        int spatial = board_size * board_size;
                        int s_index = out_idx % spatial;
                        val *= mask[batch * spatial + s_index];
                    }
                    if (relu && val < 0) {
                        val = 0.f;
                    }
                    Y[out_idx] = val;
                }
            }
        }
    }
}

void winograd3_transform_in(const float *in, float *V,
                            int batch, int channels, int board_size, cudaStream_t stream) {
    const int ptiles = GetWinogradP(board_size);
    const int total_elements = channels * batch * ptiles;

    const int block_size = KBLOCKSIZE;
    const int blocks = DivUp(total_elements, block_size);

    const int c_pad = channels;
    const int p_pad = batch * ptiles;

    transform_in_kernel<<<blocks, block_size, 0, stream>>>(
        in, V, channels, c_pad, p_pad, board_size, batch);

    ReportCUDAErrors(cudaGetLastError());
}

void winograd3_transform_out(const float *M, const float *biases,
                             const float *eltwise, const float *mask,
                             float *out,
                             int batch, int channels, int board_size,
                             bool relu, cudaStream_t stream) {
    const int ptiles = GetWinogradP(board_size);
    const int total_elements = channels * batch * ptiles;

    const int block_size = KBLOCKSIZE;
    const int blocks = DivUp(total_elements, block_size);

    const int k_pad = channels;
    const int p_pad = batch * ptiles;

    transform_out_kernel<<<blocks, block_size, 0, stream>>>(
        M, biases, eltwise, mask, out, channels,
        k_pad, p_pad, board_size, batch, relu);

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

} // namespace CUDA
#endif
