#include "neural/cuda/cuda_kernels.h"
#include "neural/winograd_helper.h"

#ifdef USE_CUDA

namespace cuda {

template <typename T>
__global__ void add_vectors_kernel(T *a, T *b, T *c,
                                   int asize, int bsize, int size, bool relu) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < size) {
        float aval = (float)(a[i % asize]);
        float bval = (float)(b[i % bsize]);
        float cval = aval + bval;

        if (relu && (cval < 0)) {
            cval = 0;
        }
        c[i] = (T)cval;
    }
}

template <typename T>
void add_vectors(T *a, T *b, T *c,
                 int asize, int bsize, int size,
                 bool relu, cudaStream_t stream) {
    const int block_size = KBLOCKSIZE;
    const int blocks = DivUp(size, block_size);

    add_vectors_kernel<<<blocks, block_size, 0, stream>>>(
        a, b, c, asize, bsize, size, relu);

    ReportCUDAErrors(cudaGetLastError());
}

template <typename T>
__global__ void add_spatial_kernel(T *data, const T *biases,
                                   const T *residual, const T *mask,
                                   int bsize, int N, int C, int spatial, bool relu) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int size = N * C * spatial;
    if (index < size) {
        int batch = index / (C * spatial);
        int b_index = (index / spatial) % bsize;
        int s_index = index % spatial;

        float val = (float)(data[index]);

        if (biases) {
            val += (float)(biases[b_index]);
        }
        if (residual) {
            val += (float)(residual[index]);
        }
        if (mask) {
            val *= (float)(mask[batch * spatial + s_index]);
        }

        if (relu && val < 0) {
            val = 0;
        }
        data[index] = (T)val;
    }
}

template <typename T>
void add_spatial(T *data, const T *biases,
                 const T *residual, const T *mask,
                 int bsize, int batch, int channels, int spatial,
                 bool relu, cudaStream_t stream) {
    const int total_elements = batch * channels * spatial;
    const int block_size = KBLOCKSIZE;
    const int blocks = DivUp(total_elements, block_size);

    add_spatial_kernel<<<blocks, block_size, 0, stream>>>(
        data, biases, residual, mask, bsize, batch, channels, spatial, relu);

    ReportCUDAErrors(cudaGetLastError());
}

// template <typename T>
// __global__ void batchnorm_kernel(T *data, const T *means, const T *stddevs,
//                                  const T *eltwise, const T *mask,
//                                  int N, int C, int spatial, bool relu) {
//
//     int index = threadIdx.x + blockDim.x * blockIdx.x;
//     int size = N * C * spatial;
//     if (index < size) {
//         int batch = index / (C * spatial);
//         int c_index = (index / spatial) % C;
//         int s_index = index % spatial;
// 
//         float el = (float)(data[index]);
//         float mean = (float)(means[c_index]);
//         float scale_stddev = (float)(stddevs[c_index]);
//
//         el -= mean;
//         el *= scale_stddev;
//         if (eltwise) {
//             el += (float)(eltwise[index]);
//         }
//         if (mask) {
//             el *= (float)(mask[batch * spatial + s_index]);
//         }
//         if (relu && el < 0) {
//             el = 0;
//         }
//         data[index] = (T)el;
//     }
// }
//
// template <typename T>
// void batchnorm(T *data, const T *means, const T *stddevs,
//                const T *eltwise, const T *mask,
//                int batch, int channels, int spatial,
//                bool relu, cudaStream_t stream) {
//     const int total_elements = batch * channels * spatial;
//     const int block_size = KBLOCKSIZE;
//     const int blocks = DivUp(total_elements, block_size);
//
//     batchnorm_kernel<<<blocks, block_size, 0 ,stream>>>(
//         data, means, stddevs, eltwise, mask, batch, channels, spatial, relu);
//
//     ReportCUDAErrors(cudaGetLastError());
// }

// template <typename T>
// __global__ void im2col_kernel(int filter_size, int pad, int C, int H, int W,
//                               int output_h, int output_w,
//                               T *data_im,
//                               T *data_col) {
//     int total_elements = C * H * W;
//     int index = threadIdx.x + blockDim.x * blockIdx.x;
//     if (index < total_elements) {
//         int CWH_size = C * W * H;
//         int w_index = index % CWH_size;
//         int w_out = w_index % output_w;
//         int h_index = w_index / output_w;
//         int h_out = h_index % output_h;
//
//         int channel_in = h_index / output_h;
//         int channel_out = channel_in * filter_size * filter_size;
//
//         int h_in = h_out - pad;
//         int w_in = w_out - pad;
//
//         T *data_col_ptr = data_col;
//         data_col_ptr +=
//             (channel_out * output_h + h_out) * output_w + w_out;
//         const T *data_im_ptr = data_im;
//         data_im_ptr += (channel_in * H + h_in) * W + w_in;
//
//         for (int kernel_row = 0; kernel_row < filter_size; ++kernel_row) {
//             for (int kernel_col = 0; kernel_col < filter_size; ++kernel_col) {
//                 int h = h_in + kernel_row;
//                 int w = w_in + kernel_col;
//                 *data_col_ptr = (h >= 0 && w >= 0 && h < H && w < W)
//                                     ? data_im_ptr[kernel_row * W + kernel_col] : (T)(0);
//                 data_col_ptr += output_w * output_h;
//             }
//         }
//     }
// }
//
// template <typename T>
// void im2col(int filter_size, int channels, int H, int W,
//             T *input, T *output, cudaStream_t stream) {
//     const int total_elements = channels * H * W;
//     const int block_size = KBLOCKSIZE;
//     const int blocks = DivUp(total_elements, block_size);
//
//     const int pad = (filter_size / 2);
//     const int output_h = H + 2 * pad - filter_size + 1;
//     const int output_w = W + 2 * pad - filter_size + 1;
// 
//     im2col_kernel<<<blocks, block_size, 0, stream>>>(
//         filter_size, pad, channels, H, W, output_h, output_w, input, output);
//
//     ReportCUDAErrors(cudaGetLastError());
// }

template <typename T>
__global__ void im2col_batched_kernel(T *data_col, T *data_im,
                                      int filter_size, int pad, int N, int C, int H, int W,
                                      int output_h, int output_w) {
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

        T *data_col_ptr = data_col;
        data_col_ptr += n_index * (C * filter_size * filter_size * output_h * output_w) +
                            (channel_out * output_h + h_out) * output_w + w_out;
        const T *data_im_ptr = data_im;
        data_im_ptr += n_index * CWH_size +
                          (channel_in * H + h_in) * W + w_in;

        for (int kernel_row = 0; kernel_row < filter_size; ++kernel_row) {
            for (int kernel_col = 0; kernel_col < filter_size; ++kernel_col) {
                int h = h_in + kernel_row;
                int w = w_in + kernel_col;
                *data_col_ptr = (h >= 0 && w >= 0 && h < H && w < W)
                                    ? data_im_ptr[kernel_row * W + kernel_col] : T(0);
                data_col_ptr += output_w * output_h;
            }
        }
    }
}

template <typename T>
void im2col_batched(T *data_col, T *data_im,
                    int filter_size, int batch,
                    int channels, int height, int width,
                    cudaStream_t stream) {
    const int total_elements = batch * channels * height * width;
    const int block_size = KBLOCKSIZE;
    const int blocks = DivUp(total_elements, block_size);

    const int pad = (filter_size / 2);
    const int output_h = height + 2 * pad - filter_size + 1;
    const int output_w = width + 2 * pad - filter_size + 1;

    im2col_batched_kernel<<<blocks, block_size, 0, stream>>>(
        data_col, data_im, filter_size, pad, batch,
        channels, height, width, output_h, output_w);

    ReportCUDAErrors(cudaGetLastError());
}

template <typename T>
__global__ void global_pooling_kernel(T *output, T *input, const T *mask,
                                      const T *sqrt_mask, int N, int C, int spatial) {
    int total_elements = N * C;
    int index = threadIdx.x + blockDim.x * blockIdx.x; // index = [0 ~ batch * channels]
    if (index < total_elements) {
        int n = index / C;
        int c = index % C;

        T *input_ptr = input + index * spatial;
        float vsum = 0;
        float vmax = -5000.0f; // crazy negative value
        float vsqrt = sqrt((float)spatial);
        if (sqrt_mask) {
            vsqrt = (float)(sqrt_mask[n]);
        }

        #pragma unroll
        for (int i = 0; i < spatial; ++i) {
            float val = (float)(input_ptr[i]);
            float vmask = 1.0f;

            if (mask) {
                vmask = (float)(mask[n * spatial + i]);
            }

            vsum += val;
            if ((1.0f-vmask) * (-5000.0f) + val > vmax) {
                vmax = val;
            }
        }

        float vmean = vsum / (vsqrt*vsqrt);

        int offset = c + n * 3 * C;

        output[offset + 0 * C] = (T)(vmean);
        output[offset + 1 * C] = (T)(vmean * (vsqrt - 14.f) * 0.1f);
        output[offset + 2 * C] = (T)(vmax);
    }
}

template <typename T>
void global_pooling(T *output, T *input, const T *mask, const T *sqrt_mask,
                    int batch, int channels, int spatial, cudaStream_t stream) {
    const int total_elements = batch * channels;
    const int block_size = KBLOCKSIZE;
    const int blocks = DivUp(total_elements, block_size);

    global_pooling_kernel<<<blocks, block_size, 0, stream>>>(
        output, input, mask, sqrt_mask, batch, channels, spatial);

    ReportCUDAErrors(cudaGetLastError());
}

template <typename T>
__global__ void head_global_pooling_kernel(T *output, T *input,
                                           const T *sqrt_mask,
                                           int N, int C, int spatial) {
    int total_elements = N * C;
    int index = threadIdx.x + blockDim.x * blockIdx.x; // index = [0 ~ batch * channels]
    if (index < total_elements) {
        int n = index / C;
        int c = index % C;

        T *input_ptr = input + index * spatial;
        float vsum = 0;

        float vsqrt = sqrt((float)spatial);
        if (sqrt_mask) {
            vsqrt = (float)(sqrt_mask[n]);
        }

        #pragma unroll
        for (int i = 0; i < spatial; ++i) {
            vsum += (float)(input_ptr[i]);
        }

        float vmean = vsum / (vsqrt * vsqrt);

        int offset = c + n * 3 * C;

        output[offset + 0 * C] = (T)(vmean);
        output[offset + 1 * C] = (T)(vmean * (vsqrt - 14.f) * 0.1f);
        output[offset + 2 * C] = (T)(vmean * ((vsqrt - 14.f) * (vsqrt - 14.f) * 0.01f - 0.1f));
    }
}

template <typename T>
void head_global_pooling(T *output, T *input, const T *sqrt_mask,
                         int batch, int channels, int spatial, cudaStream_t stream) {
    const int total_elements = batch * channels;
    const int block_size = KBLOCKSIZE;
    const int blocks = DivUp(total_elements, block_size);

    head_global_pooling_kernel<<<blocks, block_size, 0, stream>>>(
        output, input, sqrt_mask, batch, channels, spatial);

    ReportCUDAErrors(cudaGetLastError());
}

template <typename T>
__global__ void se_scale_kernel(T *output,
                                const T *input,
                                const T *residual,
                                const T *se_biases,
                                const T *mask,
                                int N, int C, int spatial,
                                bool relu) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int total_elements = N * C * spatial;
    if (index < total_elements) {
        int c = (index / spatial) % C;
        int n = (index / spatial) / C;
        int s = index % spatial;
        int start_idx = n * 2 * C;

        float val = (float)(input[index]);
        float gamma = (float)(se_biases[start_idx + c]);
        gamma = 1.0f / (1.0f + exp(-gamma));

        float beta = (float)(se_biases[start_idx + c + C]);
        float res = 0.0f;
        if (residual) {
            res = (float)(residual[index]);
        }

        val = gamma * val + beta + res;

        if (mask) {
            val *= (float)(mask[n * spatial + s]);
        }
        if (relu && val < 0) {
            val = 0;
        }
        output[index] = (T)val;
    }
}

template <typename T>
void se_scale(T *output, const T *input, const T *residual,
              const T *se_biases, const T *mask, int batch,
              int channels, int spatial, bool relu, cudaStream_t stream) {
    const int total_elements = channels * spatial * batch;
    const int block_size = KBLOCKSIZE;
    const int blocks = DivUp(total_elements, block_size);

    se_scale_kernel<<<blocks, block_size, 0, stream>>>(
        output, input, residual, se_biases, mask, batch, channels, spatial, relu);

    ReportCUDAErrors(cudaGetLastError());
}

template <typename T>
__global__ void channel_pooling_kernel(T *output, T *input,
                                       const T *sqrt_mask,
                                       int N, int C, int spatial) {
    int total_elements = N * spatial;
    int index = threadIdx.x + blockDim.x * blockIdx.x; // index = [0 ~ batch * spatial]
    if (index < total_elements) {
        int n = index / spatial;
        int s = index % spatial;

        T *input_ptr = input + n * C * spatial + s;
        float vsum = 0;
        float vmax = 0.0f;

        float vsqrt = sqrt((float)spatial);
        if (sqrt_mask) {
            vsqrt = (float)(sqrt_mask[n]);
        }

        #pragma unroll
        for (int i = 0; i < C; ++i) {
            float val = (float)(input_ptr[i * spatial]);

            vsum += val;
            if (val > vmax) {
                vmax = val;
            }
        }

        float vmean = vsum / C;
        int offset = s + n * 3 * spatial;

        output[offset + 0 * spatial] = (T)(vmean);
        output[offset + 1 * spatial] = (T)(vmean * (vsqrt - 14.f) * 0.1f);
        output[offset + 2 * spatial] = (T)(vmax);
    }
}

template <typename T>
void channel_pooling(T *output, T *input, const T *sqrt_mask,
                     int batch, int channels, int spatial, cudaStream_t stream) {
    const int total_elements = batch * spatial;
    const int block_size = KBLOCKSIZE;
    const int blocks = DivUp(total_elements, block_size);

    channel_pooling_kernel<<<blocks, block_size, 0, stream>>>(
        output, input, sqrt_mask, batch, channels, spatial);

    ReportCUDAErrors(cudaGetLastError());
}

template <typename T>
__global__ void sa_scale_kernel(T *output,
                                const T *input,
                                const T *residual,
                                const T *sa_biases,
                                int N, int C, int spatial,
                                bool relu) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int total_elements = N * C * spatial;
    if (index < total_elements) {
        int n = (index / spatial) / C;
        int s = index % spatial;
        int start_idx = n * spatial;

        float val = (float)(input[index]);
        float gamma = (float)(sa_biases[start_idx + s]);
        gamma = 1.0f / (1.0f + exp(-gamma));

        float res = 0.0f;
        if (residual) {
            res = (float)(residual[index]);
        }

        val = gamma * val + res;
        if (relu && val < 0) {
            val = 0;
        }
        output[index] = (T)val;
    }
}

template <typename T>
void sa_scale(T *output, const T *input, const T *residual,
              const T *sa_biases, int batch, int channels,
              int spatial, bool relu, cudaStream_t stream) {
    const int total_elements = channels * spatial * batch;
    const int block_size = KBLOCKSIZE;
    const int blocks = DivUp(total_elements, block_size);

    sa_scale_kernel<<<blocks, block_size, 0, stream>>>(
        output, input, residual, sa_biases, batch, channels, spatial, relu);

    ReportCUDAErrors(cudaGetLastError());
}

__device__ __forceinline__ void multiply_bt(
    float * o0, float * o1, float * o2, float * o3, float * o4, float * o5,
    float i0,   float i1,   float i2,   float i3,   float i4,   float i5
) {
    float i3m1 = i1 * -kSqrt2 + i3 * (kSqrt2 / 2.0f);
    float i4m2 = i2 * -2.0f + i4 * 1.0f;

    *o0 = i0 + i2 * (-5.0f/2.0f) + i4;
    *o1 = i3m1 + i4m2;
    *o2 = -i3m1 + i4m2;

    float i3m1_2 = i3 * (kSqrt2) + i1 * (-kSqrt2/2.0f);
    float i4m2_2 = i2 * (-1.0f/2.0f) + i4;

    *o3 = i3m1_2 + i4m2_2;
    *o4 = -i3m1_2 + i4m2_2;

    *o5 = i1 + i3 * (-5.0f/2.0f) + i5;
}

__device__ __forceinline__ void multiply_atv(
    float * o,
    float i0, float i1, float i2, float i3, float i4, float i5
) {
    float t1p2 = (i1 + i2) * (1.0f / 2.0f);
    float t1m2 = (i1 - i2) * (kSqrt2/4.0f);
    float t3p4 = i3 + i4;
    float t3m4 = (i3 - i4) * (kSqrt2);

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

// This kernel is imported from Leela Zero.
template <typename T>
__global__ void transform_in_kernel(T *V, const T *in,
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
                    x[j][i] = (float)(in[batch * C * spatial + 
                                             ch * spatial + b * W + a]);
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
                V[(i*kWinogradAlpha + j) * CPpad + offset] = (T)(T2[i][j]);
            }
        }
    }
}

// This kernel is imported from Leela Zero.
template <typename T>
__global__ void transform_out_kernel(T *Y, const T *M,
                                     const T *biases,
                                     const T *residual,
                                     const T *mask,
                                     const int K,
                                     const int k_pad, const int p_pad,
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
            bias = (float)(biases[k]);
        }

        // M dimensions are [36, outputs, batch_size * tiles].
        // Plus zero padding from SGEMM.

        // This 'offset' is for GEMM batched.
        // const int offset = block * k_pad + k;

        // This 'offset' is for GEMM strided batched.
        const int offset = k * p_pad + block;

        // Calculates transpose(A).temp_m
        #pragma unroll
        for (int xn = 0; xn < kWinogradAlpha; xn++) {
            float temp_m0 = (float)(M[(0 * kWinogradAlpha + xn) * k_pad * p_pad + offset]);
            float temp_m1 = (float)(M[(1 * kWinogradAlpha + xn) * k_pad * p_pad + offset]);
            float temp_m2 = (float)(M[(2 * kWinogradAlpha + xn) * k_pad * p_pad + offset]);
            float temp_m3 = (float)(M[(3 * kWinogradAlpha + xn) * k_pad * p_pad + offset]);
            float temp_m4 = (float)(M[(4 * kWinogradAlpha + xn) * k_pad * p_pad + offset]);
            float temp_m5 = (float)(M[(5 * kWinogradAlpha + xn) * k_pad * p_pad + offset]);
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
                    float val = o[i][j];
                    if (residual) {
                        val += (float)(residual[out_idx]);
                    }
                    if (mask) {
                        int spatial = board_size * board_size;
                        int s_index = out_idx % spatial;
                        val *= (float)(mask[batch * spatial + s_index]);
                    }
                    if (relu && val < 0) {
                        val = 0.f;
                    }
                    Y[out_idx] = (T)val;
                }
            }
        }
    }
}

template <typename T>
void winograd3_transform_in(T *V, const T *in, int batch,
                            int channels, int board_size, cudaStream_t stream) {
    const int ptiles = GetWinogradP(board_size);
    const int total_elements = channels * batch * ptiles;

    const int block_size = KBLOCKSIZE;
    const int blocks = DivUp(total_elements, block_size);

    const int c_pad = channels;
    const int p_pad = batch * ptiles;

    transform_in_kernel<<<blocks, block_size, 0, stream>>>(
        V, in, channels, c_pad, p_pad, board_size, batch);

    ReportCUDAErrors(cudaGetLastError());
}

template <typename T>
void winograd3_transform_out(T *out, const T *M, const T *biases,
                             const T *residual, const T *mask,
                             int batch, int channels, int board_size,
                             bool relu, cudaStream_t stream) {
    const int ptiles = GetWinogradP(board_size);
    const int total_elements = channels * batch * ptiles;

    const int block_size = KBLOCKSIZE;
    const int blocks = DivUp(total_elements, block_size);

    const int k_pad = channels;
    const int p_pad = batch * ptiles;

    transform_out_kernel<<<blocks, block_size, 0, stream>>>(
        out, M, biases, residual, mask, channels,
        k_pad, p_pad, board_size, batch, relu);

    ReportCUDAErrors(cudaGetLastError());
}

template<>
void gemm<float>(bool TA, bool TB, int M, int N, int K, float ALPHA,
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

template<>
void gemm_strided_batched<float>(bool TA, bool TB, int M, int N, int K, float ALPHA,
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

#ifdef ENABLE_FP16
template<>
void gemm<half>(bool TA, bool TB, int M, int N, int K, half ALPHA,
                const half *A_gpu, int lda, const half *B_gpu, int ldb,
                half BETA, half *C_gpu, int ldc, cublasHandle_t handle, cudaStream_t stream) {
    ReportCUBLASErrors(cublasSetStream(handle, stream));
    ReportCUBLASErrors(cublasHgemm(
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

template<>
void gemm_strided_batched<half>(bool TA, bool TB, int M, int N, int K, half ALPHA,
                                const half *A_gpu, int lda, int strideA, const half *B_gpu, int ldb, int strideB,
                                half BETA, half *C_gpu, int ldc, int strideC, int batchsize, cublasHandle_t handle, cudaStream_t stream) {
    ReportCUBLASErrors(cublasSetStream(handle, stream));
    ReportCUBLASErrors(cublasHgemmStridedBatched(
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
#endif

template void add_vectors<float>(float* c, float* a, float* b, int size,
                                 int asize, int bsize, bool relu, cudaStream_t stream);

template void add_spatial<float>(float *data, const float *biases,
                                 const float *residual, const float *mask,
                                 int bsize, int batch, int channels, int spatial,
                                 bool relu, cudaStream_t stream);

template void im2col_batched<float>(float *data_col, float *data_im,
                                    int filter_size, int N, int C, int H, int W,
                                    cudaStream_t stream);

template void global_pooling<float>(float *output, float *input, const float *mask,
                                    const float *sqrt_mask, int batch, int channels,
                                    int spatial, cudaStream_t stream);

template void head_global_pooling<float>(float *output, float *input, const float *sqrt_mask,
                                         int batch, int channels, int spatial, cudaStream_t stream);

template void se_scale<float>(float *output, const float *input,
                              const float *residual, const float *se_biases,
                              const float *mask, int batch, int channels,
                              int spatial, bool relu, cudaStream_t stream);

template void channel_pooling<float>(float *output, float *input, const float *sqrt_mask,
                                     int batch, int channels, int spatial, cudaStream_t stream);

template void sa_scale<float>(float *output, const float *input, const float *residual,
                              const float *sa_biases, int batch, int channels, int spatial,
                              bool relu, cudaStream_t stream);

template void winograd3_transform_in<float>(float *V, const float *in, int batch,
                                            int channels, int board_size, cudaStream_t stream);

template void winograd3_transform_out<float>(float *out, const float *M, const float *biases,
                                             const float *residual, const float *mask,
                                             int batch, int channels, int board_size,
                                             bool relu, cudaStream_t stream);

#ifdef ENABLE_FP16
template void add_vectors<half>(half *c, half *a, half *b, int size, int asize,
                                int bsize,  bool relu, cudaStream_t stream);

template void add_spatial<half>(half *data, const half *biases,
                                const half *residual, const half *mask,
                                int bsize, int batch, int channels, int spatial,
                                bool relu, cudaStream_t stream);

template void im2col_batched<half>(half *data_col, half *data_im,
                                   int filter_size, int N, int C, int H, int W,
                                   cudaStream_t stream);

template void global_pooling<half>(half *output, half *input, const half *mask,
                                   const half *sqrt_mask, int batch, int channels,
                                   int spatial, cudaStream_t stream);

template void head_global_pooling<half>(half *output, half *input, const half *sqrt_mask,
                                        int batch, int channels, int spatial, cudaStream_t stream);

template void se_scale<half>(half *output, const half *input,
                             const half *residual, const half *se_biases,
                             const half *mask, int batch, int channels,
                             int spatial, bool relu, cudaStream_t stream);

template void channel_pooling<half>(half *output, half *input, const half *sqrt_mask,
                                    int batch, int channels, int spatial, cudaStream_t stream);

template void sa_scale<half>(half *output, const half *input, const half *residual,
                             const half *sa_biases, int batch, int channels, int spatial,
                             bool relu, cudaStream_t stream);

template void winograd3_transform_in<half>(half *V, const half *in, int batch,
                                           int channels, int board_size, cudaStream_t stream);

template void winograd3_transform_out<half>(half *out, const half *M, const half *biases,
                                            const half *residual, const half *mask,
                                            int batch, int channels, int board_size,
                                            bool relu, cudaStream_t stream);
#endif

} // namespace cuda
#endif
