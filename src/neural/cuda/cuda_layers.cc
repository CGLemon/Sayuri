#include "neural/cuda/cuda_layers.h"
#include "neural/cuda/cuda_kernels.h"

#include <cassert>
#include <algorithm> 
#ifdef USE_CUDA

namespace CUDA {

Batchnorm::Batchnorm(const int max_batch,
                     const size_t board_size,
                     const size_t output_channels,
                     bool ReLU) {
    width = board_size;
    height = board_size;
    spatial_size = width * height;

    m_channels = output_channels;
    m_maxbatch = max_batch;
    m_ReLU = ReLU;
    is_loaded = false;
}

Batchnorm::~Batchnorm() {
    if (is_loaded) {
        ReportCUDAErrors(cudaFree(cuda_means));
        ReportCUDAErrors(cudaFree(cuda_stddevs));
    }
}

void Batchnorm::Forward(const int batch,
                        float *data,
                        const float *const eltwise) {
    if (!is_loaded) {
        return;
    }

    assert(batch <= m_maxbatch);
    batchnorm(data, cuda_means, cuda_stddevs,
              batch, m_channels, spatial_size, eltwise, m_ReLU);
}


void Batchnorm::LoadingWeight(const std::vector<float> &means,
                              const std::vector<float> &stddevs) {
    if (is_loaded) {
        return;
    }

    const size_t weights_size = sizeof(float) * m_channels;
    assert(weights_size == sizeof(float) * means.size() &&
           weights_size == sizeof(float) * stddevs.size());

    ReportCUDAErrors(cudaMalloc(&cuda_means, weights_size));
    ReportCUDAErrors(cudaMalloc(&cuda_stddevs, weights_size));

    ReportCUDAErrors(cudaMemcpy(cuda_means, means.data(), weights_size,
                                cudaMemcpyHostToDevice));
    ReportCUDAErrors(cudaMemcpy(cuda_stddevs, stddevs.data(), weights_size,
                                cudaMemcpyHostToDevice));
    is_loaded = true;
}

Convolution::Convolution(const int max_batch,
                         const size_t board_size, 
                         const size_t filter_size,
                         const size_t input_channels,
                         const size_t output_channels) {
    width = board_size;
    height = board_size;
    spatial_size = width * height;

    m_in_channels = input_channels;
    m_out_channels = output_channels;
    m_filter = filter_size;
    m_filter_dim = m_filter * m_filter * m_in_channels;
    m_maxbatch = max_batch;

#ifdef USE_CUDNN
    cudnn_applied = false;
#endif
    is_loaded = false;
}

Convolution::~Convolution() {
    if (is_loaded) {
        ReportCUDAErrors(cudaFree(cuda_weights));
    }

#ifdef USE_CUDNN
    if (cudnn_applied) {
        cudnnDestroyFilterDescriptor(filter_desc);
        cudnnDestroyConvolutionDescriptor(conv_desc);
        cudnnDestroyTensorDescriptor(in_tensor_desc);
        cudnnDestroyTensorDescriptor(out_tensor_desc);
        if (cuda_biases) {
            cudnnDestroyTensorDescriptor(bias_desc);
        }
    }
#endif

    if (cuda_biases) {
        ReportCUDAErrors(cudaFree(cuda_biases));
    }
}

void Convolution::Forward(const int batch, float *input, float *output,
                          void *scratch, size_t scratch_size, CudaHandel *handel) {
    if (!is_loaded) {
        return;
    }
    assert(batch <= m_maxbatch);
#ifdef USE_CUDNN
    if (!scratch) {
        return;
    }

    ReportCUDNNErrors(cudnnSetTensor4dDescriptor(in_tensor_desc,
                                                 CUDNN_TENSOR_NCHW,
                                                 CUDNN_DATA_FLOAT,
                                                 batch, m_in_channels, height, width));
    ReportCUDNNErrors(cudnnSetTensor4dDescriptor(out_tensor_desc,
                                                 CUDNN_TENSOR_NCHW,
                                                 CUDNN_DATA_FLOAT,
                                                 batch, m_out_channels, height, width));
  
    static constexpr float alpha = 1.0f, beta = 0.0f;
    ReportCUDNNErrors(cudnnConvolutionForward(
                      handel->cudnn_handel, &alpha, in_tensor_desc, input, filter_desc, cuda_weights,
                      conv_desc, conv_algo, scratch, scratch_size, &beta, out_tensor_desc,
                      output));


    if (cuda_biases) {
        ReportCUDNNErrors(cudnnAddTensor(handel->cudnn_handel, &alpha, bias_desc, cuda_biases,
                                         &alpha, out_tensor_desc, output));
    }

#else
    (void) scratch_size;

    auto op_scratch = reinterpret_cast<float*>(scratch);
    const size_t input_shift = m_in_channels * spatial_size;
    const size_t output_shift = m_out_channels * spatial_size;
    for (int b = 0; b < batch; ++b) {
        float *input_ptr = input + b * input_shift;
        float *output_ptr = output + b * output_shift;
        if (m_filter != 1) {
            im2col(m_filter, m_in_channels, height, width, input_ptr, op_scratch);
            gemm(false, false, m_out_channels, spatial_size, m_filter_dim, 1.0f,
                 cuda_weights, m_filter_dim, op_scratch, spatial_size,
                 0.0f, output_ptr, spatial_size, &handel->cublas_handel);
        } else {
            gemm(false, false, m_out_channels, spatial_size, m_filter_dim, 1.0f,
                 cuda_weights, m_filter_dim, input_ptr, spatial_size,
                 0.0f, output_ptr, spatial_size, &handel->cublas_handel);
        }
    }

    if (cuda_biases) {
        const auto op_size = m_out_channels * spatial_size;
        add_spatial(output, cuda_biases, output,
                    op_size * batch, m_out_channels, op_size * batch,
                    spatial_size, false);
    }
#endif
}

void Convolution::LoadingWeight(const std::vector<float> &weights,
                                size_t &scratch_size, CudaHandel *handel) {
    if (is_loaded) {
        return;
    }
    const size_t weights_size = sizeof(float) * weights.size();
    assert((int)weights.size() == m_filter_dim * m_out_channels);

    ReportCUDAErrors(cudaMalloc(&cuda_weights, weights_size));
    ReportCUDAErrors(cudaMemcpy(cuda_weights, weights.data(), weights_size,
                                cudaMemcpyHostToDevice));

    is_loaded = true;
    size_t apply_scratch_size = 0;

#ifdef USE_CUDNN
    if (cudnn_applied) { 
        return;
    }

    auto cudnn = handel->cudnn_handel;
    cudnnCreateFilterDescriptor(&filter_desc);
    cudnnCreateTensorDescriptor(&in_tensor_desc);
    cudnnCreateTensorDescriptor(&out_tensor_desc);

    cudnnCreateConvolutionDescriptor(&conv_desc);

    ReportCUDNNErrors(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                               m_out_channels, m_in_channels, m_filter, m_filter));
  
    const size_t padding = m_filter / 2;
    ReportCUDNNErrors(cudnnSetConvolution2dDescriptor(
                      conv_desc, padding, padding, 1, 1, 1, 1,
                      CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    ReportCUDNNErrors(cudnnSetTensor4dDescriptor(in_tensor_desc,
                                                 CUDNN_TENSOR_NCHW,
                                                 CUDNN_DATA_FLOAT,
                                                 1, m_in_channels, height, width));

    ReportCUDNNErrors(cudnnSetTensor4dDescriptor(out_tensor_desc,
                                                 CUDNN_TENSOR_NCHW,
                                                 CUDNN_DATA_FLOAT,
                                                 1, m_out_channels, height, width));

    ReportCUDNNErrors(cudnnGetConvolutionForwardAlgorithm(cudnn,
                                                          in_tensor_desc,
                                                          filter_desc, 
                                                          conv_desc,
                                                          out_tensor_desc, 
                                                          CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                          0,
                                                          &conv_algo));

    ReportCUDNNErrors(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                              in_tensor_desc,
                                                              filter_desc, 
                                                              conv_desc,
                                                              out_tensor_desc, 
                                                              conv_algo, 
                                                              &apply_scratch_size));

    cudnn_applied = true;
    const size_t max_scratch_size = std::max(apply_scratch_size, scratch_size);
    scratch_size = max_scratch_size;
#else
    apply_scratch_size = weights_size * m_filter * m_filter;
    const size_t max_scratch_size = std::max(apply_scratch_size, scratch_size);
    scratch_size = max_scratch_size;
#endif
}


void Convolution::LoadingWeight(const std::vector<float> &weights,
                                const std::vector<float> &biases,
                                size_t &scratch_size, CudaHandel *handel) {
    if (is_loaded) {
        return;
    }
    const size_t biases_size = sizeof(float) * biases.size();
    assert((int)biases.size() == m_out_channels);

    ReportCUDAErrors(cudaMalloc(&cuda_biases, biases_size));
    ReportCUDAErrors(cudaMemcpy(cuda_biases, biases.data(), biases_size,
                                cudaMemcpyHostToDevice));

#ifdef USE_CUDNN
    cudnnCreateTensorDescriptor(&bias_desc);
    ReportCUDNNErrors(cudnnSetTensor4dDescriptor(bias_desc,
                                                 CUDNN_TENSOR_NCHW,
                                                 CUDNN_DATA_FLOAT,
                                                 1, m_out_channels, 1, 1));
#endif
    LoadingWeight(weights, scratch_size, handel);
}


FullyConnect::FullyConnect(const int max_batch, const size_t inputs, 
                           const size_t outputs, bool ReLU) {
    m_maxbatch = max_batch;
    m_inputs = inputs;
    m_outputs = outputs;
    is_loaded = false;
    m_ReLU = ReLU;
}

FullyConnect::~FullyConnect() {
    if (is_loaded) {
        ReportCUDAErrors(cudaFree(cuda_weights));
        ReportCUDAErrors(cudaFree(cuda_biases));
    }
}

void FullyConnect::LoadingWeight(const std::vector<float> &weights,
                                 const std::vector<float> &biases) {
    if (is_loaded) { 
        return;
    }
    const size_t weights_size = sizeof(float) * weights.size();
    const size_t biases_size = sizeof(float) * biases.size();

    assert((int)weights.size() == m_inputs * m_outputs);
    assert((int)biases.size() == m_outputs);

    ReportCUDAErrors(cudaMalloc(&cuda_weights, weights_size));
    ReportCUDAErrors(cudaMalloc(&cuda_biases, biases_size));
  
    ReportCUDAErrors(cudaMemcpy(
        cuda_weights, weights.data(), weights_size, cudaMemcpyHostToDevice));
    ReportCUDAErrors(cudaMemcpy(
        cuda_biases, biases.data(), biases_size, cudaMemcpyHostToDevice));
    is_loaded = true;
}

void FullyConnect::Forward(const int batch, float *input, float *output, CudaHandel *handel) {
    if (!is_loaded) {
        return;
    }
    assert(batch <= m_maxbatch);
    gemm(false, true,
         batch,
         m_outputs,
         m_inputs,
         1.0f,
         input,
         m_inputs, 
         cuda_weights,
         m_inputs,
         0.0f,
         output,
         m_outputs,
         &handel->cublas_handel);

    add_vectors(output, cuda_biases, output,
                m_outputs * batch, m_outputs, m_outputs * batch, m_ReLU);
}

GlobalAvgPool::GlobalAvgPool(const int max_batch,
                             const size_t board_size,
                             const size_t channels) {
    width = board_size;
    height = board_size;
    spatial_size = width * height;

    m_maxbatch = max_batch;
    m_channels = channels;
}

void GlobalAvgPool::Forward(const int batch, float *input, float *output) {
    global_avg_pool(input, output, batch,
                    m_channels, spatial_size);
}

SEUnit::SEUnit(const int max_batch, const size_t board_size, const size_t channels, const size_t se_size) {
    width = board_size;
    height = board_size;
    spatial_size = width * height;

    m_se_size = se_size;
    m_maxbatch = max_batch;
    m_channels = channels;
    is_loaded = false;
}

void SEUnit::LoadingWeight(const std::vector<float> &weights_w1,
                           const std::vector<float> &weights_b1,
                           const std::vector<float> &weights_w2,
                           const std::vector<float> &weights_b2) {
    if (is_loaded) { 
        return;
    }
    const size_t type_size = sizeof(float);
    const size_t weights_w1_size = type_size * weights_w1.size();
    const size_t weights_b1_size = type_size * weights_b1.size();
    const size_t weights_w2_size = type_size * weights_w2.size();
    const size_t weights_b2_size = type_size * weights_b2.size();

    assert((int)weights_w1.size() == m_channels * m_se_size);
    assert((int)weights_b1.size() == m_se_size);
    assert((int)weights_w2.size() == 2 * m_se_size * m_channels);
    assert((int)weights_b2.size() == 2 * m_channels);

    ReportCUDAErrors(cudaMalloc(&cuda_weights_w1, weights_w1_size));
    ReportCUDAErrors(cudaMalloc(&cuda_weights_b1, weights_b1_size));
    ReportCUDAErrors(cudaMalloc(&cuda_weights_w2, weights_w2_size));
    ReportCUDAErrors(cudaMalloc(&cuda_weights_b2, weights_b2_size));

    const size_t fc1_scratch_size = type_size * m_maxbatch * m_se_size;
    const size_t fc2_scratch_size = type_size * 2 * m_maxbatch * m_channels;
    const size_t pool_scratch_size = type_size * m_maxbatch * m_channels;

    ReportCUDAErrors(cudaMalloc(&cuda_op[0], pool_scratch_size));
    ReportCUDAErrors(cudaMalloc(&cuda_op[1], fc1_scratch_size));
    ReportCUDAErrors(cudaMalloc(&cuda_op[2], fc2_scratch_size));

    is_loaded = true;

    ReportCUDAErrors(cudaMemcpy(
        cuda_weights_w1, weights_w1.data(), weights_w1_size, cudaMemcpyHostToDevice));
    ReportCUDAErrors(cudaMemcpy(
        cuda_weights_b1, weights_b1.data(), weights_b1_size, cudaMemcpyHostToDevice));
    ReportCUDAErrors(cudaMemcpy(
        cuda_weights_w2, weights_w2.data(), weights_w2_size, cudaMemcpyHostToDevice));
    ReportCUDAErrors(cudaMemcpy(
        cuda_weights_b2, weights_b2.data(), weights_b2_size, cudaMemcpyHostToDevice));
}

void SEUnit::Forward(const int batch, float *input, float *ouput, CudaHandel *handel) {
    global_avg_pool(input, cuda_op[0], batch, m_channels, spatial_size);

    const size_t fc1_input_size = m_channels;
    const size_t fc1_output_size = m_se_size;
    const bool fc1_relu = true;
    gemm(false, true,
         batch,
         fc1_output_size,
         fc1_input_size, 
         1.0f,
         cuda_op[0],
         fc1_input_size, 
         cuda_weights_w1,
         fc1_input_size,
         0.0f,
         cuda_op[1],
         fc1_output_size,
         &handel->cublas_handel);

    add_vectors(cuda_op[1], cuda_weights_b1, cuda_op[1],
                fc1_output_size * batch, fc1_output_size, fc1_output_size * batch, fc1_relu);

    const size_t fc2_input_size = m_se_size;
    const size_t fc2_output_size = 2 * m_channels;
    const bool fc2_relu = false;
    gemm(false, true,
         batch,
         fc2_output_size,
         fc2_input_size, 
         1.0f,
         cuda_op[1],
         fc2_input_size, 
         cuda_weights_w2,
         fc2_input_size,
         0.0f,
         cuda_op[2],
         fc2_output_size,
         &handel->cublas_handel);

    add_vectors(cuda_op[2], cuda_weights_b2, cuda_op[2],
                fc2_output_size * batch, fc2_output_size, fc2_output_size * batch, fc2_relu);

    se_scale(input, cuda_op[2], ouput, batch, m_channels, spatial_size);
}

SEUnit::~SEUnit() {
    if (is_loaded) {
        ReportCUDAErrors(cudaFree(cuda_weights_w1));
        ReportCUDAErrors(cudaFree(cuda_weights_b1));
        ReportCUDAErrors(cudaFree(cuda_weights_w2));
        ReportCUDAErrors(cudaFree(cuda_weights_b2));

        ReportCUDAErrors(cudaFree(cuda_op[0]));
        ReportCUDAErrors(cudaFree(cuda_op[1]));
        ReportCUDAErrors(cudaFree(cuda_op[2]));
    }
}

} // namespace CUDA

#endif
