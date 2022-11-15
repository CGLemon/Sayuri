#include "neural/cuda/cuda_layers.h"
#include "neural/cuda/cuda_kernels.h"
#include "neural/winograd_helper.h"
#include "neural/blas/winograd_convolution3.h"

#include <cassert>
#include <iostream>
#include <algorithm> 
#ifdef USE_CUDA

namespace CUDA {

Batchnorm::Batchnorm(CudaHandles *handles,
                     const int max_batch,
                     const size_t board_size,
                     const size_t output_channels,
                     bool ReLU) {
    width_ = board_size;
    height_ = board_size;
    spatial_size_ = width_ * height_;

    channels_ = output_channels;
    maxbatch_ = max_batch;
    relu_ = ReLU;
    loaded_ = false;
    handles_ = handles;
}

Batchnorm::~Batchnorm() {
    if (loaded_) {
        ReportCUDAErrors(cudaFree(cuda_means_));
        ReportCUDAErrors(cudaFree(cuda_stddevs_));
    }
}

void Batchnorm::Forward(const int batch,
                        float *data,
                        const float *const eltwise) {
    if (!loaded_) {
        return;
    }

    assert(batch <= maxbatch_);
    batchnorm(data, cuda_means_, cuda_stddevs_,
              batch, channels_, spatial_size_, eltwise, relu_, handles_->stream);
}


void Batchnorm::LoadingWeight(const std::vector<float> &means,
                              const std::vector<float> &stddevs) {
    if (loaded_) {
        return;
    }

    const size_t weights_size = sizeof(float) * channels_;
    assert(weights_size == sizeof(float) * means.size() &&
               weights_size == sizeof(float) * stddevs.size());

    ReportCUDAErrors(cudaMalloc(&cuda_means_, weights_size));
    ReportCUDAErrors(cudaMalloc(&cuda_stddevs_, weights_size));

    // Push the weights.
    ReportCUDAErrors(cudaMemcpy(cuda_means_, means.data(),
                                    weights_size,
                                    cudaMemcpyHostToDevice));

    ReportCUDAErrors(cudaMemcpy(cuda_stddevs_, stddevs.data(),
                                    weights_size,
                                    cudaMemcpyHostToDevice));
    loaded_ = true;
}

Convolution::Convolution(CudaHandles *handles,
                         const int max_batch,
                         const size_t board_size, 
                         const size_t filter_size,
                         const size_t input_channels,
                         const size_t output_channels) {
    width_ = board_size;
    height_ = board_size;
    board_size_ = board_size;
    spatial_size_ = width_ * height_;

    in_channels_ = input_channels;
    out_channels_ = output_channels;
    filters_ = filter_size;
    filter_dim_ = filters_ * filters_ * in_channels_;
    maxbatch_ = max_batch;

    handles_ = handles;

    loaded_ = false;
}

Convolution::~Convolution() {
    if (loaded_) {
        ReportCUDAErrors(cudaFree(cuda_weights_));

#ifdef USE_CUDNN
        cudnnDestroyFilterDescriptor(filter_desc_);
        cudnnDestroyConvolutionDescriptor(conv_desc_);
        cudnnDestroyTensorDescriptor(in_tensor_desc_);
        cudnnDestroyTensorDescriptor(out_tensor_desc_);
        if (cuda_biases_) {
            cudnnDestroyTensorDescriptor(bias_desc_);
        }

#endif

        if (cuda_biases_) {
            ReportCUDAErrors(cudaFree(cuda_biases_));
        }
    }
}

void Convolution::Forward(const int batch, float *input, float *output,
                          void *scratch, void *scratch_other, size_t scratch_size) {
    if (!loaded_) {
        return;
    }
    if (!scratch || !scratch_other) {
        return;
    }

    assert(batch <= maxbatch_);
#ifdef USE_CUDNN
    ReportCUDNNErrors(cudnnSetStream(handles_->cudnn_handle, handles_->stream));
    ReportCUDNNErrors(cudnnSetTensor4dDescriptor(in_tensor_desc_,
                                                 CUDNN_TENSOR_NCHW,
                                                 CUDNN_DATA_FLOAT,
                                                 batch, in_channels_, height_, width_));
    ReportCUDNNErrors(cudnnSetTensor4dDescriptor(out_tensor_desc_,
                                                 CUDNN_TENSOR_NCHW,
                                                 CUDNN_DATA_FLOAT,
                                                 batch, out_channels_, height_, width_));
  
    static constexpr float alpha = 1.0f, beta = 0.0f;
    ReportCUDNNErrors(cudnnConvolutionForward(
                      handles_->cudnn_handle, &alpha, in_tensor_desc_, input, filter_desc_, cuda_weights_,
                      conv_desc_, conv_algo_, scratch, scratch_size, &beta, out_tensor_desc_,
                      output));


    if (cuda_biases_) {
        ReportCUDNNErrors(cudnnAddTensor(handles_->cudnn_handle, &alpha, bias_desc_, cuda_biases_,
                                         &alpha, out_tensor_desc_, output));
    }

#else
    (void) scratch_size;
    auto scratch_op = reinterpret_cast<float*>(scratch);

    if (winograd_) {
        // TODO: Merge batch norm layer with Winograd.
        auto scratch_op_other = reinterpret_cast<float*>(scratch_other);
        const int batch_ptiles = batch * GetWinogradP(board_size_);

        winograd3_transform_in(
            input, scratch_op,
            batch, in_channels_, board_size_, handles_->stream);

        gemm_strided_batched(
            true, false,
            out_channels_, batch_ptiles, in_channels_,
            1.0f,
            cuda_weights_, out_channels_, in_channels_ * out_channels_,
            scratch_op, batch_ptiles, in_channels_ * batch_ptiles,
            0.0f,
            scratch_op_other, batch_ptiles, out_channels_ * batch_ptiles,
            kWinogradTile,
            handles_->cublas_handle, handles_->stream);

        winograd3_transform_out(
            scratch_op_other, output,
            batch, out_channels_, board_size_, handles_->stream);
    } else {
#ifdef USE_SINGLE_BATCH_CONV
        // We remain this code for debugging.
        const size_t input_shift = in_channels_ * spatial_size_;
        const size_t output_shift = out_channels_ * spatial_size_;
        for (int b = 0; b < batch; ++b) {
            float *input_ptr = input + b * input_shift;
            float *output_ptr = output + b * output_shift;
            if (filters_ != 1) {
                im2col(filters_, in_channels_, height_, width_, input_ptr, scratch_op, handles_->stream);
                gemm(false, false, out_channels_, spatial_size_, filter_dim_, 1.0f,
                     cuda_weights_, filter_dim_, scratch_op, spatial_size_,
                     0.0f, output_ptr, spatial_size_, handles_->cublas_handle, handles_->stream);
            } else {
                gemm(false, false, out_channels_, spatial_size_, filter_dim_, 1.0f,
                     cuda_weights_, filter_dim_, input_ptr, spatial_size_,
                     0.0f, output_ptr, spatial_size_, handles_->cublas_handle, handles_->stream);
            }
        }
#else
        if (filters_ != 1) {
            im2col_batched(filters_, batch, in_channels_, height_, width_, input, scratch_op, handles_->stream);
            gemm_strided_batched(
                false, false,
                out_channels_, spatial_size_, filter_dim_,
                1.0f,
                cuda_weights_, filter_dim_, 0,
                scratch_op, spatial_size_, filter_dim_ * spatial_size_,
                0.f,
                output, spatial_size_, out_channels_ * spatial_size_,
                batch,
                handles_->cublas_handle, handles_->stream);
        } else {
            gemm_strided_batched(
                false, false,
                out_channels_, spatial_size_, filter_dim_,
                1.0f,
                cuda_weights_, filter_dim_, 0,
                input, spatial_size_, in_channels_ * spatial_size_,
                0.f,
                output, spatial_size_, out_channels_ * spatial_size_,
                batch,
                handles_->cublas_handle, handles_->stream);
        }
#endif
    }
    if (cuda_biases_) {
        const auto op_size = out_channels_ * spatial_size_;
        add_spatial(output, cuda_biases_, output,
                    op_size * batch, out_channels_, op_size * batch,
                    spatial_size_, false, handles_->stream);
    }
#endif
}

void Convolution::LoadingWeight(const std::vector<float> &weights,
                                size_t &scratch_size, bool winograd) {
    if (loaded_) {
        return;
    }
    assert((int)weights.size() == filter_dim_ * out_channels_);

    if (filters_ == 3 && winograd) {
        winograd_ = true;
    } else {
        winograd_ = false;
    }

#ifdef USE_CUDNN
    // TODO: Use the Winograd with cuDNN.
    winograd_ = false;
#endif
    std::vector<float> weights_copy = weights;

    if (winograd_) {
        weights_copy = WinogradTransformF(
                           weights_copy, out_channels_, in_channels_);
    }

    const size_t weights_size = sizeof(float) * weights_copy.size();

    ReportCUDAErrors(cudaMalloc(&cuda_weights_, weights_size));
    ReportCUDAErrors(cudaMemcpy(cuda_weights_, weights_copy.data(), weights_size,
                                    cudaMemcpyHostToDevice));

    loaded_ = true;
    size_t apply_scratch_size = 0;

#ifdef USE_CUDNN
    cudnnCreateFilterDescriptor(&filter_desc_);
    cudnnCreateTensorDescriptor(&in_tensor_desc_);
    cudnnCreateTensorDescriptor(&out_tensor_desc_);

    cudnnCreateConvolutionDescriptor(&conv_desc_);

    ReportCUDNNErrors(cudnnSetFilter4dDescriptor(filter_desc_, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                                 out_channels_, in_channels_, filters_, filters_));
  
    const size_t padding = filters_ / 2;
    ReportCUDNNErrors(cudnnSetConvolution2dDescriptor(
                      conv_desc_, padding, padding, 1, 1, 1, 1,
                      CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    ReportCUDNNErrors(cudnnSetTensor4dDescriptor(in_tensor_desc_,
                                                 CUDNN_TENSOR_NCHW,
                                                 CUDNN_DATA_FLOAT,
                                                 maxbatch_, in_channels_, height_, width_));

    ReportCUDNNErrors(cudnnSetTensor4dDescriptor(out_tensor_desc_,
                                                 CUDNN_TENSOR_NCHW,
                                                 CUDNN_DATA_FLOAT,
                                                 maxbatch_, out_channels_, height_, width_));

    ReportCUDNNErrors(cudnnSetConvolutionMathType(conv_desc_, CUDNN_DEFAULT_MATH));

#if CUDNN_MAJOR >= 8
    cudnnConvolutionFwdAlgoPerf_t  conv_perf;
    int returned_cnt;
    ReportCUDNNErrors(cudnnFindConvolutionForwardAlgorithm(handles_->cudnn_handle,
                                                           in_tensor_desc_,
                                                           filter_desc_,
                                                           conv_desc_,
                                                           out_tensor_desc_,
                                                           1,
                                                           &returned_cnt,
                                                           &conv_perf));
    conv_algo_ = conv_perf.algo;
#else
    ReportCUDNNErrors(cudnnGetConvolutionForwardAlgorithm(handles_->cudnn_handle,
                                                          in_tensor_desc_,
                                                          filter_desc_,
                                                          conv_desc_,
                                                          out_tensor_desc_,
                                                          CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                          0,
                                                          &conv_algo_));
#endif
    ReportCUDNNErrors(cudnnGetConvolutionForwardWorkspaceSize(handles_->cudnn_handle,
                                                              in_tensor_desc_,
                                                              filter_desc_,
                                                              conv_desc_,
                                                              out_tensor_desc_,
                                                              conv_algo_,
                                                              &apply_scratch_size));

    scratch_size = std::max(apply_scratch_size, scratch_size);
#else
    // TODO: Seem there are overflow bug in the convolution
    //       Kernel. We set a greater buffer size to avoid this 
    //       bug. Need to fix it later.
    const int overflow_factor = 16;
    int scratch_size_base = 0;

    if (winograd_) {
        scratch_size_base = kWinogradTile *
                                GetWinogradP(board_size_) *
                                std::max(out_channels_, in_channels_);
    } else {
        scratch_size_base = filter_dim_ * spatial_size_;
    }
    apply_scratch_size = maxbatch_ * scratch_size_base * overflow_factor;
    scratch_size = std::max(apply_scratch_size, scratch_size);
#endif
}


void Convolution::LoadingWeight(const std::vector<float> &weights,
                                const std::vector<float> &biases,
                                size_t &scratch_size, bool winpgrad) {
    if (loaded_) {
        return;
    }
    const size_t biases_size = sizeof(float) * biases.size();
    assert((int)biases.size() == out_channels_);

    ReportCUDAErrors(cudaMalloc(&cuda_biases_, biases_size));
    ReportCUDAErrors(cudaMemcpy(cuda_biases_, biases.data(), biases_size,
                                cudaMemcpyHostToDevice));

#ifdef USE_CUDNN
    cudnnCreateTensorDescriptor(&bias_desc_);
    ReportCUDNNErrors(cudnnSetTensor4dDescriptor(bias_desc_,
                                                 CUDNN_TENSOR_NCHW,
                                                 CUDNN_DATA_FLOAT,
                                                 1, out_channels_, 1, 1));
#endif
    LoadingWeight(weights, scratch_size, winpgrad);
}


FullyConnect::FullyConnect(CudaHandles *handles,
                           const int max_batch, const size_t inputs, 
                           const size_t outputs, bool ReLU) {
    maxbatch_ = max_batch;
    inputs_ = inputs;
    outputs_ = outputs;
    loaded_ = false;
    relu_ = ReLU;
    handles_ = handles;
}

FullyConnect::~FullyConnect() {
    if (loaded_) {
        ReportCUDAErrors(cudaFree(cuda_weights_));
        ReportCUDAErrors(cudaFree(cuda_biases_));
    }
}

void FullyConnect::LoadingWeight(const std::vector<float> &weights,
                                 const std::vector<float> &biases) {
    if (loaded_) { 
        return;
    }
    const size_t weights_size = sizeof(float) * weights.size();
    const size_t biases_size = sizeof(float) * biases.size();

    assert((int)weights.size() == inputs_ * outputs_);
    assert((int)biases.size() == outputs_);

    ReportCUDAErrors(cudaMalloc(&cuda_weights_, weights_size));
    ReportCUDAErrors(cudaMalloc(&cuda_biases_, biases_size));
  
    ReportCUDAErrors(cudaMemcpy(
        cuda_weights_, weights.data(), weights_size, cudaMemcpyHostToDevice));
    ReportCUDAErrors(cudaMemcpy(
        cuda_biases_, biases.data(), biases_size, cudaMemcpyHostToDevice));
    loaded_ = true;
}

void FullyConnect::Forward(const int batch, float *input, float *output) {
    if (!loaded_) {
        return;
    }
    assert(batch <= maxbatch_);
    gemm(false, true,
         batch,
         outputs_,
         inputs_,
         1.0f,
         input,
         inputs_, 
         cuda_weights_,
         inputs_,
         0.0f,
         output,
         outputs_,
         handles_->cublas_handle,
         handles_->stream);

    add_vectors(output, cuda_biases_, output,
                outputs_ * batch, outputs_, outputs_ * batch, relu_, handles_->stream );
}

GlobalPooling::GlobalPooling(CudaHandles *handles,
                             bool is_value_head,
                             const int max_batch,
                             const size_t board_size,
                             const size_t channels) {
    width_ = board_size;
    height_ = board_size;
    spatial_size_ = width_ * height_;
    is_value_head_ = is_value_head;

    maxbatch_ = max_batch;
    channels_ = channels;
    handles_ = handles;
}

void GlobalPooling::Forward(const int batch, float *input, float *output) {
    const int board_size = (width_ + height_) / 2;
    const float b_diff = (float)board_size - kAvgBSize;

    if (is_value_head_) {
        const float b_coeff0 = b_diff / 10.f;
        const float b_coeff1 = b_diff * b_diff / 100.f - kBSizeVaraince;

        head_global_pool(input, output, b_coeff0, b_coeff1, batch,
                         channels_, spatial_size_, handles_->stream);
    } else {
        const float b_coeff = b_diff / 10.f;
        global_pool(input, output, b_coeff, batch,
                    channels_, spatial_size_, handles_->stream);
    }
}


SEUnit::SEUnit(CudaHandles *handles, const int max_batch,
               const size_t board_size, const size_t channels, const size_t se_size) {
    width_ = board_size;
    height_ = board_size;
    spatial_size_ = width_ * height_;

    se_size_ = se_size;
    maxbatch_ = max_batch;
    channels_ = channels;
    loaded_ = false;
    handles_ = handles;
}

void SEUnit::LoadingWeight(const std::vector<float> &weights_w1,
                           const std::vector<float> &weights_b1,
                           const std::vector<float> &weights_w2,
                           const std::vector<float> &weights_b2) {
    if (loaded_) { 
        return;
    }
    const size_t type_size = sizeof(float);
    const size_t weights_w1_size = type_size * weights_w1.size();
    const size_t weights_b1_size = type_size * weights_b1.size();
    const size_t weights_w2_size = type_size * weights_w2.size();
    const size_t weights_b2_size = type_size * weights_b2.size();

    assert((int)weights_w1.size() == channels_ * se_size_);
    assert((int)weights_b1.size() == se_size_);
    assert((int)weights_w2.size() == 2 * se_size_  * channels_);
    assert((int)weights_b2.size() == 2 * channels_);

    ReportCUDAErrors(cudaMalloc(&cuda_weights_w1_, weights_w1_size));
    ReportCUDAErrors(cudaMalloc(&cuda_weights_b1_, weights_b1_size));
    ReportCUDAErrors(cudaMalloc(&cuda_weights_w2_, weights_w2_size));
    ReportCUDAErrors(cudaMalloc(&cuda_weights_b2_, weights_b2_size));

    const size_t fc1_scratch_size = type_size * maxbatch_ * se_size_;
    const size_t fc2_scratch_size = type_size * 2 * maxbatch_ * channels_;
    const size_t pool_scratch_size = type_size * maxbatch_ * 3 * channels_;

    ReportCUDAErrors(cudaMalloc(&cuda_op_[0], pool_scratch_size));
    ReportCUDAErrors(cudaMalloc(&cuda_op_[1], fc1_scratch_size));
    ReportCUDAErrors(cudaMalloc(&cuda_op_[2], fc2_scratch_size));

    loaded_ = true;

    ReportCUDAErrors(cudaMemcpy(
        cuda_weights_w1_, weights_w1.data(), weights_w1_size, cudaMemcpyHostToDevice));
    ReportCUDAErrors(cudaMemcpy(
        cuda_weights_b1_, weights_b1.data(), weights_b1_size, cudaMemcpyHostToDevice));
    ReportCUDAErrors(cudaMemcpy(
        cuda_weights_w2_, weights_w2.data(), weights_w2_size, cudaMemcpyHostToDevice));
    ReportCUDAErrors(cudaMemcpy(
        cuda_weights_b2_, weights_b2.data(), weights_b2_size, cudaMemcpyHostToDevice));
}

void SEUnit::Forward(const int batch, float *input, float *ouput) {
    const int board_size = (width_ + height_) / 2;
    const float b_coeff = ((float)board_size - kAvgBSize) / 10.f;
    global_pool(input, cuda_op_[0], b_coeff, batch, channels_, spatial_size_, handles_->stream);

    const size_t fc1_input_size = 3 * channels_;
    const size_t fc1_output_size = se_size_;
    const bool fc1_relu = true;
    gemm(false, true,
         batch,
         fc1_output_size,
         fc1_input_size, 
         1.0f,
         cuda_op_[0],
         fc1_input_size, 
         cuda_weights_w1_,
         fc1_input_size,
         0.0f,
         cuda_op_[1],
         fc1_output_size,
         handles_->cublas_handle,
         handles_->stream);

    add_vectors(cuda_op_[1], cuda_weights_b1_, cuda_op_[1],
                fc1_output_size * batch, fc1_output_size, fc1_output_size * batch, fc1_relu, handles_->stream);

    const size_t fc2_input_size = se_size_;
    const size_t fc2_output_size = 2 * channels_;
    const bool fc2_relu = false;
    gemm(false, true,
         batch,
         fc2_output_size,
         fc2_input_size, 
         1.0f,
         cuda_op_[1],
         fc2_input_size, 
         cuda_weights_w2_,
         fc2_input_size,
         0.0f,
         cuda_op_[2],
         fc2_output_size,
         handles_->cublas_handle,
         handles_->stream);

    add_vectors(cuda_op_[2], cuda_weights_b2_, cuda_op_[2],
                fc2_output_size * batch, fc2_output_size, fc2_output_size * batch, fc2_relu, handles_->stream);

    se_scale(input, cuda_op_[2], ouput, batch, channels_, spatial_size_, handles_->stream);
}

SEUnit::~SEUnit() {
    if (loaded_) {
        ReportCUDAErrors(cudaFree(cuda_weights_w1_));
        ReportCUDAErrors(cudaFree(cuda_weights_b1_));
        ReportCUDAErrors(cudaFree(cuda_weights_w2_));
        ReportCUDAErrors(cudaFree(cuda_weights_b2_));

        ReportCUDAErrors(cudaFree(cuda_op_[0]));
        ReportCUDAErrors(cudaFree(cuda_op_[1]));
        ReportCUDAErrors(cudaFree(cuda_op_[2]));
    }
}

} // namespace CUDA

#endif
