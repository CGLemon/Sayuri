#ifdef USE_CUDA

#include "neural/cuda/cuda_layers.h"
#include "neural/cuda/cuda_kernels.h"
#include "neural/winograd_helper.h"
#include "neural/blas/winograd_convolution3.h"
#include "utils/half.h"

#include <cassert>
#include <iostream>
#include <algorithm>

namespace cuda {

void AddVectors(bool fp16, void *c, void *a, void *b,
                int size, int asize, int bsize,
                bool relu, cudaStream_t stream) {
    if (fp16) {
#ifdef ENABLE_FP16
        add_vectors(
            (half *)c, (half *)a, (half *)b,
            size, asize, bsize, relu, stream);
#endif
    } else {
        add_vectors(
            (float *)c, (float *)a, (float *)b,
            size, asize, bsize, relu, stream);
    }
}

void AddSpatial(bool fp16, void *data, const void *biases,
                const void *residual, const void *mask,
                int bsize, int batch, int channels, int spatial,
                bool relu, cudaStream_t stream) {
    if (fp16) {
#ifdef ENABLE_FP16
        add_spatial(
            (half *)data, (const half *)biases,
            (const half *)residual, (const half *)mask,
            bsize, batch, channels, spatial, relu, stream);
#endif
    } else {
        add_spatial(
            (float *)data, (const float *)biases,
            (const float *)residual, (const float *)mask,
            bsize, batch, channels, spatial, relu, stream);
    }
}

void Im2ColBatched(bool fp16, void *data_col, void *data_im,
                   int filter_size, int batch, int channels,
                   int height, int width, cudaStream_t stream) {
    if (fp16) {
#ifdef ENABLE_FP16
        im2col_batched(
            (half *)data_col, (half *)data_im, filter_size,
            batch, channels, height, width, stream);
#endif
    } else {
        im2col_batched(
            (float *)data_col, (float *)data_im, filter_size,
            batch, channels, height, width, stream);
    }
}

void NormalGlobalPooling(bool fp16, void *output, void *input,
                         const void *mask, const void *sqrt_mask,
                         int batch, int channels, int spatial, cudaStream_t stream) {
    if (fp16) {
#ifdef ENABLE_FP16
        global_pooling(
            (half *)output, (half *)input, (const half *)mask,
            (const half *)sqrt_mask, batch, channels, spatial, stream);
#endif
    } else {
        global_pooling(
            (float *)output, (float *)input, (const float *)mask,
            (const float *)sqrt_mask, batch, channels, spatial, stream);
    }
}

void HeadGlobalPooling(bool fp16, void *output, void *input, const void *sqrt_mask,
                       int batch, int channels, int spatial, cudaStream_t stream) {
    if (fp16) {
#ifdef ENABLE_FP16
        head_global_pooling(
            (half *)output, (half *)input, (const half *)sqrt_mask,
            batch, channels, spatial, stream);
#endif
    } else {
        head_global_pooling(
            (float *)output, (float *)input, (const float *)sqrt_mask,
            batch, channels, spatial, stream);
    }
}

void SeScale(bool fp16, void *output, const void *input,
             const void *residual, const void *se_biases,
             const void *mask, int batch, int channels,
             int spatial, bool relu, cudaStream_t stream) {
    if (fp16) {
#ifdef ENABLE_FP16
        se_scale(
            (half *)output, (const half *)input,
            (const half*)residual, (const half *)se_biases,
            (const half *)mask, batch, channels, spatial, relu, stream);
#endif
    } else {
        se_scale(
            (float *)output, (const float *)input,
            (const float*)residual, (const float *)se_biases,
            (const float *)mask, batch, channels, spatial, relu, stream);
    }
}

void ChannelPooling(bool fp16, void *output, void *input, const void *sqrt_mask,
                    int batch, int channels, int spatial, cudaStream_t stream) {
    if (fp16) {
#ifdef ENABLE_FP16
        channel_pooling(
            (half *)output, (half *)input, (const half *)sqrt_mask,
            batch, channels, spatial, stream);
#endif
    } else {
        channel_pooling(
            (float *)output, (float *)input, (const float *)sqrt_mask,
            batch, channels, spatial, stream);
    }
}

void SaScale(bool fp16, void *output, const void *input,
             const void *residual, const void *sa_biases,
             int batch, int channels, int spatial,
             bool relu, cudaStream_t stream) {
    if (fp16) {
#ifdef ENABLE_FP16
        sa_scale(
            (half *)output, (const half *)input, 
            (const half*)residual, (const half *)sa_biases,
            batch, channels, spatial, relu, stream);
#endif
    } else {
        sa_scale(
            (float *)output, (const float *)input, 
            (const float*)residual, (const float *)sa_biases,
            batch, channels, spatial, relu, stream);
    }
}

void Winograd3TransformIn(bool fp16, void *output, const void *input,
                          int batch, int channels, int board_size, cudaStream_t stream) {
    if (fp16) {
#ifdef ENABLE_FP16
        winograd3_transform_in(
            (half *)output, (const half *)input,
            batch, channels, board_size, stream);
#endif
    } else {
        winograd3_transform_in(
            (float *)output, (const float *)input,
            batch, channels, board_size, stream);
    }
}

void Winograd3TransformOut(bool fp16, void *output, const void *input,
                           const void *biases, const void *residual, const void *mask,
                           int batch, int channels, int board_size, bool relu, cudaStream_t stream) {
    if (fp16) {
#ifdef ENABLE_FP16
        winograd3_transform_out(
            (half *)output, (const half *)input, (const half *)biases,
            (const half *)residual, (const half *)mask,
            batch, channels, board_size, relu, stream);
#endif
    } else {
        winograd3_transform_out(
            (float *)output, (const float *)input, (const float *)biases,
            (const float *)residual, (const float *)mask,
            batch, channels, board_size, relu, stream);
    }
}

void Gemm(bool fp16, bool TA, bool TB,
          int M, int N, int K,
          float ALPHA,
          const void *A_gpu, int lda,
          const void *B_gpu, int ldb,
          float BETA,
          void *C_gpu, int ldc,
          cublasHandle_t handle, cudaStream_t stream) {
    if (fp16) {
#ifdef ENABLE_FP16
        half_float_t alpha = GetFp16(ALPHA);
        half_float_t beta = GetFp16(BETA);
        gemm(TA, TB,
             M, N, K,
             *(half*)&alpha,
             (const half *)A_gpu, lda,
             (const half *)B_gpu, ldb,
             *(half*)&beta,
             (half *)C_gpu, ldc,
             handle, stream);
#endif
    } else {
        gemm(TA, TB,
             M, N, K,
             ALPHA,
             (const float *)A_gpu, lda,
             (const float *)B_gpu, ldb,
             BETA,
             (float *)C_gpu, ldc,
             handle, stream);
    }
}

void GemmStridedBatched(bool fp16, bool TA, bool TB,
                        int M, int N, int K,
                        float ALPHA,
                        const void *A_gpu, int lda, int strideA,
                        const void *B_gpu, int ldb, int strideB,
                        float BETA,
                        void *C_gpu, int ldc, int strideC,
                        int batchsize,
                        cublasHandle_t handle, cudaStream_t stream) {
    if (fp16) {
#ifdef ENABLE_FP16
        half_float_t alpha = GetFp16(ALPHA);
        half_float_t beta = GetFp16(BETA);
        gemm_strided_batched(
            TA, TB, M, N, K,
            *(half*)&alpha,
            (const half *)A_gpu, lda, strideA,
            (const half *)B_gpu, ldb, strideB,
            *(half*)&beta,
            (half *)C_gpu, ldc, strideC,
            batchsize,
            handle, stream
        );
#endif
    } else {
        gemm_strided_batched(
            TA, TB, M, N, K,
            ALPHA,
            (const float *)A_gpu, lda, strideA,
            (const float *)B_gpu, ldb, strideB,
            BETA,
            (float *)C_gpu, ldc, strideC,
            batchsize,
            handle, stream
        );
    }
}

Convolution::Convolution(CudaHandles *handles,
                         const int max_batch,
                         const int board_size, 
                         const int filter_size,
                         const int input_channels,
                         const int output_channels,
                         bool ReLU) {
    width_ = board_size;
    height_ = board_size;
    spatial_size_ = width_ * height_;

    in_channels_ = input_channels;
    out_channels_ = output_channels;
    filters_ = filter_size;
    filter_dim_ = filters_ * filters_ * in_channels_;
    maxbatch_ = max_batch;

    handles_ = handles;

    fp16_ = handles->fp16;
    loaded_ = false;
    relu_ = ReLU;

#ifdef USE_CUDNN
    cudnnCreateFilterDescriptor(&filter_desc_);
    cudnnCreateTensorDescriptor(&in_tensor_desc_);
    cudnnCreateTensorDescriptor(&out_tensor_desc_);
    cudnnCreateConvolutionDescriptor(&conv_desc_);
    cudnnCreateTensorDescriptor(&bias_desc_);
#endif
}

Convolution::~Convolution() {
    if (loaded_) {
        ReportCUDAErrors(cudaFree(cuda_weights_));

#ifdef USE_CUDNN
        cudnnDestroyFilterDescriptor(filter_desc_);
        cudnnDestroyConvolutionDescriptor(conv_desc_);
        cudnnDestroyTensorDescriptor(in_tensor_desc_);
        cudnnDestroyTensorDescriptor(out_tensor_desc_);
        cudnnDestroyTensorDescriptor(bias_desc_);
#endif

        if (cuda_biases_) {
            ReportCUDAErrors(cudaFree(cuda_biases_));
        }
    }
}

void Convolution::Forward(const int batch,
                          void *output, void *input,
                          const void *residual,
                          const void *mask,
                          void *scratch, void *scratch_other, size_t scratch_size) {
    if (!loaded_) {
        return;
    }
    if (!scratch || !scratch_other) {
        return;
    }

    assert(batch <= maxbatch_);

#ifdef USE_CUDNN
    cudnnDataType_t cudnn_data_type = GetCudnnDataType(fp16_);

    ReportCUDNNErrors(cudnnSetStream(handles_->cudnn_handle, handles_->stream));   
    ReportCUDNNErrors(cudnnSetTensor4dDescriptor(in_tensor_desc_,
                                                 CUDNN_TENSOR_NCHW,
                                                 cudnn_data_type,
                                                 batch, in_channels_, height_, width_));
    ReportCUDNNErrors(cudnnSetTensor4dDescriptor(out_tensor_desc_,
                                                 CUDNN_TENSOR_NCHW,
                                                 cudnn_data_type,
                                                 batch, out_channels_, height_, width_));
  
    static constexpr float alpha = 1.0f, beta = 0.0f;
    ReportCUDNNErrors(cudnnConvolutionForward(
                      handles_->cudnn_handle, &alpha, in_tensor_desc_, input, filter_desc_, cuda_weights_,
                      conv_desc_, conv_algo_, scratch, scratch_size, &beta, out_tensor_desc_,
                      output));

    AddSpatial(
        fp16_, output, cuda_biases_,
        residual, mask,
        out_channels_,
        batch, out_channels_, spatial_size_, relu_,
        handles_->stream);

#else
    (void) scratch_size;
    auto scratch_op = reinterpret_cast<void *>(scratch);

    const int board_size = (width_ + height_) / 2;

    if (winograd_) {
        auto scratch_op_other = reinterpret_cast<void *>(scratch_other);
        const int batch_ptiles = batch * GetWinogradP(board_size);

        Winograd3TransformIn(
            fp16_, scratch_op, input,
            batch, in_channels_, board_size, handles_->stream);
        GemmStridedBatched(
            fp16_, true, false,
            out_channels_, batch_ptiles, in_channels_,
            1.0f,
            cuda_weights_, out_channels_, in_channels_ * out_channels_,
            scratch_op, batch_ptiles, in_channels_ * batch_ptiles,
            0.0f,
            scratch_op_other, batch_ptiles, out_channels_ * batch_ptiles,
            kWinogradTile,
            handles_->cublas_handle, handles_->stream);
        Winograd3TransformOut(
            fp16_, output, scratch_op_other,
            cuda_biases_, residual, mask,
            batch, out_channels_, board_size, relu_, handles_->stream);
    } else {
        if (filters_ != 1) {
            Im2ColBatched(
                fp16_, scratch_op, input, filters_,
                batch, in_channels_, height_, width_, handles_->stream);
            GemmStridedBatched(
                fp16_, false, false,
                out_channels_, spatial_size_, filter_dim_,
                1.0f,
                cuda_weights_, filter_dim_, 0,
                scratch_op, spatial_size_, filter_dim_ * spatial_size_,
                0.f,
                output, spatial_size_, out_channels_ * spatial_size_,
                batch,
                handles_->cublas_handle, handles_->stream);
        } else {
            GemmStridedBatched(
                fp16_, false, false,
                out_channels_, spatial_size_, filter_dim_,
                1.0f,
                cuda_weights_, filter_dim_, 0,
                input, spatial_size_, in_channels_ * spatial_size_,
                0.f,
                output, spatial_size_, out_channels_ * spatial_size_,
                batch,
                handles_->cublas_handle, handles_->stream);
        }
        AddSpatial(
            fp16_, output, cuda_biases_,
            residual, mask,
            out_channels_,
            batch, out_channels_, spatial_size_, relu_,
            handles_->stream);
    }
#endif
}

void Convolution::LoadWeights(const std::vector<float> &weights,
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

    MallocAndCopy(fp16_, &cuda_weights_, weights_copy);

    loaded_ = true;
    size_t apply_scratch_size = 0;

#ifdef USE_CUDNN
    cudnnDataType_t cudnn_data_type = GetCudnnDataType(fp16_);

    if (handles_->has_tensor_cores) {
        ReportCUDNNErrors(cudnnSetConvolutionMathType(
                              conv_desc_, CUDNN_TENSOR_OP_MATH));
    }

    ReportCUDNNErrors(cudnnSetFilter4dDescriptor(filter_desc_, cudnn_data_type, CUDNN_TENSOR_NCHW,
                                                 out_channels_, in_channels_, filters_, filters_));
  
    const size_t padding = filters_ / 2;
    ReportCUDNNErrors(cudnnSetConvolution2dDescriptor(
                      conv_desc_, padding, padding, 1, 1, 1, 1,
                      CUDNN_CROSS_CORRELATION, cudnn_data_type));

    ReportCUDNNErrors(cudnnSetTensor4dDescriptor(in_tensor_desc_,
                                                 CUDNN_TENSOR_NCHW,
                                                 cudnn_data_type,
                                                 maxbatch_, in_channels_, height_, width_));

    ReportCUDNNErrors(cudnnSetTensor4dDescriptor(out_tensor_desc_,
                                                 CUDNN_TENSOR_NCHW,
                                                 cudnn_data_type,
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
    const int board_size = (width_ + height_) / 2;
    int scratch_size_base = 0;

    if (winograd_) {
        scratch_size_base = kWinogradTile *
                                GetWinogradP(board_size) *
                                std::max(out_channels_, in_channels_);
    } else {
        scratch_size_base = filter_dim_ * spatial_size_;
    }
    apply_scratch_size = maxbatch_ * scratch_size_base * sizeof(float);
    scratch_size = std::max(apply_scratch_size, scratch_size);
#endif
}


void Convolution::LoadWeights(const std::vector<float> &weights,
                              const std::vector<float> &biases,
                              size_t &scratch_size, bool winpgrad) {
    if (loaded_) {
        return;
    }

    MallocAndCopy(fp16_, &cuda_biases_, biases);

#ifdef USE_CUDNN
    cudnnDataType_t cudnn_data_type = GetCudnnDataType(fp16_);
    ReportCUDNNErrors(cudnnSetTensor4dDescriptor(bias_desc_,
                                                 CUDNN_TENSOR_NCHW,
                                                 cudnn_data_type,
                                                 1, out_channels_, 1, 1));
#endif
    LoadWeights(weights, scratch_size, winpgrad);
}


FullyConnect::FullyConnect(CudaHandles *handles,
                           const int max_batch,
                           const int inputs, 
                           const int outputs,
                           bool ReLU) {
    maxbatch_ = max_batch;
    inputs_ = inputs;
    outputs_ = outputs;
    fp16_ = handles->fp16;
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

void FullyConnect::LoadWeights(const std::vector<float> &weights,
                               const std::vector<float> &biases) {
    if (loaded_) { 
        return;
    }
    MallocAndCopy(fp16_, &cuda_weights_, weights);
    MallocAndCopy(fp16_, &cuda_biases_, biases);
    loaded_ = true;
}

void FullyConnect::Forward(const int batch, void *output, void *input) {
    if (!loaded_) {
        return;
    }
    assert(batch <= maxbatch_);
    Gemm(fp16_, false, true,
         batch, outputs_, inputs_,
         1.0f,
         input, inputs_, 
         cuda_weights_, inputs_,
         0.0f,
         output, outputs_,
         handles_->cublas_handle, handles_->stream);
    AddVectors(
        fp16_, output, cuda_biases_, output,
        outputs_ * batch, outputs_, outputs_ * batch, relu_, handles_->stream );
}

GlobalPooling::GlobalPooling(CudaHandles *handles,
                             bool is_value_head,
                             const int max_batch,
                             const int board_size,
                             const int channels) {
    width_ = board_size;
    height_ = board_size;
    spatial_size_ = width_ * height_;
    is_value_head_ = is_value_head;

    fp16_ = handles->fp16;
    maxbatch_ = max_batch;
    channels_ = channels;
    handles_ = handles;
}

void GlobalPooling::Forward(const int batch, void *output,
                            void *input, void *mask, void *sqrt_mask) {
    if (is_value_head_) {
        HeadGlobalPooling(
            fp16_, output, input, sqrt_mask, batch,
            channels_, spatial_size_, handles_->stream);
    } else {
        NormalGlobalPooling(
            fp16_, output, input, mask, sqrt_mask, batch,
            channels_, spatial_size_, handles_->stream);
    }
}


SEUnit::SEUnit(CudaHandles *handles,
               const int max_batch,
               const int board_size,
               const int channels,
               const int se_size,
               bool ReLU) {
    width_ = board_size;
    height_ = board_size;
    spatial_size_ = width_ * height_;
    relu_ = ReLU;

    fp16_ = handles->fp16;
    se_size_ = se_size;
    maxbatch_ = max_batch;
    channels_ = channels;
    loaded_ = false;
    handles_ = handles;
}

void SEUnit::LoadWeights(const std::vector<float> &weights_w1,
                         const std::vector<float> &weights_b1,
                         const std::vector<float> &weights_w2,
                         const std::vector<float> &weights_b2) {
    if (loaded_) { 
        return;
    }
    MallocAndCopy(fp16_, &cuda_weights_w1_, weights_w1);
    MallocAndCopy(fp16_, &cuda_weights_b1_, weights_b1);
    MallocAndCopy(fp16_, &cuda_weights_w2_, weights_w2);
    MallocAndCopy(fp16_, &cuda_weights_b2_, weights_b2);

    const size_t fc1_scratch_size  = maxbatch_ * se_size_;
    const size_t fc2_scratch_size  = maxbatch_ * 2 * channels_;
    const size_t pool_scratch_size = maxbatch_ * 3 * channels_;

    MallocCudaOp(fp16_, &(cuda_op_[0]), pool_scratch_size);
    MallocCudaOp(fp16_, &(cuda_op_[1]), fc1_scratch_size);
    MallocCudaOp(fp16_, &(cuda_op_[2]), fc2_scratch_size);
    loaded_ = true;
}

void SEUnit::Forward(const int batch, void *ouput, void *input,
                     void *residual, void *mask, void *sqrt_mask) {
    if (!loaded_) {
        return;
    }
    NormalGlobalPooling(
        fp16_, cuda_op_[0], input, mask, sqrt_mask,
        batch, channels_, spatial_size_, handles_->stream);

    const size_t fc1_input_size = 3 * channels_;
    const size_t fc1_output_size = se_size_;
    const bool fc1_relu = true;
    Gemm(fp16_, false, true,
         batch, fc1_output_size, fc1_input_size, 
         1.0f,
         cuda_op_[0], fc1_input_size, 
         cuda_weights_w1_, fc1_input_size,
         0.0f,
         cuda_op_[1], fc1_output_size,
         handles_->cublas_handle, handles_->stream);
    AddVectors(
        fp16_, cuda_op_[1], cuda_weights_b1_, cuda_op_[1],
        fc1_output_size * batch, fc1_output_size, fc1_output_size * batch, fc1_relu, handles_->stream);

    const size_t fc2_input_size = se_size_;
    const size_t fc2_output_size = 2 * channels_;
    const bool fc2_relu = false;
    Gemm(fp16_, false, true,
         batch, fc2_output_size, fc2_input_size, 
         1.0f,
         cuda_op_[1], fc2_input_size,
         cuda_weights_w2_, fc2_input_size,
         0.0f,
         cuda_op_[2], fc2_output_size,
         handles_->cublas_handle, handles_->stream);

    AddVectors(
        fp16_, cuda_op_[2], cuda_weights_b2_, cuda_op_[2],
        fc2_output_size * batch, fc2_output_size, fc2_output_size * batch, fc2_relu, handles_->stream);
    SeScale(
        fp16_, ouput, input, residual, cuda_op_[2], mask,
        batch, channels_, spatial_size_, relu_, handles_->stream);
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

SAUnit::SAUnit(CudaHandles *handles,
               const int max_batch,
               const int board_size,
               const int channels,
               bool ReLU) {
    width_ = board_size;
    height_ = board_size;
    spatial_size_ = width_ * height_;
    relu_ = ReLU;

    fp16_ = handles->fp16;
    maxbatch_ = max_batch;
    channels_ = channels;
    loaded_ = false;
    handles_ = handles;

    conv_ = Convolution(
        handles, maxbatch_, board_size,
        7, 3, 1, false);
}

void SAUnit::LoadWeights(const std::vector<float> &weights,
                         const std::vector<float> &biases,
                         size_t &scratch_size, bool winograd) {
    if (loaded_) { 
        return;
    }

    const size_t pool_scratch_size  = maxbatch_ * 3 * spatial_size_;
    const size_t conv_out_size  = maxbatch_ * 1 * spatial_size_;

    MallocCudaOp(fp16_, &(cuda_op_[0]), pool_scratch_size);
    MallocCudaOp(fp16_, &(cuda_op_[1]), conv_out_size);

    conv_.LoadWeights(weights, biases, scratch_size, winograd);
    loaded_ = true;
}

void SAUnit::Forward(const int batch, void *output, void *input,
                     void *residual, void *mask, void *sqrt_mask,
                     void *scratch, void *scratch_other, size_t scratch_size) {
    ChannelPooling(
        fp16_, cuda_op_[0], input, sqrt_mask, batch,
        channels_, spatial_size_, handles_->stream);
    void *null_op = nullptr;
    conv_.Forward(
        batch, cuda_op_[1], cuda_op_[0], null_op, mask,
        scratch, scratch_other, scratch_size);
    SaScale(
        fp16_, output, input, residual, cuda_op_[1],
        batch, channels_, spatial_size_, relu_, handles_->stream);
}

SAUnit::~SAUnit() {
    if (loaded_) {
        ReportCUDAErrors(cudaFree(cuda_op_[0]));
        ReportCUDAErrors(cudaFree(cuda_op_[1]));
    }
}

} // namespace cuda

#endif
