#pragma once

#ifdef USE_CUDA
#include "neural/cuda/cuda_common.h"

#include <vector>
#include <array>

namespace cuda {

void AddSpatial(bool fp16, void *data, const void *biases,
                const void *residual, const void *mask,
                int bsize, int batch, int channels, int spatial,
                bool relu, cudaStream_t stream);

class LayerBasic {
protected:
    CudaHandles *handles_{nullptr};
    bool loaded_{false};
    bool relu_{false};
    bool fp16_{false};

    int maxbatch_{0};
    int width_{0};
    int height_{0};
    int spatial_size_{0};
};

class Convolution : public LayerBasic {
public:
    Convolution() = default;
    Convolution(CudaHandles *handles, const int batch,
                const int board_size, const int filter,
                const int in_channels, const int out_channels,
                bool ReLU = true);
    ~Convolution();

    void Forward(const int batch,
                 void *output, void *input,
                 const void *residual, const void *mask,
                 void *scratch, void *scratch_other, size_t scratch_size);

    void LoadingWeight(const std::vector<float> &weights,
                       size_t &scratch_size,
                       bool winograd);

    void LoadingWeight(const std::vector<float> &weights,
                       const std::vector<float> &biases,
                       size_t &scratch_size,
                       bool winograd);

private:
    int filter_dim_;
    int filters_;
    int in_channels_;
    int out_channels_;
    bool winograd_;

#ifdef USE_CUDNN
    cudnnFilterDescriptor_t filter_desc_;
    cudnnTensorDescriptor_t in_tensor_desc_;
    cudnnTensorDescriptor_t out_tensor_desc_;

    cudnnConvolutionDescriptor_t conv_desc_;

    cudnnTensorDescriptor_t bias_desc_;
    cudnnConvolutionFwdAlgo_t conv_algo_;
#endif

    void *cuda_weights_;
    void *cuda_biases_{nullptr};
};

class FullyConnect : public LayerBasic {
public:
    FullyConnect() = default;
    FullyConnect(CudaHandles *handles,
                 const int batch,
                 const int inputs, 
                 const int outputs,
                 bool ReLU);
    ~FullyConnect();

    void Forward(const int batch, void *output, void *input);

    void LoadingWeight(const std::vector<float> &weights,
                       const std::vector<float> &biases);
private:
    int inputs_;
    int outputs_;

    void *cuda_weights_;
    void *cuda_biases_;
};

class GlobalPooling : public LayerBasic {
public:
    GlobalPooling() = default; 
    GlobalPooling(CudaHandles *handles,
                  bool is_value_head,
                  const int batch,
                  const int board_size,
                  const int channels);

    void Forward(const int batch,
                 void *output, void *input,
                 void *mask, void *sqrt_mask);

private:
    int channels_;
    bool is_value_head_;
};

class SEUnit : public LayerBasic {
public:
    SEUnit() = default;
    SEUnit(CudaHandles *handles,
           const int batch,
           const int board_size,
           const int channels,
           const int se_size);
    ~SEUnit();

    void LoadingWeight(const std::vector<float> &weights_w1,
                       const std::vector<float> &weights_b1,
                       const std::vector<float> &weights_w2,
                       const std::vector<float> &weights_b2);

    void Forward(const int batch, void *output, void *input, void *mask);
 
private:
    int se_size_;
    int channels_;

    std::array<void *, 3> cuda_op_;

    void *cuda_weights_w1_;
    void *cuda_weights_b1_;
    void *cuda_weights_w2_;
    void *cuda_weights_b2_;
};
} // namespace cuda

#endif
