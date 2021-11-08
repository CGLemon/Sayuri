#pragma once

#ifdef USE_CUDA
#include "neural/cuda/cuda_common.h"

#include <vector>
#include <array>

namespace CUDA {

class LayerBasic {
protected:
    CudaHandles *handles_;
    bool loaded_{false};
    bool relu_{false};

    int maxbatch_{0};
    int width_{0};
    int height_{0};
    int spatial_size_{0};
    int channels_{0};
};

class Batchnorm : public LayerBasic {
public:
    Batchnorm() = default;
    Batchnorm(CudaHandles *handles, const int batch,
              const size_t board_size, const size_t channels, bool ReLU = true);
    ~Batchnorm();

    void Forward(const int batch, float *data,
                 const float *const eltwise = nullptr);

    void LoadingWeight(const std::vector<float> &means,
                       const std::vector<float> &stddevs);
private:
    float *cuda_means_;
    float *cuda_stddevs_;
};

class Convolution : public LayerBasic {
public:
    Convolution() = default;
    Convolution(CudaHandles *handles, const int batch,
                const size_t board_size, const size_t filter,
                const size_t in_channels, const size_t out_channels);
    ~Convolution();

    void Forward(const int batch, float *input, float *output,
                 void *scratch, size_t scratch_size);

    void LoadingWeight(const std::vector<float> &weights,
                       size_t &scratch_size);

    void LoadingWeight(const std::vector<float> &weights,
                       const std::vector<float> &biases,
                       size_t &scratch_size);

private:
    int filter_dim_;
    int filters_;
    int in_channels_;
    int out_channels_;

#ifdef USE_CUDNN
    cudnnFilterDescriptor_t filter_desc_;
    cudnnTensorDescriptor_t in_tensor_desc_;
    cudnnTensorDescriptor_t out_tensor_desc_;

    cudnnConvolutionDescriptor_t conv_desc_;

    cudnnTensorDescriptor_t bias_desc_;
    cudnnConvolutionFwdAlgo_t conv_algo_;
#endif

    float *cuda_weights_;
    float *cuda_biases_{nullptr};
};

class FullyConnect : public LayerBasic {
public:
    FullyConnect() = default;
    FullyConnect(CudaHandles *handles,
                 const int batch, const size_t inputs, 
                 const size_t outputs, bool ReLU);
    ~FullyConnect();

    void Forward(const int batch,
                 float *input,
                 float *output);

    void LoadingWeight(const std::vector<float> &weights,
                       const std::vector<float> &biases);
private:
    bool relu_;
    int inputs_;
    int outputs_;

    float *cuda_weights_;
    float *cuda_biases_;
};

class GlobalAvgPool: public LayerBasic {
public:
    GlobalAvgPool() = default; 
    GlobalAvgPool(CudaHandles *handles,
                  const int batch,
                  const size_t board_size,
                  const size_t channels);

    void Forward(const int batch, float *input, float *output);
};

class SEUnit : public LayerBasic {
public:
    SEUnit() = default;
    SEUnit(CudaHandles *handles, const int batch,
           const size_t board_size, const size_t channels, const size_t se_size);
    ~SEUnit();

    void LoadingWeight(const std::vector<float> &weights_w1,
                       const std::vector<float> &weights_b1,
                       const std::vector<float> &weights_w2,
                       const std::vector<float> &weights_b2);

    void Forward(const int batch, float *input, float *output);
 
private:
    int se_size_;

    std::array<float *, 3> cuda_op_;

    float *cuda_weights_w1_;
    float *cuda_weights_b1_;
    float *cuda_weights_w2_;
    float *cuda_weights_b2_;
};
} // namespace CUDA

#endif
