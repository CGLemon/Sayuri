#pragma once

#ifdef USE_CUDA
#include "neural/cuda/cuda_common.h"

#include <vector>
#include <array>

namespace CUDA {

static constexpr auto CONV_WIDTH = 9;
static constexpr auto CONV_HEIGHT = 9;

class Batchnorm {
public:
    Batchnorm() = default;
    Batchnorm(const int batch, const size_t board_size, const size_t channels, bool ReLU = true);
    ~Batchnorm();

    void Forward(const int batch, float *data,
                 const float *const eltwise = nullptr);

    void LoadingWeight(const std::vector<float> &means,
                       const std::vector<float> &stddevs);
private:
    int width;
    int height;
    int spatial_size;

    int m_channels;
    int m_maxbatch;

    bool m_ReLU;
    bool is_loaded{false};
    float *cuda_means;
    float *cuda_stddevs;
};

class Convolution {
public:
    Convolution() = default;
    Convolution(const int batch, const size_t board_size, const size_t filter,
                const size_t in_channels, const size_t out_channels);
    ~Convolution();

    void Forward(const int batch, float *input, float *output,
                 void *scratch, size_t scratch_size, CudaHandel *handel);

    void LoadingWeight(const std::vector<float> &weights,
                       size_t &scratch_size, CudaHandel *handel);

    void LoadingWeight(const std::vector<float> &weights,
                       const std::vector<float> &biases,
                       size_t &scratch_size, CudaHandel *handel);

private:
    int width;
    int height;
    int spatial_size;

    int m_filter_dim;
    int m_maxbatch;
    int m_filter;
    int m_in_channels;
    int m_out_channels;

#ifdef USE_CUDNN
    cudnnFilterDescriptor_t filter_desc;
    cudnnTensorDescriptor_t in_tensor_desc;
    cudnnTensorDescriptor_t out_tensor_desc;

    cudnnConvolutionDescriptor_t conv_desc;

    cudnnTensorDescriptor_t bias_desc;
    cudnnConvolutionFwdAlgo_t conv_algo;

    bool cudnn_applied{false};
#endif

    bool is_loaded{false};
    float *cuda_weights;
    float *cuda_biases{nullptr};
};

class FullyConnect {
public:
    FullyConnect() = default;
    FullyConnect(const int batch, const size_t inputs, 
                 const size_t outputs, bool ReLU);
    ~FullyConnect();

    void Forward(const int batch,
                 float *input,
                 float *output,
                 CudaHandel *handel);

    void LoadingWeight(const std::vector<float> &weights,
                       const std::vector<float> &biases);
private:
    bool m_ReLU;
    int m_maxbatch;
    int m_inputs;
    int m_outputs;

    bool is_loaded{false};
    float *cuda_weights;
    float *cuda_biases;
};

class GlobalAvgPool {
public:
    GlobalAvgPool() = default; 
    GlobalAvgPool(const int batch,
                  const size_t board_size,
                  const size_t channels);

    void Forward(const int batch, float *input, float *output);

private:
    int width;
    int height;
    int spatial_size;

    int m_maxbatch;
    int m_channels;
};

class SEUnit {
public:
    SEUnit() = default;
    SEUnit(const int batch, const size_t board_size, const size_t channels, const size_t se_size);
    ~SEUnit();

    void LoadingWeight(const std::vector<float> &weights_w1,
                       const std::vector<float> &weights_b1,
                       const std::vector<float> &weights_w2,
                       const std::vector<float> &weights_b2);

    void Forward(const int batch, float *input, float *output, CudaHandel *handel);
 
private:
    int width;
    int height;
    int spatial_size;

    int m_se_size;
    int m_maxbatch;
    int m_channels;

    bool is_loaded{false};
    std::array<float *, 3> cuda_op;

    float *cuda_weights_w1;
    float *cuda_weights_b1;
    float *cuda_weights_w2;
    float *cuda_weights_b2;
};
} // namespace CUDA

#endif
