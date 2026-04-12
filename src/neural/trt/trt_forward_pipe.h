#pragma once

#ifdef USE_TENSORRT

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "NvInfer.h"
#include "neural/activation.h"
#include "neural/batch_forward_pipe.h"
#include "neural/cuda/cuda_common.h"
#include "neural/description.h"
#include "neural/network_basic.h"
#include "neural/trt/trt_common.h"

class TrtForwardPipe : public BatchForwardPipe {
public:
    void Initialize(std::shared_ptr<DNNWeights> weights) override;
    OutputResult Forward(const InputData& input) override;
    bool Valid() const override;
    void Construct(ForwardPipeOption option, std::shared_ptr<DNNWeights> weights) override;
    void Release() override;
    void Destroy() override;
    int GetNumWorkers() const override;
    std::vector<OutputResult> BatchForward(int gpu, const std::vector<InputData>& inputs) override;

private:
    struct BackendContext {
        trt::InferPtr<nvinfer1::IExecutionContext> execution_context_{nullptr};
        std::map<std::string, void*> buffers_;
    };

    class TrtEngine {
    public:
        bool Build(bool dump_gpu_info,
                   int gpu,
                   int max_batch_size,
                   int board_size,
                   std::shared_ptr<DNNWeights> weights,
                   trt::Logger& logger);
        std::vector<OutputResult> BatchForward(const std::vector<InputData>& batch_input);
        void Destroy();

    private:
        bool CreatePlan(trt::InferPtr<nvinfer1::INetworkDefinition>& network,
                        trt::InferPtr<nvinfer1::IBuilderConfig>& config,
                        trt::InferPtr<nvinfer1::IBuilder>& builder,
                        int max_batch_size,
                        trt::Logger& logger);
        bool BuildNetwork(trt::InferPtr<nvinfer1::INetworkDefinition>& network);

        void SetComputationMode(cuda::CudaHandles* handles);
        void FillOutputs(const std::vector<float>& batch_prob,
                         const std::vector<float>& batch_prob_pass,
                         const std::vector<float>& batch_value_misc,
                         const std::vector<float>& batch_ownership,
                         const std::vector<InputData>& batch_input,
                         std::vector<OutputResult>& batch_output_result);
        void FreeCudaWeights();

        bool ReadFileBinary(const std::string& filename, std::string& out);
        bool WriteFileBinary(const std::string& filename, const std::string& data);
        std::string GetDeviceIdent(const char* device_name);

        static std::mutex tune_mutex_;

        nvinfer1::ILayer* BuildResidualBlock(trt::InferPtr<nvinfer1::INetworkDefinition>& network,
                                             nvinfer1::ITensor* input,
                                             BlockBasic* tower_ptr);
        nvinfer1::ILayer* BuildBottleneckBlock(trt::InferPtr<nvinfer1::INetworkDefinition>& network,
                                               nvinfer1::ITensor* input,
                                               BlockBasic* tower_ptr);
        nvinfer1::ILayer*
        BuildNestedBottleneckBlock(trt::InferPtr<nvinfer1::INetworkDefinition>& network,
                                   nvinfer1::ITensor* input,
                                   BlockBasic* tower_ptr);
        nvinfer1::ILayer* BuildMixerBlock(trt::InferPtr<nvinfer1::INetworkDefinition>& network,
                                          nvinfer1::ITensor* input,
                                          BlockBasic* tower_ptr);
        nvinfer1::ILayer*
        BuildSqueezeExcitationLayer(trt::InferPtr<nvinfer1::INetworkDefinition>& network,
                                    nvinfer1::ITensor* residual,
                                    nvinfer1::ITensor* input,
                                    BlockBasic* tower_ptr);
        void BuildPolicyHead(trt::InferPtr<nvinfer1::INetworkDefinition>& network,
                             nvinfer1::ITensor* input);
        void BuildValueHead(trt::InferPtr<nvinfer1::INetworkDefinition>& network,
                            nvinfer1::ITensor* input);

        nvinfer1::ILayer* BuildConvLayer(trt::InferPtr<nvinfer1::INetworkDefinition>& network,
                                         nvinfer1::ITensor* input,
                                         unsigned int filter,
                                         unsigned int in_channels,
                                         unsigned int out_channels,
                                         const std::vector<float>& weights,
                                         const std::vector<float>& biases,
                                         bool depth_wise = false);
        nvinfer1::ILayer* BuildActivationLayer(trt::InferPtr<nvinfer1::INetworkDefinition>& network,
                                               nvinfer1::ITensor* input,
                                               Activation act,
                                               bool need_mask = true);
        nvinfer1::ILayer* BuildGPoolLayer(trt::InferPtr<nvinfer1::INetworkDefinition>& network,
                                          nvinfer1::ITensor* input,
                                          bool is_value_head = false);
        nvinfer1::ILayer* BuildMaskLayer(trt::InferPtr<nvinfer1::INetworkDefinition>& network,
                                         nvinfer1::ITensor* input);

        cuda::CudaHandles handles_;
        std::shared_ptr<DNNWeights> weights_{nullptr};
        trt::InferPtr<nvinfer1::IRuntime> runtime_{nullptr};
        trt::InferPtr<nvinfer1::ICudaEngine> cuda_engine_{nullptr};
        std::unique_ptr<BackendContext> context_{nullptr};

        std::string weights_file_;
        int board_size_{0};
        int max_batch_{0};

        nvinfer1::ITensor* input_mask_{nullptr};
        nvinfer1::ILayer* mask_sum_layer_{nullptr};
        nvinfer1::ILayer* mask_scale_layer_{nullptr};
        nvinfer1::ILayer* mask_quad_layer_{nullptr};
        nvinfer1::ICastLayer* shape_layer_{nullptr};

        std::vector<void*> cuda_weights_op_;
        std::vector<std::unique_ptr<float[]>> extra_weights_;
    };

    trt::Logger trt_logger_;
    std::vector<std::unique_ptr<TrtEngine>> nngraphs_;

    bool dump_gpu_info_{false};
    int max_batch_per_nn_{0};
    int board_size_{0};
};

#endif
