#include "neural/trt/trt_forward_pipe.h"

#ifdef USE_TENSORRT

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "neural/encoder.h"
#include "utils/format.h"
#include "utils/log.h"
#include "utils/option.h"
#include "utils/sha256.h"

std::mutex TrtForwardPipe::TrtEngine::tune_mutex_;

void TrtForwardPipe::Initialize(std::shared_ptr<DNNWeights> weights) {
    LOGGING << cuda::GetBackendInfo();

    dump_gpu_info_ = true;

    auto option = ForwardPipeOption::Get()
                      .SetBoardSize(GetOption<int>("default_boardsize"))
                      .SetBatchSize(GetOption<int>("batch_size"));
    Construct(option, weights);

    BatchForwardPipe::AssignWorkers(static_cast<int>(nngraphs_.size()));
}

OutputResult TrtForwardPipe::Forward(const InputData& input) {
    return BatchForwardPipe::SendQueryAndWait(input);
}

bool TrtForwardPipe::Valid() const {
    return weights_ != nullptr;
}

int TrtForwardPipe::GetNumWorkers() const {
    return static_cast<int>(nngraphs_.size());
}

std::vector<OutputResult> TrtForwardPipe::BatchForward(int gpu,
                                                       const std::vector<InputData>& inputs) {
    return nngraphs_[gpu]->BatchForward(inputs);
}

void TrtForwardPipe::Construct(ForwardPipeOption option, std::shared_ptr<DNNWeights> weights) {
    if (weights) {
        weights_ = weights;
    }
    if (weights_ == nullptr) {
        return;
    }

    int board_size = option.IsValidBoardSize() ? option.board_size : board_size_;
    int batch_size = option.IsValidBatchSize() ? option.batch_size : max_batch_per_nn_;
    board_size = std::max(board_size, GetOption<int>("fixed_nn_boardsize"));

    if (board_size == 0 || batch_size == 0) {
        LOGGING << "NN board size/batch size should be larger than zero.\n";
        return;
    }

    BatchForwardPipe::SetForwardingSize(batch_size);

    if (board_size_ >= board_size && batch_size <= max_batch_per_nn_) {
        // TensorRT batches already carry an input mask, and BatchForwardPipe
        // can reorder between the requested board size and the constructed NN
        // board size. Reuse an existing larger engine instead of rebuilding it.
        return;
    }
    Release();

    board_size_ = board_size;
    max_batch_per_nn_ = batch_size;
    BatchForwardPipe::SetBoardSize(board_size);

    const auto cuda_device_cnt = cuda::GetDeviceCount();
    const auto specific_gpus_cnt = GetOptionCount("gpus");
    auto gpus_list = std::vector<int>{};

    for (int idx = 0; idx < specific_gpus_cnt; ++idx) {
        auto gpu_id = GetOption<int>("gpus", idx);
        if (gpu_id < cuda_device_cnt) {
            gpus_list.emplace_back(gpu_id);
        } else {
            LOGGING << Format("Not found GPU device (%d).\n", gpu_id);
        }
    }

    if (gpus_list.empty()) {
        LOGGING << "Not found any specific GPU device! Now assign the GPU(s) automatically.\n";
        for (int i = 0; i < cuda_device_cnt; ++i) {
            gpus_list.emplace_back(i);
        }
    }
    if (gpus_list.empty()) {
        throw std::runtime_error("No executable GPU device!");
    }

    for (size_t i = 0; i < gpus_list.size(); ++i) {
        nngraphs_.emplace_back(std::make_unique<TrtEngine>());
    }

    for (size_t i = 0; i < gpus_list.size(); ++i) {
        if (!nngraphs_[i]->Build(dump_gpu_info_,
                                 gpus_list[i],
                                 max_batch_per_nn_,
                                 board_size_,
                                 weights_,
                                 trt_logger_)) {
            throw std::runtime_error("TensorRT backend: failed to construct network!");
        }
    }

    dump_gpu_info_ = false;
}

void TrtForwardPipe::Release() {
    for (auto& g : nngraphs_) {
        g->Destroy();
    }
    nngraphs_.clear();
}

void TrtForwardPipe::Destroy() {
    Release();
    BatchForwardPipe::QuitWorkers();
}

bool TrtForwardPipe::TrtEngine::Build(bool dump_gpu_info,
                                      int gpu,
                                      int max_batch_size,
                                      int board_size,
                                      std::shared_ptr<DNNWeights> weights,
                                      trt::Logger& logger) {
    if (weights) {
        weights_ = weights;
    }
    if (weights_ == nullptr) {
        return false;
    }

    Destroy();

    board_size_ = board_size;
    max_batch_ = max_batch_size;

    cuda::SetDevice(gpu);
    handles_.ApplyOnCurrentDevice();
    SetComputationMode(&handles_);

    if (dump_gpu_info) {
        LOGGING << cuda::GetCurrentDeviceInfo(&handles_);
    }

    logger.SetReportableSeverity(nvinfer1::ILogger::Severity::kWARNING);

    auto builder = trt::InferPtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger.Get()));
    if (!builder) {
        LOGGING << "TensorRT backend: failed to create builder.\n";
        return false;
    }

    auto config = trt::InferPtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        LOGGING << "TensorRT backend: failed to create builder config.\n";
        return false;
    }

    if (handles_.fp16) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);

        // Honor the layers which are required to run in FP32.
        config->setFlag(nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
    }

    auto profile = builder->createOptimizationProfile();
    if (!profile) {
        LOGGING << "TensorRT backend: failed to create optimization profile.\n";
        return false;
    }

    profile->setDimensions("InputFeature",
                           nvinfer1::OptProfileSelector::kMIN,
                           nvinfer1::Dims4(1, weights_->input_channels, board_size_, board_size_));
    profile->setDimensions(
        "InputFeature",
        nvinfer1::OptProfileSelector::kOPT,
        nvinfer1::Dims4(max_batch_, weights_->input_channels, board_size_, board_size_));
    profile->setDimensions(
        "InputFeature",
        nvinfer1::OptProfileSelector::kMAX,
        nvinfer1::Dims4(max_batch_, weights_->input_channels, board_size_, board_size_));
    profile->setDimensions("InputMask",
                           nvinfer1::OptProfileSelector::kMIN,
                           nvinfer1::Dims4(1, 1, board_size_, board_size_));
    profile->setDimensions("InputMask",
                           nvinfer1::OptProfileSelector::kOPT,
                           nvinfer1::Dims4(max_batch_, 1, board_size_, board_size_));
    profile->setDimensions("InputMask",
                           nvinfer1::OptProfileSelector::kMAX,
                           nvinfer1::Dims4(max_batch_, 1, board_size_, board_size_));
    profile->setDimensions("BatchSize",
                           nvinfer1::OptProfileSelector::kMIN,
                           nvinfer1::Dims4(1, weights_->residual_channels, 1, 1));
    profile->setDimensions("BatchSize",
                           nvinfer1::OptProfileSelector::kOPT,
                           nvinfer1::Dims4(max_batch_, weights_->residual_channels, 1, 1));
    profile->setDimensions("BatchSize",
                           nvinfer1::OptProfileSelector::kMAX,
                           nvinfer1::Dims4(max_batch_, weights_->residual_channels, 1, 1));
    config->addOptimizationProfile(profile);

    auto network = trt::InferPtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0U));
    if (!network) {
        LOGGING << "TensorRT backend: failed to create network definition.\n";
        return false;
    }

    network->setName(weights_->name.c_str());
    if (!BuildNetwork(network)) {
        LOGGING << "TensorRT backend: failed to build network.\n";
        return false;
    }

    const auto dev_prop = cuda::GetDeviceProp();
    if (dev_prop.major >= 8) {
        config->setTacticSources(
            1U << static_cast<uint32_t>(nvinfer1::TacticSource::kJIT_CONVOLUTIONS));
        config->setBuilderOptimizationLevel(2);
    }
    config->setProfileStream(cudaStreamPerThread);
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 31);

    if (!CreatePlan(network, config, builder, max_batch_, logger)) {
        return false;
    }

    TRT_ASSERT(context_->execution_context_->setOptimizationProfileAsync(0, cudaStreamPerThread));
    cuda::ReportCUDAErrors(cudaStreamSynchronize(cudaStreamPerThread));

    return true;
}

bool TrtForwardPipe::TrtEngine::CreatePlan(trt::InferPtr<nvinfer1::INetworkDefinition>& network,
                                           trt::InferPtr<nvinfer1::IBuilderConfig>& config,
                                           trt::InferPtr<nvinfer1::IBuilder>& builder,
                                           int max_batch_size,
                                           trt::Logger& logger) {
    auto GetNetworkCacheName = [](const std::filesystem::path& filepath) {
        auto network_name = filepath.filename().string();
        const auto dot_pos = network_name.find('.');
        if (dot_pos != std::string::npos) {
            network_name.erase(dot_pos);
        }
        return network_name;
    };

    std::string model_data;
    if (!ReadFileBinary(weights_->path, model_data)) {
        LOGGING << "Unable to read weights file for TensorRT plan cache.\n";
        return false;
    }

    const std::string model_hash = sha256::GetDigest(model_data);
    const auto dev_prop = cuda::GetDeviceProp();
    const std::string device_ident = GetDeviceIdent(dev_prop.name);
    const std::string precision = handles_.fp16 ? "half" : "single";
    const auto network_name = GetNetworkCacheName(std::filesystem::path(network->getName()));
    const auto trt_version = getInferLibVersion();

    const std::string plan_cache_file = Format("trt-%d_gpu-%s_net-%s_%dx%d_batch%d_%s",
                                               trt_version,
                                               device_ident.c_str(),
                                               network_name.c_str(),
                                               board_size_,
                                               board_size_,
                                               max_batch_size,
                                               precision.c_str());
    const std::string param_str = Format("_%d_%s_%d_%d_%d_%s",
                                         trt_version,
                                         device_ident.c_str(),
                                         board_size_,
                                         board_size_,
                                         max_batch_size,
                                         precision.c_str());

    std::string plan;
    {
        std::lock_guard<std::mutex> lock(tune_mutex_);

        std::string cache_file_data;
        if (ReadFileBinary(plan_cache_file, cache_file_data)) {
            const auto model_hash_len = sha256::GetHexDigestLength();
            if (cache_file_data.size() < model_hash_len + param_str.size()) {
                LOGGING << "Could not parse plan, unexpected size in " << plan_cache_file << ".\n";
                cache_file_data.clear();
            } else {
                const auto cached_param =
                    cache_file_data.substr(cache_file_data.size() - param_str.size());
                const auto cached_model_hash = cache_file_data.substr(
                    cache_file_data.size() - model_hash_len - param_str.size(), model_hash_len);

                if (cached_model_hash != model_hash) {
                    LOGGING << "Plan cache is corrupted or is for the wrong model in "
                            << plan_cache_file << ".\n";
                    cache_file_data.clear();
                } else if (cached_param != param_str) {
                    LOGGING << "Plan cache is corrupted or is for the wrong parameters in "
                            << plan_cache_file << ".\n";
                    cache_file_data.clear();
                } else {
                    plan = cache_file_data.substr(
                        0, cache_file_data.size() - model_hash_len - param_str.size());
                    LOGGING << "Using existing plan cache at " << plan_cache_file << ".\n";
                }
            }
        }

        if (plan.empty()) {
            LOGGING << "Creating new plan cache...\n";

            auto plan_buffer = trt::InferPtr<nvinfer1::IHostMemory>(
                builder->buildSerializedNetwork(*network, *config));
            if (!plan_buffer) {
                LOGGING << "TensorRT backend: failed to create plan.\n";
                return false;
            }

            plan.assign(static_cast<const char*>(plan_buffer->data()),
                        static_cast<const char*>(plan_buffer->data()) + plan_buffer->size());

            auto cache_payload = plan;
            cache_payload.append(model_hash);
            cache_payload.append(param_str);

#ifdef NDEBUG
            bool cache_tensorrt_plan = GetOption<bool>("cache_tensorrt_plan");
            if (cache_tensorrt_plan) {
                if (!WriteFileBinary(plan_cache_file, cache_payload)) {
                    LOGGING << "Unable to save TensorRT plan cache to " << plan_cache_file << ".\n";
                } else {
                    LOGGING << "Saved new plan cache to " << plan_cache_file << ".\n";
                }
            }
#endif
        }
    }

    runtime_ = trt::InferPtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger.Get()));
    if (!runtime_) {
        LOGGING << "createInferRuntime error.\n";
        return false;
    }

    cuda_engine_ = trt::InferPtr<nvinfer1::ICudaEngine>(
        runtime_->deserializeCudaEngine(plan.data(), plan.size()));
    if (!cuda_engine_) {
        LOGGING << "deserializeCudaEngine error.\n";
        return false;
    }

    context_ = std::make_unique<BackendContext>();
    context_->execution_context_ =
        trt::InferPtr<nvinfer1::IExecutionContext>(cuda_engine_->createExecutionContext());
    if (!context_->execution_context_) {
        LOGGING << "failed to create execution context.\n";
        return false;
    }

    for (int i = 0; i < cuda_engine_->getNbIOTensors(); ++i) {
        auto name = std::string(cuda_engine_->getIOTensorName(i));
        auto dims = cuda_engine_->getTensorShape(name.c_str());
        size_t element_size = name == "BatchSize" ? sizeof(int32_t) : sizeof(float);
        size_t bytes = element_size * static_cast<size_t>(max_batch_size);
        for (int d = 1; d < dims.nbDims; ++d) {
            bytes *= static_cast<size_t>(dims.d[d]);
        }

        void* buffer = nullptr;
        cuda::ReportCUDAErrors(cudaMalloc(&buffer, bytes));
        if (name == "BatchSize") {
            auto batch_tensor = std::vector<int32_t>(
                static_cast<size_t>(max_batch_size) * weights_->residual_channels, 0);
            cuda::ReportCUDAErrors(
                cudaMemcpy(buffer, batch_tensor.data(), bytes, cudaMemcpyHostToDevice));
        }

        context_->buffers_.emplace(name, buffer);
        if (cuda_engine_->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT) {
            TRT_ASSERT(context_->execution_context_->setInputTensorAddress(name.c_str(), buffer));
        } else {
            TRT_ASSERT(context_->execution_context_->setOutputTensorAddress(name.c_str(), buffer));
        }
    }

    return true;
}

void TrtForwardPipe::TrtEngine::SetComputationMode(cuda::CudaHandles* handles) {
    const auto dev_prop = cuda::GetDeviceProp();

    if (dev_prop.major <= 6 || !GetOption<bool>("fp16")) {
        handles->fp16 = false;
    }

    if (!handles->fp16) {
        handles->has_tensor_cores = false;
    }
}

bool TrtForwardPipe::TrtEngine::BuildNetwork(trt::InferPtr<nvinfer1::INetworkDefinition>& network) {
    input_mask_ = network->addInput(
        "InputMask", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4(-1, 1, board_size_, board_size_));
    TRT_ASSERT(input_mask_ != nullptr);
    input_mask_->setAllowedFormats(1U << static_cast<int>(nvinfer1::TensorFormat::kLINEAR));

    mask_sum_layer_ =
        network->addReduce(*input_mask_, nvinfer1::ReduceOperation::kSUM, 1U << 2 | 1U << 3, true);
    auto mask_width_layer =
        network->addUnary(*mask_sum_layer_->getOutput(0), nvinfer1::UnaryOperation::kSQRT);

    auto mask_scale_shift = std::make_unique<float[]>(1);
    auto mask_scale_scale = std::make_unique<float[]>(1);
    mask_scale_shift[0] = -1.4f;
    mask_scale_scale[0] = 0.1f;
    mask_scale_layer_ = network->addScale(*mask_width_layer->getOutput(0),
                                          nvinfer1::ScaleMode::kUNIFORM,
                                          {nvinfer1::DataType::kFLOAT, mask_scale_shift.get(), 1},
                                          {nvinfer1::DataType::kFLOAT, mask_scale_scale.get(), 1},
                                          {nvinfer1::DataType::kFLOAT, nullptr, 0});
    extra_weights_.push_back(std::move(mask_scale_shift));
    extra_weights_.push_back(std::move(mask_scale_scale));

    auto mask_center_shift = std::make_unique<float[]>(1);
    auto mask_center_power = std::make_unique<float[]>(1);
    mask_center_shift[0] = -14.0f;
    mask_center_power[0] = 2.0f;
    auto mask_center_square_layer =
        network->addScale(*mask_width_layer->getOutput(0),
                          nvinfer1::ScaleMode::kUNIFORM,
                          {nvinfer1::DataType::kFLOAT, mask_center_shift.get(), 1},
                          {nvinfer1::DataType::kFLOAT, nullptr, 0},
                          {nvinfer1::DataType::kFLOAT, mask_center_power.get(), 1});
    extra_weights_.push_back(std::move(mask_center_shift));
    extra_weights_.push_back(std::move(mask_center_power));

    auto mask_quad_shift = std::make_unique<float[]>(1);
    auto mask_quad_scale = std::make_unique<float[]>(1);
    mask_quad_shift[0] = -0.1f;
    mask_quad_scale[0] = 0.01f;
    mask_quad_layer_ = network->addScale(*mask_center_square_layer->getOutput(0),
                                         nvinfer1::ScaleMode::kUNIFORM,
                                         {nvinfer1::DataType::kFLOAT, mask_quad_shift.get(), 1},
                                         {nvinfer1::DataType::kFLOAT, mask_quad_scale.get(), 1},
                                         {nvinfer1::DataType::kFLOAT, nullptr, 0});
    extra_weights_.push_back(std::move(mask_quad_shift));
    extra_weights_.push_back(std::move(mask_quad_scale));

    auto batch_size_tensor =
        network->addInput("BatchSize",
                          nvinfer1::DataType::kINT32,
                          nvinfer1::Dims4(-1, weights_->residual_channels, 1, 1));
    TRT_ASSERT(batch_size_tensor != nullptr);
    batch_size_tensor->setAllowedFormats(1U << static_cast<int>(nvinfer1::TensorFormat::kLINEAR));
    auto input_shape_layer = network->addShape(*batch_size_tensor);
    shape_layer_ = network->addCast(*input_shape_layer->getOutput(0), nvinfer1::DataType::kINT32);

    auto input_feature =
        network->addInput("InputFeature",
                          nvinfer1::DataType::kFLOAT,
                          nvinfer1::Dims4(-1, weights_->input_channels, board_size_, board_size_));
    TRT_ASSERT(input_feature != nullptr);
    input_feature->setAllowedFormats(1U << static_cast<int>(nvinfer1::TensorFormat::kLINEAR));

    auto initial_conv_layer = BuildConvLayer(network,
                                             input_feature,
                                             weights_->input_conv.GetFilter(),
                                             weights_->input_conv.GetInputs(),
                                             weights_->input_conv.GetOutputs(),
                                             weights_->input_conv.GetWeights(),
                                             weights_->input_conv.GetBiases());
    auto output_conv_layer =
        BuildActivationLayer(network, initial_conv_layer->getOutput(0), weights_->default_act);
    auto output_conv = output_conv_layer->getOutput(0);

    for (int i = 0; i < weights_->residual_blocks; ++i) {
        auto tower_ptr = weights_->tower[i].get();
        if (auto* block = tower_ptr->Cast<ResidualBlock>()) {
            output_conv = BuildResidualBlock(network, output_conv, block)->getOutput(0);
        } else if (auto* block = tower_ptr->Cast<BottleneckBlock>()) {
            output_conv = BuildBottleneckBlock(network, output_conv, block)->getOutput(0);
        } else if (auto* block = tower_ptr->Cast<NestedBottleneckBlock>()) {
            output_conv = BuildNestedBottleneckBlock(network, output_conv, block)->getOutput(0);
        } else if (auto* block = tower_ptr->Cast<MixerBlock>()) {
            output_conv = BuildMixerBlock(network, output_conv, block)->getOutput(0);
        } else if (auto* block = tower_ptr->Cast<TransformerBlock>()) {
            output_conv = BuildTransformerBlock(network, output_conv, block)->getOutput(0);
        } else {
            throw std::runtime_error("Unknown block type for TensorRT backend.");
        }
    }

    BuildPolicyHead(network, output_conv);
    BuildValueHead(network, output_conv);

    LOGGING << "Done constructing TensorRT network.\n";
    return true;
}

nvinfer1::ILayer*
TrtForwardPipe::TrtEngine::BuildResidualBlock(trt::InferPtr<nvinfer1::INetworkDefinition>& network,
                                              nvinfer1::ITensor* input,
                                              ResidualBlock* block) {
    auto first_conv_layer = BuildConvLayer(network,
                                           input,
                                           block->conv1.GetFilter(),
                                           block->conv1.GetInputs(),
                                           block->conv1.GetOutputs(),
                                           block->conv1.GetWeights(),
                                           block->conv1.GetBiases());
    auto first_activation_layer =
        BuildActivationLayer(network, first_conv_layer->getOutput(0), weights_->default_act);

    auto second_conv_layer = BuildConvLayer(network,
                                            first_activation_layer->getOutput(0),
                                            block->conv2.GetFilter(),
                                            block->conv2.GetInputs(),
                                            block->conv2.GetOutputs(),
                                            block->conv2.GetWeights(),
                                            block->conv2.GetBiases());
    auto second_mask_layer = BuildMaskLayer(network, second_conv_layer->getOutput(0));

    if (block->apply_se) {
        return BuildSqueezeExcitationLayer(network, input, second_mask_layer->getOutput(0), block);
    }

    auto merge_layer = network->addElementWise(
        *input, *second_mask_layer->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
    return BuildActivationLayer(network, merge_layer->getOutput(0), weights_->default_act, false);
}

nvinfer1::ILayer* TrtForwardPipe::TrtEngine::BuildBottleneckBlock(
    trt::InferPtr<nvinfer1::INetworkDefinition>& network,
    nvinfer1::ITensor* input,
    BottleneckBlock* block) {
    auto pre_conv_layer = BuildConvLayer(network,
                                         input,
                                         block->pre_btl_conv.GetFilter(),
                                         block->pre_btl_conv.GetInputs(),
                                         block->pre_btl_conv.GetOutputs(),
                                         block->pre_btl_conv.GetWeights(),
                                         block->pre_btl_conv.GetBiases());
    auto pre_activation_layer =
        BuildActivationLayer(network, pre_conv_layer->getOutput(0), weights_->default_act);

    auto first_conv_layer = BuildConvLayer(network,
                                           pre_activation_layer->getOutput(0),
                                           block->conv1.GetFilter(),
                                           block->conv1.GetInputs(),
                                           block->conv1.GetOutputs(),
                                           block->conv1.GetWeights(),
                                           block->conv1.GetBiases());
    auto first_activation_layer =
        BuildActivationLayer(network, first_conv_layer->getOutput(0), weights_->default_act);

    auto second_conv_layer = BuildConvLayer(network,
                                            first_activation_layer->getOutput(0),
                                            block->conv2.GetFilter(),
                                            block->conv2.GetInputs(),
                                            block->conv2.GetOutputs(),
                                            block->conv2.GetWeights(),
                                            block->conv2.GetBiases());
    auto second_activation_layer =
        BuildActivationLayer(network, second_conv_layer->getOutput(0), weights_->default_act);

    auto post_conv_layer = BuildConvLayer(network,
                                          second_activation_layer->getOutput(0),
                                          block->post_btl_conv.GetFilter(),
                                          block->post_btl_conv.GetInputs(),
                                          block->post_btl_conv.GetOutputs(),
                                          block->post_btl_conv.GetWeights(),
                                          block->post_btl_conv.GetBiases());
    auto post_mask_layer = BuildMaskLayer(network, post_conv_layer->getOutput(0));

    if (block->apply_se) {
        return BuildSqueezeExcitationLayer(network, input, post_mask_layer->getOutput(0), block);
    }

    auto merge_layer = network->addElementWise(
        *input, *post_mask_layer->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
    return BuildActivationLayer(network, merge_layer->getOutput(0), weights_->default_act, false);
}

nvinfer1::ILayer* TrtForwardPipe::TrtEngine::BuildNestedBottleneckBlock(
    trt::InferPtr<nvinfer1::INetworkDefinition>& network,
    nvinfer1::ITensor* input,
    NestedBottleneckBlock* block) {
    auto pre_conv_layer = BuildConvLayer(network,
                                         input,
                                         block->pre_btl_conv.GetFilter(),
                                         block->pre_btl_conv.GetInputs(),
                                         block->pre_btl_conv.GetOutputs(),
                                         block->pre_btl_conv.GetWeights(),
                                         block->pre_btl_conv.GetBiases());
    auto pre_activation_layer =
        BuildActivationLayer(network, pre_conv_layer->getOutput(0), weights_->default_act);

    auto first_conv_layer = BuildConvLayer(network,
                                           pre_activation_layer->getOutput(0),
                                           block->conv1.GetFilter(),
                                           block->conv1.GetInputs(),
                                           block->conv1.GetOutputs(),
                                           block->conv1.GetWeights(),
                                           block->conv1.GetBiases());
    auto first_activation_layer =
        BuildActivationLayer(network, first_conv_layer->getOutput(0), weights_->default_act);

    auto second_conv_layer = BuildConvLayer(network,
                                            first_activation_layer->getOutput(0),
                                            block->conv2.GetFilter(),
                                            block->conv2.GetInputs(),
                                            block->conv2.GetOutputs(),
                                            block->conv2.GetWeights(),
                                            block->conv2.GetBiases());
    auto second_mask_layer = BuildMaskLayer(network, second_conv_layer->getOutput(0));
    auto second_merge_layer = network->addElementWise(*second_mask_layer->getOutput(0),
                                                      *pre_activation_layer->getOutput(0),
                                                      nvinfer1::ElementWiseOperation::kSUM);
    auto second_activation_layer =
        BuildActivationLayer(network, second_merge_layer->getOutput(0), weights_->default_act);

    auto third_conv_layer = BuildConvLayer(network,
                                           second_activation_layer->getOutput(0),
                                           block->conv3.GetFilter(),
                                           block->conv3.GetInputs(),
                                           block->conv3.GetOutputs(),
                                           block->conv3.GetWeights(),
                                           block->conv3.GetBiases());
    auto third_activation_layer =
        BuildActivationLayer(network, third_conv_layer->getOutput(0), weights_->default_act);

    auto fourth_conv_layer = BuildConvLayer(network,
                                            third_activation_layer->getOutput(0),
                                            block->conv4.GetFilter(),
                                            block->conv4.GetInputs(),
                                            block->conv4.GetOutputs(),
                                            block->conv4.GetWeights(),
                                            block->conv4.GetBiases());
    auto fourth_mask_layer = BuildMaskLayer(network, fourth_conv_layer->getOutput(0));
    auto fourth_merge_layer = network->addElementWise(*fourth_mask_layer->getOutput(0),
                                                      *second_activation_layer->getOutput(0),
                                                      nvinfer1::ElementWiseOperation::kSUM);
    auto fourth_activation_layer =
        BuildActivationLayer(network, fourth_merge_layer->getOutput(0), weights_->default_act);

    auto post_conv_layer = BuildConvLayer(network,
                                          fourth_activation_layer->getOutput(0),
                                          block->post_btl_conv.GetFilter(),
                                          block->post_btl_conv.GetInputs(),
                                          block->post_btl_conv.GetOutputs(),
                                          block->post_btl_conv.GetWeights(),
                                          block->post_btl_conv.GetBiases());
    auto post_mask_layer = BuildMaskLayer(network, post_conv_layer->getOutput(0));

    if (block->apply_se) {
        return BuildSqueezeExcitationLayer(network, input, post_mask_layer->getOutput(0), block);
    }

    auto merge_layer = network->addElementWise(
        *input, *post_mask_layer->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
    return BuildActivationLayer(network, merge_layer->getOutput(0), weights_->default_act, false);
}

nvinfer1::ILayer*
TrtForwardPipe::TrtEngine::BuildMixerBlock(trt::InferPtr<nvinfer1::INetworkDefinition>& network,
                                           nvinfer1::ITensor* input,
                                           MixerBlock* block) {
    auto dw_conv_layer = BuildConvLayer(network,
                                        input,
                                        block->dw_conv.GetFilter(),
                                        block->dw_conv.GetInputs(),
                                        block->dw_conv.GetOutputs(),
                                        block->dw_conv.GetWeights(),
                                        block->dw_conv.GetBiases(),
                                        true);
    auto dw_activation_layer =
        BuildActivationLayer(network, dw_conv_layer->getOutput(0), weights_->default_act);
    auto dw_merge_layer = network->addElementWise(
        *input, *dw_activation_layer->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);

    auto first_conv_layer = BuildConvLayer(network,
                                           dw_merge_layer->getOutput(0),
                                           block->conv1.GetFilter(),
                                           block->conv1.GetInputs(),
                                           block->conv1.GetOutputs(),
                                           block->conv1.GetWeights(),
                                           block->conv1.GetBiases());
    auto first_activation_layer =
        BuildActivationLayer(network, first_conv_layer->getOutput(0), weights_->default_act);

    auto second_conv_layer = BuildConvLayer(network,
                                            first_activation_layer->getOutput(0),
                                            block->conv2.GetFilter(),
                                            block->conv2.GetInputs(),
                                            block->conv2.GetOutputs(),
                                            block->conv2.GetWeights(),
                                            block->conv2.GetBiases());
    auto second_mask_layer = BuildMaskLayer(network, second_conv_layer->getOutput(0));

    if (block->apply_se) {
        return BuildSqueezeExcitationLayer(
            network, dw_merge_layer->getOutput(0), second_mask_layer->getOutput(0), block);
    }

    auto merge_layer = network->addElementWise(*dw_merge_layer->getOutput(0),
                                               *second_mask_layer->getOutput(0),
                                               nvinfer1::ElementWiseOperation::kSUM);
    return BuildActivationLayer(network, merge_layer->getOutput(0), weights_->default_act, false);
}

nvinfer1::ILayer* TrtForwardPipe::TrtEngine::BuildTransformerBlock(
    trt::InferPtr<nvinfer1::INetworkDefinition>& network,
    nvinfer1::ITensor* input,
    TransformerBlock* block) {
    const auto num_heads = block->num_heads;
    const auto head_dim = block->head_dim;
    const auto attn_channels = num_heads * head_dim;
    const auto num_intersections = board_size_ * board_size_;

    // The self-attention sublayer. Every intersection is a token, so the
    // fullyconnect layers are the 1x1 convolution.
    auto attn_norm_layer = BuildRMSNormLayer(network, input, block->attn_norm.GetGammas());

    // Fold the attention scale into the query projection. The RoPE is linear,
    // so it is the same as scaling the scores.
    const auto scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    auto q_weights = block->q_proj.GetWeights();
    auto q_biases = block->q_proj.GetBiases();
    for (auto& w : q_weights) {
        w *= scale;
    }
    for (auto& b : q_biases) {
        b *= scale;
    }

    auto q_proj_layer = BuildConvLayer(network,
                                       attn_norm_layer->getOutput(0),
                                       1,
                                       block->q_proj.GetInputs(),
                                       block->q_proj.GetOutputs(),
                                       q_weights,
                                       q_biases);
    auto k_proj_layer = BuildConvLayer(network,
                                       attn_norm_layer->getOutput(0),
                                       1,
                                       block->k_proj.GetInputs(),
                                       block->k_proj.GetOutputs(),
                                       block->k_proj.GetWeights(),
                                       block->k_proj.GetBiases());
    auto v_proj_layer = BuildConvLayer(network,
                                       attn_norm_layer->getOutput(0),
                                       1,
                                       block->v_proj.GetInputs(),
                                       block->v_proj.GetOutputs(),
                                       block->v_proj.GetWeights(),
                                       block->v_proj.GetBiases());

    // The queries, keys and values are [N, heads, head_dim, S].
    auto rotary_matrix_layer = BuildRotaryMatrixLayer(network, block->rope.GetFrequencies());
    auto queries_layer = BuildRotaryEmbeddingLayer(network,
                                                   q_proj_layer->getOutput(0),
                                                   rotary_matrix_layer->getOutput(0),
                                                   num_heads,
                                                   head_dim);
    auto keys_layer = BuildRotaryEmbeddingLayer(network,
                                                k_proj_layer->getOutput(0),
                                                rotary_matrix_layer->getOutput(0),
                                                num_heads,
                                                head_dim);
    auto values_layer = network->addShuffle(*v_proj_layer->getOutput(0));
    values_layer->setReshapeDimensions(nvinfer1::Dims4(0, num_heads, head_dim, num_intersections));

    // scores[i, j] = dot(queries[:, i], keys[:, j])
    auto scores_layer = network->addMatrixMultiply(*queries_layer->getOutput(0),
                                                   nvinfer1::MatrixOperation::kTRANSPOSE,
                                                   *keys_layer->getOutput(0),
                                                   nvinfer1::MatrixOperation::kNONE);
    auto scores_mask_layer =
        network->addElementWise(*scores_layer->getOutput(0),
                                *BuildAttentionMaskLayer(network)->getOutput(0),
                                nvinfer1::ElementWiseOperation::kSUM);
    auto softmax_layer = network->addSoftMax(*scores_mask_layer->getOutput(0));
    softmax_layer->setAxes(1U << 3);

    // attention[:, i] = sum_j scores[i, j] * values[:, j]
    auto attention_layer = network->addMatrixMultiply(*values_layer->getOutput(0),
                                                      nvinfer1::MatrixOperation::kNONE,
                                                      *softmax_layer->getOutput(0),
                                                      nvinfer1::MatrixOperation::kTRANSPOSE);
    auto attention_merge_layer = network->addShuffle(*attention_layer->getOutput(0));
    attention_merge_layer->setReshapeDimensions(
        nvinfer1::Dims4(0, attn_channels, board_size_, board_size_));

    auto out_proj_layer = BuildConvLayer(network,
                                         attention_merge_layer->getOutput(0),
                                         1,
                                         block->out_proj.GetInputs(),
                                         block->out_proj.GetOutputs(),
                                         block->out_proj.GetWeights(),
                                         block->out_proj.GetBiases());
    auto out_proj_mask_layer = BuildMaskLayer(network, out_proj_layer->getOutput(0));
    auto attn_merge_layer = network->addElementWise(
        *input, *out_proj_mask_layer->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);

    // The gated feedforward sublayer.
    auto ffn_norm_layer =
        BuildRMSNormLayer(network, attn_merge_layer->getOutput(0), block->ffn_norm.GetGammas());

    auto ffn_fc1_layer = BuildConvLayer(network,
                                        ffn_norm_layer->getOutput(0),
                                        1,
                                        block->ffn_fc1.GetInputs(),
                                        block->ffn_fc1.GetOutputs(),
                                        block->ffn_fc1.GetWeights(),
                                        block->ffn_fc1.GetBiases());
    auto ffn_fc1_activation =
        BuildActivationLayer(network, ffn_fc1_layer->getOutput(0), weights_->default_act, false);
    auto ffn_gate_layer = BuildConvLayer(network,
                                         ffn_norm_layer->getOutput(0),
                                         1,
                                         block->ffn_gate.GetInputs(),
                                         block->ffn_gate.GetOutputs(),
                                         block->ffn_gate.GetWeights(),
                                         block->ffn_gate.GetBiases());
    auto ffn_glu_layer = network->addElementWise(*ffn_fc1_activation->getOutput(0),
                                                 *ffn_gate_layer->getOutput(0),
                                                 nvinfer1::ElementWiseOperation::kPROD);

    auto ffn_fc2_layer = BuildConvLayer(network,
                                        ffn_glu_layer->getOutput(0),
                                        1,
                                        block->ffn_fc2.GetInputs(),
                                        block->ffn_fc2.GetOutputs(),
                                        block->ffn_fc2.GetWeights(),
                                        block->ffn_fc2.GetBiases());
    auto ffn_fc2_mask_layer = BuildMaskLayer(network, ffn_fc2_layer->getOutput(0));
    return network->addElementWise(*attn_merge_layer->getOutput(0),
                                   *ffn_fc2_mask_layer->getOutput(0),
                                   nvinfer1::ElementWiseOperation::kSUM);
}

nvinfer1::ILayer* TrtForwardPipe::TrtEngine::BuildSqueezeExcitationLayer(
    trt::InferPtr<nvinfer1::INetworkDefinition>& network,
    nvinfer1::ITensor* residual,
    nvinfer1::ITensor* input,
    BlockBasic* tower_ptr) {
    auto gpool_layer = BuildGPoolLayer(network, input);

    auto fc1_layer = BuildConvLayer(network,
                                    gpool_layer->getOutput(0),
                                    1,
                                    tower_ptr->squeeze.GetInputs(),
                                    tower_ptr->squeeze.GetOutputs(),
                                    tower_ptr->squeeze.GetWeights(),
                                    tower_ptr->squeeze.GetBiases());
    auto fc1_activation =
        BuildActivationLayer(network, fc1_layer->getOutput(0), weights_->default_act, false);

    auto fc2_layer = BuildConvLayer(network,
                                    fc1_activation->getOutput(0),
                                    1,
                                    tower_ptr->excite.GetInputs(),
                                    tower_ptr->excite.GetOutputs(),
                                    tower_ptr->excite.GetWeights(),
                                    tower_ptr->excite.GetBiases());

    auto gamma_layer = network->addSlice(*fc2_layer->getOutput(0),
                                         nvinfer1::Dims4(0, 0, 0, 0),
                                         nvinfer1::Dims4(0, weights_->residual_channels, 1, 1),
                                         nvinfer1::Dims4(1, 1, 1, 1));
    gamma_layer->setInput(2, *shape_layer_->getOutput(0));

    auto bias_layer = network->addSlice(*fc2_layer->getOutput(0),
                                        nvinfer1::Dims4(0, weights_->residual_channels, 0, 0),
                                        nvinfer1::Dims4(0, weights_->residual_channels, 1, 1),
                                        nvinfer1::Dims4(1, 1, 1, 1));
    bias_layer->setInput(2, *shape_layer_->getOutput(0));

    auto sigmoid_layer =
        network->addActivation(*gamma_layer->getOutput(0), nvinfer1::ActivationType::kSIGMOID);
    auto scale_layer = network->addElementWise(
        *sigmoid_layer->getOutput(0), *input, nvinfer1::ElementWiseOperation::kPROD);
    auto se_layer = network->addElementWise(*scale_layer->getOutput(0),
                                            *bias_layer->getOutput(0),
                                            nvinfer1::ElementWiseOperation::kSUM);
    auto se_mask_layer = BuildMaskLayer(network, se_layer->getOutput(0));
    auto merge_layer = network->addElementWise(
        *se_mask_layer->getOutput(0), *residual, nvinfer1::ElementWiseOperation::kSUM);
    return BuildActivationLayer(network, merge_layer->getOutput(0), weights_->default_act);
}

void TrtForwardPipe::TrtEngine::BuildPolicyHead(
    trt::InferPtr<nvinfer1::INetworkDefinition>& network, nvinfer1::ITensor* input) {
    nvinfer1::ILayer* act_policy_layer = nullptr;

    auto policy_conv_layer = BuildConvLayer(network,
                                            input,
                                            weights_->p_hd_conv.GetFilter(),
                                            weights_->p_hd_conv.GetInputs(),
                                            weights_->p_hd_conv.GetOutputs(),
                                            weights_->p_hd_conv.GetWeights(),
                                            weights_->p_hd_conv.GetBiases());

    if (weights_->policy_head_type == PolicyHeadType::kRepLK) {
        auto pre_act_policy_layer =
            BuildActivationLayer(network, policy_conv_layer->getOutput(0), weights_->default_act);
        auto depthwise_layer = BuildConvLayer(network,
                                              pre_act_policy_layer->getOutput(0),
                                              weights_->p_dw_conv.GetFilter(),
                                              weights_->p_dw_conv.GetInputs(),
                                              weights_->p_dw_conv.GetOutputs(),
                                              weights_->p_dw_conv.GetWeights(),
                                              weights_->p_dw_conv.GetBiases(),
                                              true);
        auto depthwise_act_layer =
            BuildActivationLayer(network, depthwise_layer->getOutput(0), weights_->default_act);
        auto pointwise_layer = BuildConvLayer(network,
                                              depthwise_act_layer->getOutput(0),
                                              weights_->p_pt_conv.GetFilter(),
                                              weights_->p_pt_conv.GetInputs(),
                                              weights_->p_pt_conv.GetOutputs(),
                                              weights_->p_pt_conv.GetWeights(),
                                              weights_->p_pt_conv.GetBiases());
        act_policy_layer =
            BuildActivationLayer(network, pointwise_layer->getOutput(0), weights_->default_act);
    } else {
        act_policy_layer =
            BuildActivationLayer(network, policy_conv_layer->getOutput(0), weights_->default_act);
    }

    auto policy_pool_layer = BuildGPoolLayer(network, act_policy_layer->getOutput(0));
    auto policy_inter_layer = BuildConvLayer(network,
                                             policy_pool_layer->getOutput(0),
                                             1,
                                             weights_->p_inter_fc.GetInputs(),
                                             weights_->p_inter_fc.GetOutputs(),
                                             weights_->p_inter_fc.GetWeights(),
                                             weights_->p_inter_fc.GetBiases());
    auto policy_inter_act = BuildActivationLayer(
        network, policy_inter_layer->getOutput(0), weights_->default_act, false);
    auto policy_bias_layer = network->addElementWise(*act_policy_layer->getOutput(0),
                                                     *policy_inter_act->getOutput(0),
                                                     nvinfer1::ElementWiseOperation::kSUM);
    auto policy_mask_layer = BuildMaskLayer(network, policy_bias_layer->getOutput(0));

    auto prob_conv_layer = BuildConvLayer(network,
                                          policy_mask_layer->getOutput(0),
                                          weights_->prob_conv.GetFilter(),
                                          weights_->prob_conv.GetInputs(),
                                          weights_->prob_conv.GetOutputs(),
                                          weights_->prob_conv.GetWeights(),
                                          weights_->prob_conv.GetBiases());
    auto output_prob = prob_conv_layer->getOutput(0);
    network->markOutput(*output_prob);
    output_prob->setName("output_prob");
    output_prob->setAllowedFormats(1U << static_cast<int>(nvinfer1::TensorFormat::kLINEAR));
    output_prob->setType(nvinfer1::DataType::kFLOAT);

    auto pass_layer = BuildConvLayer(network,
                                     policy_inter_act->getOutput(0),
                                     1,
                                     weights_->pass_fc.GetInputs(),
                                     weights_->pass_fc.GetOutputs(),
                                     weights_->pass_fc.GetWeights(),
                                     weights_->pass_fc.GetBiases());
    auto output_prob_pass = pass_layer->getOutput(0);
    network->markOutput(*output_prob_pass);
    output_prob_pass->setName("output_prob_pass");
    output_prob_pass->setAllowedFormats(1U << static_cast<int>(nvinfer1::TensorFormat::kLINEAR));
    output_prob_pass->setType(nvinfer1::DataType::kFLOAT);
}

void TrtForwardPipe::TrtEngine::BuildValueHead(trt::InferPtr<nvinfer1::INetworkDefinition>& network,
                                               nvinfer1::ITensor* input) {
    auto value_conv_layer = BuildConvLayer(network,
                                           input,
                                           weights_->v_hd_conv.GetFilter(),
                                           weights_->v_hd_conv.GetInputs(),
                                           weights_->v_hd_conv.GetOutputs(),
                                           weights_->v_hd_conv.GetWeights(),
                                           weights_->v_hd_conv.GetBiases());
    auto value_act_layer =
        BuildActivationLayer(network, value_conv_layer->getOutput(0), weights_->default_act);

    auto ownership_layer = BuildConvLayer(network,
                                          value_act_layer->getOutput(0),
                                          weights_->v_ownership.GetFilter(),
                                          weights_->v_ownership.GetInputs(),
                                          weights_->v_ownership.GetOutputs(),
                                          weights_->v_ownership.GetWeights(),
                                          weights_->v_ownership.GetBiases());
    auto output_ownership = ownership_layer->getOutput(0);
    network->markOutput(*output_ownership);
    output_ownership->setName("output_ownership");
    output_ownership->setAllowedFormats(1U << static_cast<int>(nvinfer1::TensorFormat::kLINEAR));
    output_ownership->setType(nvinfer1::DataType::kFLOAT);

    auto value_pool_layer = BuildGPoolLayer(network, value_act_layer->getOutput(0), true);
    auto value_inter_layer = BuildConvLayer(network,
                                            value_pool_layer->getOutput(0),
                                            1,
                                            weights_->v_inter_fc.GetInputs(),
                                            weights_->v_inter_fc.GetOutputs(),
                                            weights_->v_inter_fc.GetWeights(),
                                            weights_->v_inter_fc.GetBiases());
    auto value_inter_act = BuildActivationLayer(
        network, value_inter_layer->getOutput(0), weights_->default_act, false);
    auto value_misc_layer = BuildConvLayer(network,
                                           value_inter_act->getOutput(0),
                                           1,
                                           weights_->v_misc.GetInputs(),
                                           weights_->v_misc.GetOutputs(),
                                           weights_->v_misc.GetWeights(),
                                           weights_->v_misc.GetBiases());
    auto output_val = value_misc_layer->getOutput(0);
    network->markOutput(*output_val);
    output_val->setName("output_val");
    output_val->setAllowedFormats(1U << static_cast<int>(nvinfer1::TensorFormat::kLINEAR));
    output_val->setType(nvinfer1::DataType::kFLOAT);
}

nvinfer1::ILayer*
TrtForwardPipe::TrtEngine::BuildConvLayer(trt::InferPtr<nvinfer1::INetworkDefinition>& network,
                                          nvinfer1::ITensor* input,
                                          unsigned int filter,
                                          unsigned int in_channels,
                                          unsigned int out_channels,
                                          const std::vector<float>& weights,
                                          const std::vector<float>& biases,
                                          bool depth_wise) {
    (void)in_channels;

    const auto data_type = handles_.fp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT;

    void* cuda_weights = nullptr;
    cuda::MallocAndCopy(handles_.fp16, &cuda_weights, weights);
    cuda_weights_op_.push_back(cuda_weights);

    void* cuda_biases = nullptr;
    nvinfer1::Weights bias_blob{data_type, nullptr, 0};
    if (!biases.empty()) {
        cuda::MallocAndCopy(handles_.fp16, &cuda_biases, biases);
        cuda_weights_op_.push_back(cuda_biases);
        bias_blob = nvinfer1::Weights{data_type, cuda_biases, static_cast<int64_t>(biases.size())};
    }

    auto conv_layer =
        network->addConvolutionNd(*input,
                                  out_channels,
                                  {2, {static_cast<int>(filter), static_cast<int>(filter)}},
                                  {data_type, cuda_weights, static_cast<int64_t>(weights.size())},
                                  bias_blob);
    TRT_ASSERT(conv_layer != nullptr);

    if (depth_wise) {
        conv_layer->setNbGroups(out_channels);
    }
    if (filter != 1) {
        conv_layer->setDilationNd({2, {1, 1}});
        conv_layer->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);
    }
    return conv_layer;
}

nvinfer1::ILayer* TrtForwardPipe::TrtEngine::BuildActivationLayer(
    trt::InferPtr<nvinfer1::INetworkDefinition>& network,
    nvinfer1::ITensor* input,
    Activation act,
    bool need_mask) {
    auto* tensor = input;
    if (need_mask) {
        tensor = BuildMaskLayer(network, input)->getOutput(0);
    }

    nvinfer1::ILayer* act_layer = nullptr;
    switch (act) {
        case Activation::kIdentity:
            act_layer = network->addIdentity(*tensor);
            break;
        case Activation::kReLU:
            act_layer = network->addActivation(*tensor, nvinfer1::ActivationType::kRELU);
            break;
        case Activation::kELU:
            act_layer = network->addActivation(*tensor, nvinfer1::ActivationType::kELU);
            break;
        case Activation::kSELU:
            act_layer = network->addActivation(*tensor, nvinfer1::ActivationType::kSELU);
            break;
        case Activation::kGELU:
            act_layer = network->addActivation(*tensor, nvinfer1::ActivationType::kGELU_TANH);
            break;
        case Activation::kMISH: {
            auto softplus_layer =
                network->addActivation(*tensor, nvinfer1::ActivationType::kSOFTPLUS);
            auto tanh_layer = network->addActivation(*softplus_layer->getOutput(0),
                                                     nvinfer1::ActivationType::kTANH);
            act_layer = network->addElementWise(
                *tensor, *tanh_layer->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
            break;
        }
        case Activation::kSwish: {
            auto sigmoid_layer =
                network->addActivation(*tensor, nvinfer1::ActivationType::kSIGMOID);
            act_layer = network->addElementWise(
                *tensor, *sigmoid_layer->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
            break;
        }
        case Activation::kHardSwish: {
            auto hard_sigmoid =
                network->addActivation(*tensor, nvinfer1::ActivationType::kHARD_SIGMOID);
            hard_sigmoid->setAlpha(1.0f / 6.0f);
            hard_sigmoid->setBeta(0.5f);
            act_layer = network->addElementWise(
                *tensor, *hard_sigmoid->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
            break;
        }
        default:
            throw std::runtime_error("Unknown activation type.");
    }

    TRT_ASSERT(act_layer != nullptr);
    return act_layer;
}

nvinfer1::ILayer*
TrtForwardPipe::TrtEngine::BuildGPoolLayer(trt::InferPtr<nvinfer1::INetworkDefinition>& network,
                                           nvinfer1::ITensor* input,
                                           bool is_value_head) {
    auto gpool_sum_layer =
        network->addReduce(*input, nvinfer1::ReduceOperation::kSUM, 1U << 2 | 1U << 3, true);
    auto gpool_mean_layer = network->addElementWise(*gpool_sum_layer->getOutput(0),
                                                    *mask_sum_layer_->getOutput(0),
                                                    nvinfer1::ElementWiseOperation::kDIV);
    auto gpool_mean_scale_layer = network->addElementWise(*gpool_mean_layer->getOutput(0),
                                                          *mask_scale_layer_->getOutput(0),
                                                          nvinfer1::ElementWiseOperation::kPROD);

    nvinfer1::ILayer* third_input_layer = nullptr;
    if (is_value_head) {
        third_input_layer = network->addElementWise(*gpool_mean_layer->getOutput(0),
                                                    *mask_quad_layer_->getOutput(0),
                                                    nvinfer1::ElementWiseOperation::kPROD);
    } else {
        auto shift_weights = std::make_unique<float[]>(1);
        shift_weights[0] = -1.0f;
        auto shift_layer = network->addScale(*input_mask_,
                                             nvinfer1::ScaleMode::kUNIFORM,
                                             {nvinfer1::DataType::kFLOAT, shift_weights.get(), 1},
                                             {nvinfer1::DataType::kFLOAT, nullptr, 0},
                                             {nvinfer1::DataType::kFLOAT, nullptr, 0});
        extra_weights_.push_back(std::move(shift_weights));

        auto scale_weights = std::make_unique<float[]>(1);
        scale_weights[0] = 5000.0f;
        auto scale_layer = network->addScale(*shift_layer->getOutput(0),
                                             nvinfer1::ScaleMode::kUNIFORM,
                                             {nvinfer1::DataType::kFLOAT, nullptr, 0},
                                             {nvinfer1::DataType::kFLOAT, scale_weights.get(), 1},
                                             {nvinfer1::DataType::kFLOAT, nullptr, 0});
        extra_weights_.push_back(std::move(scale_weights));

        auto add_layer = network->addElementWise(
            *input, *scale_layer->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
        third_input_layer = network->addReduce(
            *add_layer->getOutput(0), nvinfer1::ReduceOperation::kMAX, 1U << 2 | 1U << 3, true);
    }

    nvinfer1::ITensor* concat_inputs[] = {gpool_mean_layer->getOutput(0),
                                          gpool_mean_scale_layer->getOutput(0),
                                          third_input_layer->getOutput(0)};
    auto concat_layer = network->addConcatenation(concat_inputs, 3);
    concat_layer->setAxis(1);
    return concat_layer;
}

nvinfer1::ILayer*
TrtForwardPipe::TrtEngine::BuildMaskLayer(trt::InferPtr<nvinfer1::INetworkDefinition>& network,
                                          nvinfer1::ITensor* input) {
    TRT_ASSERT(input_mask_ != nullptr);
    return network->addElementWise(*input, *input_mask_, nvinfer1::ElementWiseOperation::kPROD);
}

nvinfer1::ILayer*
TrtForwardPipe::TrtEngine::BuildRMSNormLayer(trt::InferPtr<nvinfer1::INetworkDefinition>& network,
                                             nvinfer1::ITensor* input,
                                             const std::vector<float>& gammas) {
    const auto channels = gammas.size();

    auto square_layer =
        network->addElementWise(*input, *input, nvinfer1::ElementWiseOperation::kPROD);
    auto mean_layer = network->addReduce(
        *square_layer->getOutput(0), nvinfer1::ReduceOperation::kAVG, 1U << 1, true);

    // (mean + epsilon)^(-0.5)
    auto epsilon_shift = std::make_unique<float[]>(1);
    auto rsqrt_power = std::make_unique<float[]>(1);
    epsilon_shift[0] = 1e-6f;
    rsqrt_power[0] = -0.5f;
    auto rsqrt_layer = network->addScale(*mean_layer->getOutput(0),
                                         nvinfer1::ScaleMode::kUNIFORM,
                                         {nvinfer1::DataType::kFLOAT, epsilon_shift.get(), 1},
                                         {nvinfer1::DataType::kFLOAT, nullptr, 0},
                                         {nvinfer1::DataType::kFLOAT, rsqrt_power.get(), 1});
    extra_weights_.push_back(std::move(epsilon_shift));
    extra_weights_.push_back(std::move(rsqrt_power));

    auto norm_layer = network->addElementWise(
        *input, *rsqrt_layer->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);

    // The squares overflow easily in FP16, so keep the normalization in FP32
    // like the Python and the CUDA implementations.
    if (handles_.fp16) {
        for (auto* layer :
             std::array<nvinfer1::ILayer*, 4>{square_layer, mean_layer, rsqrt_layer, norm_layer}) {
            layer->setPrecision(nvinfer1::DataType::kFLOAT);
            layer->setOutputType(0, nvinfer1::DataType::kFLOAT);
        }
    }

    auto gamma_weights = std::make_unique<float[]>(channels);
    std::copy(std::begin(gammas), std::end(gammas), gamma_weights.get());
    auto gamma_layer = network->addScale(
        *norm_layer->getOutput(0),
        nvinfer1::ScaleMode::kCHANNEL,
        {nvinfer1::DataType::kFLOAT, nullptr, 0},
        {nvinfer1::DataType::kFLOAT, gamma_weights.get(), static_cast<int64_t>(channels)},
        {nvinfer1::DataType::kFLOAT, nullptr, 0});
    extra_weights_.push_back(std::move(gamma_weights));
    return gamma_layer;
}

nvinfer1::ILayer* TrtForwardPipe::TrtEngine::BuildRotaryMatrixLayer(
    trt::InferPtr<nvinfer1::INetworkDefinition>& network, const std::vector<float>& frequencies) {
    const auto num_pairs = static_cast<int>(frequencies.size()) / 2;
    const auto num_intersections = board_size_ * board_size_;
    const auto matrix_size = static_cast<int64_t>(num_pairs) * 4 * num_intersections;

    // One 2x2 rotation matrix per element pair, per intersection. The layout is
    // [1, pairs, 2 (output), 2 (input), S]. The angles only depend on the weights
    // and the board size, so they are computed here.
    auto matrix_weights = std::make_unique<float[]>(matrix_size);
    for (int p = 0; p < num_pairs; ++p) {
        const auto freq_x = frequencies[2 * p + 0];
        const auto freq_y = frequencies[2 * p + 1];
        float* matrix_ptr = matrix_weights.get() + static_cast<int64_t>(p) * 4 * num_intersections;

        for (int idx = 0; idx < num_intersections; ++idx) {
            const auto pos_x = static_cast<float>(idx % board_size_);
            const auto pos_y = static_cast<float>(idx / board_size_);
            const auto angle = pos_x * freq_x + pos_y * freq_y;
            const auto cos_val = std::cos(angle);
            const auto sin_val = std::sin(angle);

            matrix_ptr[0 * num_intersections + idx] = cos_val;
            matrix_ptr[1 * num_intersections + idx] = -sin_val;
            matrix_ptr[2 * num_intersections + idx] = sin_val;
            matrix_ptr[3 * num_intersections + idx] = cos_val;
        }
    }

    auto matrix_layer =
        network->addConstant(nvinfer1::Dims{5, {1, num_pairs, 2, 2, num_intersections}},
                             {nvinfer1::DataType::kFLOAT, matrix_weights.get(), matrix_size});
    extra_weights_.push_back(std::move(matrix_weights));
    return matrix_layer;
}

nvinfer1::ILayer* TrtForwardPipe::TrtEngine::BuildRotaryEmbeddingLayer(
    trt::InferPtr<nvinfer1::INetworkDefinition>& network,
    nvinfer1::ITensor* input,
    nvinfer1::ITensor* matrix,
    int num_heads,
    int head_dim) {
    const auto num_pairs = num_heads * head_dim / 2;
    const auto num_intersections = board_size_ * board_size_;

    // Rotate the adjacent element pairs of the heads by broadcasting them over
    // the rows of the matrices, [N, pairs, 1, 2, S] x [1, pairs, 2, 2, S], then
    // summing the products of each row.
    auto pair_layer = network->addShuffle(*input);
    pair_layer->setReshapeDimensions(nvinfer1::Dims{5, {0, num_pairs, 1, 2, num_intersections}});
    auto product_layer = network->addElementWise(
        *pair_layer->getOutput(0), *matrix, nvinfer1::ElementWiseOperation::kPROD);
    auto rotate_layer = network->addReduce(
        *product_layer->getOutput(0), nvinfer1::ReduceOperation::kSUM, 1U << 3, false);

    auto head_layer = network->addShuffle(*rotate_layer->getOutput(0));
    head_layer->setReshapeDimensions(nvinfer1::Dims4(0, num_heads, head_dim, num_intersections));
    return head_layer;
}

nvinfer1::ILayer* TrtForwardPipe::TrtEngine::BuildAttentionMaskLayer(
    trt::InferPtr<nvinfer1::INetworkDefinition>& network) {
    if (attn_mask_layer_ != nullptr) {
        return attn_mask_layer_;
    }
    TRT_ASSERT(input_mask_ != nullptr);

    // The bias is 0 on the board and the large negative value out of the board,
    // so the softmax ignores the keys out of the board. The value is the same as
    // the max pooling of the global pooling layer.
    auto bias_shift = std::make_unique<float[]>(1);
    auto bias_scale = std::make_unique<float[]>(1);
    bias_shift[0] = -5000.0f;
    bias_scale[0] = 5000.0f;
    auto bias_layer = network->addScale(*input_mask_,
                                        nvinfer1::ScaleMode::kUNIFORM,
                                        {nvinfer1::DataType::kFLOAT, bias_shift.get(), 1},
                                        {nvinfer1::DataType::kFLOAT, bias_scale.get(), 1},
                                        {nvinfer1::DataType::kFLOAT, nullptr, 0});
    extra_weights_.push_back(std::move(bias_shift));
    extra_weights_.push_back(std::move(bias_scale));

    auto keys_layer = network->addShuffle(*bias_layer->getOutput(0));
    keys_layer->setReshapeDimensions(nvinfer1::Dims4(0, 1, 1, board_size_ * board_size_));
    attn_mask_layer_ = keys_layer;
    return attn_mask_layer_;
}

std::vector<OutputResult>
TrtForwardPipe::TrtEngine::BatchForward(const std::vector<InputData>& batch_input) {
    const auto batch_size = static_cast<int>(batch_input.size());
    if (batch_size == 0) {
        return {};
    }

    assert(max_batch_ >= batch_size);
    cuda::SetDevice(handles_.gpu_id);

    const auto max_board_size = board_size_;
    const auto input_channels = weights_->input_channels;
    const auto num_intersections = max_board_size * max_board_size;
    const auto probabilities_channels = weights_->probabilities_channels;
    const auto pass_probability_outputs = weights_->pass_probability_outputs;
    const auto value_misc_outputs = weights_->value_misc_outputs;
    const auto ownership_channels = weights_->ownership_channels;

    auto batch_planes =
        std::vector<float>(static_cast<size_t>(batch_size) * input_channels * num_intersections);
    auto batch_mask = std::vector<float>(static_cast<size_t>(batch_size) * num_intersections, 0.0f);

    for (int b = 0; b < batch_size; ++b) {
        const auto& input = batch_input[b];
        if (input.board_size <= 0) {
            throw std::runtime_error(
                Format("TensorRT backend: invalid board size %d in batch sample %d.\n",
                       input.board_size,
                       b));
        }
        if (input.board_size > max_board_size) {
            throw std::runtime_error(
                Format("TensorRT backend: input board size %d in batch sample %d exceeds "
                       "constructed board size %d. Reconstruct with the largest board size "
                       "before batching mixed boards.\n",
                       input.board_size,
                       b,
                       max_board_size));
        }

        std::copy(std::begin(input.planes),
                  std::begin(input.planes) + input_channels * num_intersections,
                  std::begin(batch_planes) + b * input_channels * num_intersections);

        for (int idx = 0; idx < num_intersections; ++idx) {
            const int x = idx % max_board_size;
            const int y = idx / max_board_size;
            if (x < input.board_size && y < input.board_size) {
                batch_mask[b * num_intersections + idx] = 1.0f;
            }
        }
    }

    auto input_feature_it = context_->buffers_.find("InputFeature");
    auto input_mask_it = context_->buffers_.find("InputMask");
    assert(input_feature_it != context_->buffers_.end());
    assert(input_mask_it != context_->buffers_.end());

    cuda::ReportCUDAErrors(cudaMemcpyAsync(input_feature_it->second,
                                           batch_planes.data(),
                                           static_cast<size_t>(batch_size) * input_channels *
                                               num_intersections * sizeof(float),
                                           cudaMemcpyHostToDevice,
                                           cudaStreamPerThread));
    cuda::ReportCUDAErrors(
        cudaMemcpyAsync(input_mask_it->second,
                        batch_mask.data(),
                        static_cast<size_t>(batch_size) * num_intersections * sizeof(float),
                        cudaMemcpyHostToDevice,
                        cudaStreamPerThread));

    TRT_ASSERT(context_->execution_context_->setInputShape(
        "InputFeature",
        nvinfer1::Dims4(batch_size, input_channels, max_board_size, max_board_size)));
    TRT_ASSERT(context_->execution_context_->setInputShape(
        "InputMask", nvinfer1::Dims4(batch_size, 1, max_board_size, max_board_size)));
    TRT_ASSERT(context_->execution_context_->setInputShape(
        "BatchSize", nvinfer1::Dims4(batch_size, weights_->residual_channels, 1, 1)));
    TRT_ASSERT(context_->execution_context_->allInputDimensionsSpecified());
    TRT_ASSERT(context_->execution_context_->enqueueV3(cudaStreamPerThread));

    auto batch_prob = std::vector<float>(static_cast<size_t>(batch_size) * probabilities_channels *
                                         num_intersections);
    auto batch_prob_pass =
        std::vector<float>(static_cast<size_t>(batch_size) * pass_probability_outputs);
    auto batch_value_misc =
        std::vector<float>(static_cast<size_t>(batch_size) * value_misc_outputs);
    auto batch_ownership = std::vector<float>(static_cast<size_t>(batch_size) * ownership_channels *
                                              num_intersections);

    auto output_prob_it = context_->buffers_.find("output_prob");
    auto output_prob_pass_it = context_->buffers_.find("output_prob_pass");
    auto output_val_it = context_->buffers_.find("output_val");
    auto output_ownership_it = context_->buffers_.find("output_ownership");
    assert(output_prob_it != context_->buffers_.end());
    assert(output_prob_pass_it != context_->buffers_.end());
    assert(output_val_it != context_->buffers_.end());
    assert(output_ownership_it != context_->buffers_.end());

    cuda::ReportCUDAErrors(cudaMemcpyAsync(batch_prob.data(),
                                           output_prob_it->second,
                                           batch_prob.size() * sizeof(float),
                                           cudaMemcpyDeviceToHost,
                                           cudaStreamPerThread));
    cuda::ReportCUDAErrors(cudaMemcpyAsync(batch_prob_pass.data(),
                                           output_prob_pass_it->second,
                                           batch_prob_pass.size() * sizeof(float),
                                           cudaMemcpyDeviceToHost,
                                           cudaStreamPerThread));
    cuda::ReportCUDAErrors(cudaMemcpyAsync(batch_value_misc.data(),
                                           output_val_it->second,
                                           batch_value_misc.size() * sizeof(float),
                                           cudaMemcpyDeviceToHost,
                                           cudaStreamPerThread));
    cuda::ReportCUDAErrors(cudaMemcpyAsync(batch_ownership.data(),
                                           output_ownership_it->second,
                                           batch_ownership.size() * sizeof(float),
                                           cudaMemcpyDeviceToHost,
                                           cudaStreamPerThread));

    cuda::ReportCUDAErrors(cudaStreamSynchronize(cudaStreamPerThread));

    auto batch_output_result = std::vector<OutputResult>(batch_size);
    FillOutputs(batch_prob,
                batch_prob_pass,
                batch_value_misc,
                batch_ownership,
                batch_input,
                batch_output_result);
    return batch_output_result;
}

void TrtForwardPipe::TrtEngine::FillOutputs(const std::vector<float>& batch_prob,
                                            const std::vector<float>& batch_prob_pass,
                                            const std::vector<float>& batch_value_misc,
                                            const std::vector<float>& batch_ownership,
                                            const std::vector<InputData>& batch_input,
                                            std::vector<OutputResult>& batch_output_result) {
    const int batch_size = static_cast<int>(batch_output_result.size());
    const auto num_intersections = board_size_ * board_size_;
    const auto encoder_version = Encoder::GetEncoderVersion(weights_->version);
    const auto probabilities_channels = weights_->probabilities_channels;
    const auto pass_probability_outputs = weights_->pass_probability_outputs;
    const auto value_misc_outputs = weights_->value_misc_outputs;
    const auto ownership_channels = weights_->ownership_channels;

    if (encoder_version == 1) {
        for (int b = 0; b < batch_size; ++b) {
            auto& output_result = batch_output_result[b];
            const auto& input = batch_input[b];
            const int pol_offset = probabilities_channels * num_intersections;
            const int own_offset = ownership_channels * num_intersections;
            for (int idx = 0; idx < num_intersections; ++idx) {
                const int pol_index =
                    b * pol_offset +
                    static_cast<int>(PolicyBufferOffset::kNormal) * num_intersections + idx;
                const int own_index = b * own_offset + idx;
                output_result.probabilities[idx] = batch_prob[pol_index];
                output_result.ownership[idx] = batch_ownership[own_index];
            }
            output_result.pass_probability = batch_prob_pass[b * pass_probability_outputs];

            output_result.wdl[0] = batch_value_misc[b * value_misc_outputs + 0];
            output_result.wdl[1] = batch_value_misc[b * value_misc_outputs + 1];
            output_result.wdl[2] = batch_value_misc[b * value_misc_outputs + 2];
            output_result.stm_winrate = batch_value_misc[b * value_misc_outputs + 3];
            output_result.final_score = batch_value_misc[b * value_misc_outputs + 4];
            output_result.q_error = 0.0f;
            output_result.score_error = 0.0f;

            output_result.offset = PolicyBufferOffset::kNormal;
            output_result.board_size = input.board_size;
            output_result.komi = input.komi;
            output_result.fp16 = handles_.fp16;
        }
    } else if (encoder_version == 2) {
        for (int b = 0; b < batch_size; ++b) {
            auto& output_result = batch_output_result[b];
            const auto& input = batch_input[b];
            const int pol_offset = probabilities_channels * num_intersections;
            const int own_offset = ownership_channels * num_intersections;
            for (int idx = 0; idx < num_intersections; ++idx) {
                const int pol_index =
                    b * pol_offset + static_cast<int>(input.offset) * num_intersections + idx;
                const int own_index = b * own_offset + idx;
                output_result.probabilities[idx] = batch_prob[pol_index];
                output_result.ownership[idx] = batch_ownership[own_index];
            }
            output_result.pass_probability = batch_prob_pass[b * pass_probability_outputs];

            output_result.wdl[0] = batch_value_misc[b * value_misc_outputs + 0];
            output_result.wdl[1] = batch_value_misc[b * value_misc_outputs + 1];
            output_result.wdl[2] = batch_value_misc[b * value_misc_outputs + 2];
            output_result.stm_winrate = batch_value_misc[b * value_misc_outputs + 3];
            output_result.final_score = batch_value_misc[b * value_misc_outputs + 8];
            output_result.q_error = batch_value_misc[b * value_misc_outputs + 13];
            output_result.score_error = batch_value_misc[b * value_misc_outputs + 14];

            output_result.offset = input.offset;
            output_result.board_size = input.board_size;
            output_result.komi = input.komi;
            output_result.fp16 = handles_.fp16;
        }
    }
}

void TrtForwardPipe::TrtEngine::FreeCudaWeights() {
    for (auto* ptr : cuda_weights_op_) {
        if (ptr != nullptr) {
            cuda::ReportCUDAErrors(cudaFree(ptr));
        }
    }
    cuda_weights_op_.clear();
}

void TrtForwardPipe::TrtEngine::Destroy() {
    if (handles_.initialized) {
        cuda::SetDevice(handles_.gpu_id);
    }

    if (context_) {
        for (auto& entry : context_->buffers_) {
            if (entry.second != nullptr) {
                cuda::ReportCUDAErrors(cudaFree(entry.second));
            }
        }
        context_->buffers_.clear();
        context_.reset();
    }

    cuda_engine_.reset();
    runtime_.reset();

    FreeCudaWeights();
    extra_weights_.clear();

    input_mask_ = nullptr;
    mask_sum_layer_ = nullptr;
    mask_scale_layer_ = nullptr;
    mask_quad_layer_ = nullptr;
    shape_layer_ = nullptr;
    attn_mask_layer_ = nullptr;

    handles_.Release();
}

bool TrtForwardPipe::TrtEngine::ReadFileBinary(const std::string& filename, std::string& out) {
    auto in = std::ifstream(filename, std::ios::binary | std::ios::ate);
    if (!in) {
        return false;
    }

    const auto file_size = in.tellg();
    if (file_size < 0) {
        return false;
    }

    out.resize(static_cast<size_t>(file_size));
    in.seekg(0, std::ios::beg);
    in.read(&out[0], file_size);
    return static_cast<bool>(in);
}

bool TrtForwardPipe::TrtEngine::WriteFileBinary(const std::string& filename,
                                                const std::string& data) {
    auto out = std::ofstream(filename, std::ios::binary | std::ios::trunc);
    if (!out) {
        return false;
    }
    out.write(data.data(), static_cast<std::streamsize>(data.size()));
    return static_cast<bool>(out);
}

std::string TrtForwardPipe::TrtEngine::GetDeviceIdent(const char* device_name) {
    auto device_ident = sha256::GetDigest(device_name, std::strlen(device_name)).substr(0, 8);
    std::transform(device_ident.begin(),
                   device_ident.end(),
                   device_ident.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return device_ident;
}

#endif
