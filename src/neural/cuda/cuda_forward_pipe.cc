#ifdef USE_CUDA

#include "neural/cuda/cuda_forward_pipe.h"

#include <sstream>
#include <stdexcept>

#include "neural/cuda/cuda_kernels.h"
#include "neural/encoder.h"
#include "utils/format.h"
#include "utils/log.h"
#include "utils/option.h"

void CudaForwardPipe::Initialize(std::shared_ptr<DNNWeights> weights) {
    LOGGING << cuda::GetBackendInfo();

    dump_gpu_info_ = true;

    auto option = ForwardPipeOption::Get()
                      .SetBoardSize(GetOption<int>("default_boardsize"))
                      .SetBatchSize(GetOption<int>("batch_size"));
    Construct(option, weights);

    BatchForwardPipe::AssignWorkers(nngraphs_.size());
}

OutputResult CudaForwardPipe::Forward(const InputData& input) {
    return BatchForwardPipe::SendQueryAndWait(input);
}

std::vector<OutputResult> CudaForwardPipe::BatchForward(int gpu,
                                                        const std::vector<InputData>& inputs) {
    return nngraphs_[gpu]->BatchForward(inputs);
}

bool CudaForwardPipe::Valid() const {
    return weights_ != nullptr;
}

int CudaForwardPipe::GetNumWorkers() const {
    return static_cast<int>(nngraphs_.size());
}

void CudaForwardPipe::Construct(ForwardPipeOption option, std::shared_ptr<DNNWeights> weights) {
    // Construct the network with parameters (e.g., board_size) and weights.
    // If the current parameters are the same as the new ones, exit the function
    // immediately.
    if (weights) {
        weights_ = weights;
    }
    if (weights_ == nullptr) {
        // Note: Will use dummy backend
        return;
    }

    int board_size = option.IsValidBoardSize() ? option.board_size : board_size_;
    int batch_size = option.IsValidBatchSize() ? option.batch_size : max_batch_per_nn_;
    // Select the matched board size.
    board_size = std::max(board_size, GetOption<int>("fixed_nn_boardsize"));

    if (board_size == 0 || batch_size == 0) {
        LOGGING << "NN board size/batch size should be larger than zero.\n";
        return;
    }

    // Reject the unsupported blocks before allocating any GPU resource.
    for (auto& tower_ptr : weights_->tower) {
        if (!tower_ptr->IsResidualBlock() && !tower_ptr->IsBottleneckBlock() &&
            !tower_ptr->IsNestedBottleneckBlock() && !tower_ptr->IsMixerBlock() &&
            !tower_ptr->IsTransformerBlock()) {
            throw std::runtime_error("Unknown block type for CUDA backend.");
        }
    }

    // The actual inference batch size may be smaller than the maximum batch size
    // supported by the neural network. In this case, we only need to dynamically
    // adjust the batch size used for the current forward pipe.
    BatchForwardPipe::SetForwardingSize(batch_size);

    if (board_size_ == board_size && batch_size <= max_batch_per_nn_) {
        // The current network already supports this configuration. No reinitialization is
        // required.
        return;
    }
    Release();

    board_size_ = board_size;
    max_batch_per_nn_ = batch_size;
    BatchForwardPipe::SetBoardSize(board_size);

    // Dynamically allocates GPU resources for neural network computation.
    // It prioritizes user-specified GPUs, validates them, and if none are
    // specified or valid, it automatically assigns all available CUDA devices.
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
        nngraphs_.emplace_back(std::make_unique<NNGraph>());
    }

    // Construct network graph for each valid GPU.
    for (auto i = size_t{0}; i < gpus_list.size(); ++i) {
        nngraphs_[i]->ConstructGraph(
            dump_gpu_info_, gpus_list[i], max_batch_per_nn_, board_size_, weights_);
    }

    dump_gpu_info_ = false; // don't show the GPU info next time.
}

void CudaForwardPipe::Release() {
    for (auto& g : nngraphs_) {
        g->DestroyGraph();
    }
    nngraphs_.clear();
}

void CudaForwardPipe::Destroy() {
    Release();
    BatchForwardPipe::QuitWorkers();
}

void CudaForwardPipe::NNGraph::ConstructGraph(bool dump_gpu_info,
                                              const int gpu,
                                              const int max_batch_size,
                                              const int board_size,
                                              std::shared_ptr<DNNWeights> weights) {
    if (graph_ != nullptr) {
        return;
    }

    graph_ = std::make_unique<Graph>();
    weights_ = weights;

    cuda::SetDevice(gpu);
    handles_.ApplyOnCurrentDevice();

    SetComputationMode(&handles_);
    if (dump_gpu_info) {
        LOGGING << cuda::GetCurrentDeviceInfo(&handles_);
    }

    board_size_ = board_size;
    scratch_size_ = 0;
    max_batch_ = max_batch_size;

    // Build the graph first.
    const auto input_channels = weights_->input_channels;
    const auto output_channels = weights_->residual_channels;
    const auto default_act = weights_->default_act;

    // input layer
    graph_->input_conv = cuda::Convolution(&handles_,
                                           max_batch_,      // max batch size
                                           board_size_,     // board size
                                           3,               // kernel size
                                           input_channels,  // input channels
                                           output_channels, // output channels
                                           default_act      // activation
    );

    // block tower
    const auto blocks = weights_->residual_blocks;
    for (int i = 0; i < weights_->residual_blocks; ++i) {
        graph_->tower.emplace_back(NNGraph::Block{});
    }

    auto peaks = PeakSizes{};
    for (int i = 0; i < blocks; ++i) {
        auto& block = graph_->tower[i];
        const auto tower_ptr = weights_->tower[i].get();
        if (tower_ptr->IsResidualBlock()) {
            ConstructResidualBlock(block, tower_ptr, peaks);
        } else if (tower_ptr->IsBottleneckBlock()) {
            ConstructBottleneckBlock(block, tower_ptr, peaks);
        } else if (tower_ptr->IsNestedBottleneckBlock()) {
            ConstructNestedBottleneckBlock(block, tower_ptr, peaks);
        } else if (tower_ptr->IsMixerBlock()) {
            ConstructMixerBlock(block, tower_ptr, peaks);
        } else if (tower_ptr->IsTransformerBlock()) {
            ConstructTransformerBlock(block, tower_ptr, peaks);
        } else {
            throw std::runtime_error("Unknown block type for CUDA backend.");
        }
        if (tower_ptr->apply_se) {
            ConstructSEUnit(block, tower_ptr);
        }
    }

    // policy head
    const auto policy_head_channels = weights_->policy_head_channels;
    const auto probabilities_channels = weights_->probabilities_channels;
    const auto pass_probability_outputs = weights_->pass_probability_outputs;
    graph_->p_hd_conv = cuda::Convolution(&handles_,
                                          max_batch_,           // max batch size
                                          board_size_,          // board size
                                          1,                    // kernel size
                                          output_channels,      // input channels
                                          policy_head_channels, // output channels
                                          default_act           // activation
    );
    if (weights_->policy_head_type == PolicyHeadType::kRepLK) {
        const auto filters = weights_->p_dw_conv.GetFilter();
        graph_->p_dw_conv = cuda::DepthwiseConvolution(&handles_,
                                                       max_batch_,           // max batch size
                                                       board_size_,          // board size
                                                       filters,              // kernel size
                                                       policy_head_channels, // input channels
                                                       default_act           // activation
        );
        graph_->p_pt_conv = cuda::Convolution(&handles_,
                                              max_batch_,           // max batch size
                                              board_size_,          // board size
                                              1,                    // kernel size
                                              policy_head_channels, // input channels
                                              policy_head_channels, // output channels
                                              default_act           // activation
        );
    }
    graph_->p_pool = cuda::GlobalPooling(&handles_,
                                         false,
                                         max_batch_,          // max batch size
                                         board_size_,         // board size
                                         policy_head_channels // input channels
    );
    graph_->p_inter = cuda::FullyConnect(&handles_,
                                         max_batch_,               // max batch size
                                         3 * policy_head_channels, // input sizes
                                         policy_head_channels,     // outpur size
                                         default_act               // activation
    );
    graph_->p_prob = cuda::Convolution(&handles_,
                                       max_batch_,             // max batch size
                                       board_size_,            // board size
                                       1,                      // kernel size
                                       policy_head_channels,   // input channels
                                       probabilities_channels, // output channels
                                       Activation::kIdentity   // activation
    );
    graph_->p_prob_pass = cuda::FullyConnect(&handles_,
                                             max_batch_,               // max batch size
                                             policy_head_channels,     // input sizes
                                             pass_probability_outputs, // outpur size
                                             Activation::kIdentity     // activation
    );

    // value head
    const auto value_head_channels = weights_->value_head_channels;
    const auto ownership_channels = weights_->ownership_channels;
    const auto value_misc_outputs = weights_->value_misc_outputs;
    graph_->v_hd_conv = cuda::Convolution(&handles_,
                                          max_batch_,          // max batch size
                                          board_size_,         // board size
                                          1,                   // kernel size
                                          output_channels,     // input channels
                                          value_head_channels, // output channels
                                          default_act          // activation
    );
    graph_->v_pool = cuda::GlobalPooling(&handles_,
                                         true,
                                         max_batch_,         // max batch size
                                         board_size_,        // board size
                                         value_head_channels // input channels
    );
    graph_->v_inter = cuda::FullyConnect(&handles_,
                                         max_batch_,              // max batch size
                                         3 * value_head_channels, // input sizes
                                         3 * value_head_channels, // outpur size
                                         default_act              // activation
    );
    graph_->v_ownership = cuda::Convolution(&handles_,
                                            max_batch_,           // max batch size
                                            board_size_,          // board size
                                            1,                    // kernel size
                                            value_head_channels,  // input channels
                                            ownership_channels,   // output channels
                                            Activation::kIdentity // activation
    );
    graph_->v_misc = cuda::FullyConnect(&handles_,
                                        max_batch_,              // max batch size
                                        3 * value_head_channels, // input size
                                        value_misc_outputs,      // output size
                                        Activation::kIdentity    // relu
    );

    // Now push the weights.

    const bool winograd = weights_->winograd;

    // input layer
    graph_->input_conv.LoadWeights(weights_->input_conv.GetWeights(),
                                   weights_->input_conv.GetBiases(),
                                   scratch_size_,
                                   winograd);

    // policy head
    graph_->p_hd_conv.LoadWeights(
        weights->p_hd_conv.GetWeights(), weights->p_hd_conv.GetBiases(), scratch_size_, winograd);

    if (weights_->policy_head_type == PolicyHeadType::kRepLK) {
        graph_->p_dw_conv.LoadWeights(weights->p_dw_conv.GetWeights(),
                                      weights->p_dw_conv.GetBiases());
        graph_->p_pt_conv.LoadWeights(weights->p_pt_conv.GetWeights(),
                                      weights->p_pt_conv.GetBiases(),
                                      scratch_size_,
                                      winograd);
    }

    graph_->p_inter.LoadWeights(weights_->p_inter_fc.GetWeights(),
                                weights_->p_inter_fc.GetBiases());

    graph_->p_prob.LoadWeights(
        weights->prob_conv.GetWeights(), weights_->prob_conv.GetBiases(), scratch_size_, winograd);

    graph_->p_prob_pass.LoadWeights(weights_->pass_fc.GetWeights(), weights_->pass_fc.GetBiases());

    // value head
    graph_->v_hd_conv.LoadWeights(
        weights->v_hd_conv.GetWeights(), weights->v_hd_conv.GetBiases(), scratch_size_, winograd);

    graph_->v_inter.LoadWeights(weights_->v_inter_fc.GetWeights(),
                                weights_->v_inter_fc.GetBiases());

    graph_->v_ownership.LoadWeights(weights->v_ownership.GetWeights(),
                                    weights_->v_ownership.GetBiases(),
                                    scratch_size_,
                                    winograd);

    graph_->v_misc.LoadWeights(weights_->v_misc.GetWeights(), weights_->v_misc.GetBiases());

    // Allocate all buffers.
    const size_t factor = max_batch_ * cuda::GetCudaTypeSize(handles_.fp16);
    const size_t num_intersections = board_size_ * board_size_;

    const size_t planes_size = factor * input_channels * num_intersections;
    const size_t spatia_size = factor * num_intersections;
    const size_t pol_size = spatia_size * probabilities_channels;
    const size_t pass_size = factor * pass_probability_outputs;
    const size_t val_size = factor * value_misc_outputs;
    const size_t ownership_size = spatia_size * ownership_channels;

    const size_t conv_op_size = factor * peaks.channels * num_intersections;

    const size_t pol_op1_size = factor * policy_head_channels * num_intersections;
    // pol_op2 may be for RepLK head
    const size_t pol_op2_size = factor * policy_head_channels * num_intersections;
    const size_t pol_op3_size = factor * policy_head_channels;

    const size_t val_op1_size = factor * value_head_channels * num_intersections;
    const size_t val_op2_size = factor * value_head_channels * 3;
    const size_t val_op3_size = factor * value_head_channels * 3;

    const size_t mask_op1_size = factor * num_intersections;
    const size_t mask_op2_size = factor;

    cuda::ReportCUDAErrors(cudaMalloc(&cuda_scratch_op_[0], scratch_size_));
    cuda::ReportCUDAErrors(cudaMalloc(&cuda_scratch_op_[1], scratch_size_));

    cuda::ReportCUDAErrors(cudaMalloc(&cuda_conv_op_[0], conv_op_size));
    cuda::ReportCUDAErrors(cudaMalloc(&cuda_conv_op_[1], conv_op_size));
    cuda::ReportCUDAErrors(cudaMalloc(&cuda_conv_op_[2], conv_op_size));
    cuda::ReportCUDAErrors(cudaMalloc(&cuda_conv_op_[3], conv_op_size));

    cuda::ReportCUDAErrors(cudaMalloc(&cuda_pol_op_[0], pol_op1_size));
    cuda::ReportCUDAErrors(cudaMalloc(&cuda_pol_op_[1], pol_op2_size));
    cuda::ReportCUDAErrors(cudaMalloc(&cuda_pol_op_[2], pol_op3_size));

    cuda::ReportCUDAErrors(cudaMalloc(&cuda_val_op_[0], val_op1_size));
    cuda::ReportCUDAErrors(cudaMalloc(&cuda_val_op_[1], val_op2_size));
    cuda::ReportCUDAErrors(cudaMalloc(&cuda_val_op_[2], val_op3_size));

    cuda::ReportCUDAErrors(cudaMalloc(&cuda_mask_op_[0], mask_op1_size));
    cuda::ReportCUDAErrors(cudaMalloc(&cuda_mask_op_[1], mask_op2_size));

    cuda_attn_op_ = {nullptr, nullptr};
    if (peaks.attn_channels > 0) {
        const size_t attn_values_size = factor * peaks.attn_channels * num_intersections;
        const size_t attn_scores_size =
            factor * peaks.attn_heads * num_intersections * num_intersections;

        cuda::ReportCUDAErrors(cudaMalloc(&cuda_attn_op_[0], attn_values_size));
        cuda::ReportCUDAErrors(cudaMalloc(&cuda_attn_op_[1], attn_scores_size));
    }

    cuda::ReportCUDAErrors(cudaMalloc(&cuda_input_planes_, planes_size));
    cuda::ReportCUDAErrors(cudaMalloc(&cuda_output_prob_pass_, pass_size));
    cuda::ReportCUDAErrors(cudaMalloc(&cuda_output_prob_, pol_size));
    cuda::ReportCUDAErrors(cudaMalloc(&cuda_output_ownership_, ownership_size));
    cuda::ReportCUDAErrors(cudaMalloc(&cuda_output_val_, val_size));

    // Locked-page memory.
    cuda::ReportCUDAErrors(cudaMallocHost(&host_mask_op_[0], mask_op1_size));
    cuda::ReportCUDAErrors(cudaMallocHost(&host_mask_op_[1], mask_op2_size));

    cuda::ReportCUDAErrors(cudaMallocHost(&host_input_planes_, planes_size));
    cuda::ReportCUDAErrors(cudaMallocHost(&host_output_prob_pass_, pass_size));
    cuda::ReportCUDAErrors(cudaMallocHost(&host_output_prob_, pol_size));
    cuda::ReportCUDAErrors(cudaMallocHost(&host_output_ownership_, ownership_size));
    cuda::ReportCUDAErrors(cudaMallocHost(&host_output_val_, val_size));
}

void CudaForwardPipe::NNGraph::ConstructResidualBlock(Block& block,
                                                      BlockBasic* tower_ptr,
                                                      PeakSizes& peaks) {
    const auto default_act = weights_->default_act;
    const bool winograd = weights_->winograd;

    const auto channels = weights_->residual_channels;
    const auto last_act = tower_ptr->apply_se ? Activation::kIdentity : default_act;

    block.conv1 = cuda::Convolution(&handles_,
                                    max_batch_,  // max batch size
                                    board_size_, // board size
                                    3,           // kernel size
                                    channels,    // input channels
                                    channels,    // output channels
                                    default_act  // activation
    );
    block.conv2 = cuda::Convolution(&handles_,
                                    max_batch_,  // max batch size
                                    board_size_, // board size
                                    3,           // kernel size
                                    channels,    // input channels
                                    channels,    // output channels
                                    last_act     // activation
    );
    peaks.channels = std::max(peaks.channels, channels);

    block.conv1.LoadWeights(
        tower_ptr->conv1.GetWeights(), tower_ptr->conv1.GetBiases(), scratch_size_, winograd);
    block.conv2.LoadWeights(
        tower_ptr->conv2.GetWeights(), tower_ptr->conv2.GetBiases(), scratch_size_, winograd);
}

void CudaForwardPipe::NNGraph::ConstructBottleneckBlock(Block& block,
                                                        BlockBasic* tower_ptr,
                                                        PeakSizes& peaks) {
    const auto default_act = weights_->default_act;
    const bool winograd = weights_->winograd;

    const auto outer_channels = weights_->residual_channels;
    const auto inner_channels = tower_ptr->bottleneck_channels;
    const auto last_act = tower_ptr->apply_se ? Activation::kIdentity : default_act;

    block.pre_btl_conv = cuda::Convolution(&handles_,
                                           max_batch_,     // max batch size
                                           board_size_,    // board size
                                           1,              // kernel size
                                           outer_channels, // input channels
                                           inner_channels, // output channels
                                           default_act     // activation
    );
    block.conv1 = cuda::Convolution(&handles_,
                                    max_batch_,     // max batch size
                                    board_size_,    // board size
                                    3,              // kernel size
                                    inner_channels, // input channels
                                    inner_channels, // output channels
                                    default_act     // activation
    );
    block.conv2 = cuda::Convolution(&handles_,
                                    max_batch_,     // max batch size
                                    board_size_,    // board size
                                    3,              // kernel size
                                    inner_channels, // input channels
                                    inner_channels, // output channels
                                    default_act     // activation
    );
    block.post_btl_conv = cuda::Convolution(&handles_,
                                            max_batch_,     // max batch size
                                            board_size_,    // board size
                                            1,              // kernel size
                                            inner_channels, // input channels
                                            outer_channels, // output channels
                                            last_act        // activation
    );
    peaks.channels = std::max({peaks.channels, inner_channels, outer_channels});

    block.pre_btl_conv.LoadWeights(tower_ptr->pre_btl_conv.GetWeights(),
                                   tower_ptr->pre_btl_conv.GetBiases(),
                                   scratch_size_,
                                   winograd);
    block.conv1.LoadWeights(
        tower_ptr->conv1.GetWeights(), tower_ptr->conv1.GetBiases(), scratch_size_, winograd);
    block.conv2.LoadWeights(
        tower_ptr->conv2.GetWeights(), tower_ptr->conv2.GetBiases(), scratch_size_, winograd);
    block.post_btl_conv.LoadWeights(tower_ptr->post_btl_conv.GetWeights(),
                                    tower_ptr->post_btl_conv.GetBiases(),
                                    scratch_size_,
                                    winograd);
}

void CudaForwardPipe::NNGraph::ConstructNestedBottleneckBlock(Block& block,
                                                              BlockBasic* tower_ptr,
                                                              PeakSizes& peaks) {
    const auto default_act = weights_->default_act;
    const bool winograd = weights_->winograd;

    const auto outer_channels = weights_->residual_channels;
    const auto inner_channels = tower_ptr->bottleneck_channels;
    const auto last_act = tower_ptr->apply_se ? Activation::kIdentity : default_act;

    block.pre_btl_conv = cuda::Convolution(&handles_,
                                           max_batch_,     // max batch size
                                           board_size_,    // board size
                                           1,              // kernel size
                                           outer_channels, // input channels
                                           inner_channels, // output channels
                                           default_act     // activation
    );
    block.conv1 = cuda::Convolution(&handles_,
                                    max_batch_,     // max batch size
                                    board_size_,    // board size
                                    3,              // kernel size
                                    inner_channels, // input channels
                                    inner_channels, // output channels
                                    default_act     // activation
    );
    block.conv2 = cuda::Convolution(&handles_,
                                    max_batch_,     // max batch size
                                    board_size_,    // board size
                                    3,              // kernel size
                                    inner_channels, // input channels
                                    inner_channels, // output channels
                                    default_act     // activation
    );
    block.conv3 = cuda::Convolution(&handles_,
                                    max_batch_,     // max batch size
                                    board_size_,    // board size
                                    3,              // kernel size
                                    inner_channels, // input channels
                                    inner_channels, // output channels
                                    default_act     // activation
    );
    block.conv4 = cuda::Convolution(&handles_,
                                    max_batch_,     // max batch size
                                    board_size_,    // board size
                                    3,              // kernel size
                                    inner_channels, // input channels
                                    inner_channels, // output channels
                                    default_act     // activation
    );
    block.post_btl_conv = cuda::Convolution(&handles_,
                                            max_batch_,     // max batch size
                                            board_size_,    // board size
                                            1,              // kernel size
                                            inner_channels, // input channels
                                            outer_channels, // output channels
                                            last_act        // activation
    );
    peaks.channels = std::max({peaks.channels, inner_channels, outer_channels});

    block.pre_btl_conv.LoadWeights(tower_ptr->pre_btl_conv.GetWeights(),
                                   tower_ptr->pre_btl_conv.GetBiases(),
                                   scratch_size_,
                                   winograd);
    block.conv1.LoadWeights(
        tower_ptr->conv1.GetWeights(), tower_ptr->conv1.GetBiases(), scratch_size_, winograd);
    block.conv2.LoadWeights(
        tower_ptr->conv2.GetWeights(), tower_ptr->conv2.GetBiases(), scratch_size_, winograd);
    block.conv3.LoadWeights(
        tower_ptr->conv3.GetWeights(), tower_ptr->conv3.GetBiases(), scratch_size_, winograd);
    block.conv4.LoadWeights(
        tower_ptr->conv4.GetWeights(), tower_ptr->conv4.GetBiases(), scratch_size_, winograd);
    block.post_btl_conv.LoadWeights(tower_ptr->post_btl_conv.GetWeights(),
                                    tower_ptr->post_btl_conv.GetBiases(),
                                    scratch_size_,
                                    winograd);
}

void CudaForwardPipe::NNGraph::ConstructMixerBlock(Block& block,
                                                   BlockBasic* tower_ptr,
                                                   PeakSizes& peaks) {
    const auto default_act = weights_->default_act;
    const bool winograd = weights_->winograd;

    const auto channels = weights_->residual_channels;
    const auto feedforwards = tower_ptr->feedforward_channels;
    const auto filters = tower_ptr->dw_conv.GetFilter();
    const auto last_act = tower_ptr->apply_se ? Activation::kIdentity : default_act;

    block.dw_conv = cuda::DepthwiseConvolution(&handles_,
                                               max_batch_,  // max batch size
                                               board_size_, // board size
                                               filters,     // kernel size
                                               channels,    // input channels
                                               default_act  // activation
    );
    block.conv1 = cuda::Convolution(&handles_,
                                    max_batch_,   // max batch size
                                    board_size_,  // board size
                                    1,            // kernel size
                                    channels,     // input channels
                                    feedforwards, // output channels
                                    default_act   // activation
    );
    block.conv2 = cuda::Convolution(&handles_,
                                    max_batch_,   // max batch size
                                    board_size_,  // board size
                                    1,            // kernel size
                                    feedforwards, // input channels
                                    channels,     // output channels
                                    last_act      // activation
    );
    peaks.channels = std::max({peaks.channels, channels, feedforwards});

    block.dw_conv.LoadWeights(tower_ptr->dw_conv.GetWeights(), tower_ptr->dw_conv.GetBiases());
    block.conv1.LoadWeights(
        tower_ptr->conv1.GetWeights(), tower_ptr->conv1.GetBiases(), scratch_size_, winograd);
    block.conv2.LoadWeights(
        tower_ptr->conv2.GetWeights(), tower_ptr->conv2.GetBiases(), scratch_size_, winograd);
}

void CudaForwardPipe::NNGraph::ConstructTransformerBlock(Block& block,
                                                         BlockBasic* tower_ptr,
                                                         PeakSizes& peaks) {
    const auto default_act = weights_->default_act;
    const bool winograd = weights_->winograd;

    const auto channels = weights_->residual_channels;
    const auto num_heads = tower_ptr->num_heads;
    const auto head_dim = tower_ptr->head_dim;
    const auto attn_channels = num_heads * head_dim;
    const auto feedforwards = tower_ptr->feedforward_channels;

    block.attn_norm = cuda::RMSNorm(&handles_,
                                    max_batch_,  // max batch size
                                    board_size_, // board size
                                    channels     // channels
    );
    block.q_proj = cuda::Convolution(&handles_,
                                     max_batch_,           // max batch size
                                     board_size_,          // board size
                                     1,                    // kernel size
                                     channels,             // input channels
                                     attn_channels,        // output channels
                                     Activation::kIdentity // activation
    );
    block.k_proj = cuda::Convolution(&handles_,
                                     max_batch_,           // max batch size
                                     board_size_,          // board size
                                     1,                    // kernel size
                                     channels,             // input channels
                                     attn_channels,        // output channels
                                     Activation::kIdentity // activation
    );
    block.v_proj = cuda::Convolution(&handles_,
                                     max_batch_,           // max batch size
                                     board_size_,          // board size
                                     1,                    // kernel size
                                     channels,             // input channels
                                     attn_channels,        // output channels
                                     Activation::kIdentity // activation
    );
    block.attention = cuda::MultiHeadAttention(&handles_,
                                               max_batch_,  // max batch size
                                               board_size_, // board size
                                               num_heads,   // number of heads
                                               head_dim     // head size
    );
    block.out_proj = cuda::Convolution(&handles_,
                                       max_batch_,           // max batch size
                                       board_size_,          // board size
                                       1,                    // kernel size
                                       attn_channels,        // input channels
                                       channels,             // output channels
                                       Activation::kIdentity // activation
    );
    block.ffn_norm = cuda::RMSNorm(&handles_,
                                   max_batch_,  // max batch size
                                   board_size_, // board size
                                   channels     // channels
    );
    block.ffn_fc1 = cuda::Convolution(&handles_,
                                      max_batch_,   // max batch size
                                      board_size_,  // board size
                                      1,            // kernel size
                                      channels,     // input channels
                                      feedforwards, // output channels
                                      default_act   // activation
    );
    block.ffn_gate = cuda::Convolution(&handles_,
                                       max_batch_,           // max batch size
                                       board_size_,          // board size
                                       1,                    // kernel size
                                       channels,             // input channels
                                       feedforwards,         // output channels
                                       Activation::kIdentity // activation
    );
    block.ffn_fc2 = cuda::Convolution(&handles_,
                                      max_batch_,           // max batch size
                                      board_size_,          // board size
                                      1,                    // kernel size
                                      feedforwards,         // input channels
                                      channels,             // output channels
                                      Activation::kIdentity // activation
    );
    peaks.channels = std::max({peaks.channels, channels, attn_channels, feedforwards});
    peaks.attn_channels = std::max(peaks.attn_channels, attn_channels);
    peaks.attn_heads = std::max(peaks.attn_heads, num_heads);

    block.attn_norm.LoadWeights(tower_ptr->attn_norm.GetGammas());
    block.q_proj.LoadWeights(
        tower_ptr->q_proj.GetWeights(), tower_ptr->q_proj.GetBiases(), scratch_size_, winograd);
    block.k_proj.LoadWeights(
        tower_ptr->k_proj.GetWeights(), tower_ptr->k_proj.GetBiases(), scratch_size_, winograd);
    block.v_proj.LoadWeights(
        tower_ptr->v_proj.GetWeights(), tower_ptr->v_proj.GetBiases(), scratch_size_, winograd);
    block.attention.LoadWeights(tower_ptr->rope.GetFrequencies());
    block.out_proj.LoadWeights(
        tower_ptr->out_proj.GetWeights(), tower_ptr->out_proj.GetBiases(), scratch_size_, winograd);
    block.ffn_norm.LoadWeights(tower_ptr->ffn_norm.GetGammas());
    block.ffn_fc1.LoadWeights(
        tower_ptr->ffn_fc1.GetWeights(), tower_ptr->ffn_fc1.GetBiases(), scratch_size_, winograd);
    block.ffn_gate.LoadWeights(
        tower_ptr->ffn_gate.GetWeights(), tower_ptr->ffn_gate.GetBiases(), scratch_size_, winograd);
    block.ffn_fc2.LoadWeights(
        tower_ptr->ffn_fc2.GetWeights(), tower_ptr->ffn_fc2.GetBiases(), scratch_size_, winograd);
}

void CudaForwardPipe::NNGraph::ConstructSEUnit(Block& block, BlockBasic* tower_ptr) {
    const auto default_act = weights_->default_act;

    const auto channels = weights_->residual_channels;
    const size_t se_size = tower_ptr->se_size;

    block.se_module = cuda::SEUnit(&handles_,
                                   max_batch_,  // max batch size
                                   board_size_, // board size
                                   channels,    // channels
                                   se_size,     // SE size
                                   default_act  // activation
    );

    block.se_module.LoadWeights(tower_ptr->squeeze.GetWeights(),
                                tower_ptr->squeeze.GetBiases(),
                                tower_ptr->excite.GetWeights(),
                                tower_ptr->excite.GetBiases());
}

void CudaForwardPipe::NNGraph::SetComputationMode(cuda::CudaHandles* handles) {
    cudaDeviceProp dev_prop = cuda::GetDeviceProp();

    if (dev_prop.major <= 6 || !GetOption<bool>("fp16")) {
        // The compute capability is too low. The 5 is Maxwell,
        // such as GTX 980 Ti. The 6 is Pascal, such as GTX 1080 Ti.
        // As fair as I know, the FP16 can work on these devices,
        // but the their performance are bad. The FP32 is better
        // choice. So disable the FP16 computation.
        handles->fp16 = false;
    }

    if (!(handles->fp16)) {
        handles->has_tensor_cores = false;
    }

    if (handles->has_tensor_cores) {
        cuda::ReportCUBLASErrors(cublasSetMathMode(handles->cublas_handle, CUBLAS_TENSOR_OP_MATH));
    }
}

bool CudaForwardPipe::NNGraph::ApplyMask(const std::vector<InputData>& batch_input) {
    const int batch_size = batch_input.size();
    if (batch_size == 0) {
        return false;
    }

    const int num_intersections = board_size_ * board_size_;
    bool should_apply_mask = false;

    for (int b = 0; b < batch_size; ++b) {
        if (board_size_ != batch_input[b].board_size) {
            should_apply_mask = true;
            break;
        }
    }

    if (should_apply_mask) {
        // There are at least two different board size
        // inputs planes. We should do the mask operation
        // for each NN layers.
        auto spat_mask = std::vector<float>(batch_size * num_intersections);
        auto sqrt_mask = std::vector<float>(batch_size);

        for (int b = 0; b < batch_size; ++b) {
            const int planes_bsize = batch_input[b].board_size;
            for (int idx = 0; idx < num_intersections; ++idx) {
                const int x = idx % board_size_;
                const int y = idx / board_size_;
                if (x < planes_bsize && y < planes_bsize) {
                    spat_mask[b * num_intersections + idx] = 1.f;
                } else {
                    spat_mask[b * num_intersections + idx] = 0.f;
                }
            }
            sqrt_mask[b] = planes_bsize;
        }

        // Copy masks on this graph's stream, so each worker thread/GPU
        // can run independently without a global lock.
        cuda::CopyToCudaOpAsync(
            handles_.fp16, &(cuda_mask_op_[0]), spat_mask, handles_.stream, &(host_mask_op_[0]));
        cuda::CopyToCudaOpAsync(
            handles_.fp16, &(cuda_mask_op_[1]), sqrt_mask, handles_.stream, &(host_mask_op_[1]));
    }

    return should_apply_mask;
}

std::vector<OutputResult>
CudaForwardPipe::NNGraph::BatchForward(const std::vector<InputData>& batch_input) {
    const auto batch_size = (int)batch_input.size();

    assert(max_batch_ >= batch_size);
    cuda::SetDevice(handles_.gpu_id);

    const auto should_apply_mask = ApplyMask(batch_input);
    const auto input_channels = weights_->input_channels;
    const auto num_intersections = board_size_ * board_size_;
    auto batch_planes = std::vector<float>(batch_size * input_channels * num_intersections);

    for (int b = 0; b < batch_size; ++b) {
        const auto& input = batch_input[b];
        for (int idx = 0; idx < input_channels * num_intersections; ++idx) {
            batch_planes[b * input_channels * num_intersections + idx] = input.planes[idx];
        }
    }

    std::array<void*, 2> mask_buf = cuda_mask_op_;
    if (!should_apply_mask) {
        // Disable the mask.
        mask_buf[0] = mask_buf[1] = nullptr;
    }

    // copy the inputs data from host memory to device
    cuda::CopyToCudaOpAsync(
        handles_.fp16, &cuda_input_planes_, batch_planes, handles_.stream, &host_input_planes_);

    // input layer
    graph_->input_conv.Forward(batch_size,
                               cuda_conv_op_[0],
                               cuda_input_planes_,
                               nullptr,
                               mask_buf[0],
                               cuda_scratch_op_[0],
                               cuda_scratch_op_[1],
                               scratch_size_);

    // block tower
    const auto blocks = weights_->residual_blocks;
    for (int i = 0; i < blocks; ++i) {
        // cuda_conv_op_[0] is input/residual.
        // cuda_conv_op_[1 ~ 2] are buffers.
        // cuda_conv_op_[3] is output before SE module.

        auto& block = graph_->tower[i];
        const auto tower_ptr = weights_->tower[i].get();
        if (tower_ptr->IsResidualBlock()) {
            ResidualBlockForward(batch_size, block, tower_ptr, mask_buf[0]);
        } else if (tower_ptr->IsBottleneckBlock()) {
            BottleneckBlockForward(batch_size, block, tower_ptr, mask_buf[0]);
        } else if (tower_ptr->IsNestedBottleneckBlock()) {
            NestedBottleneckBlockForward(batch_size, block, tower_ptr, mask_buf[0]);
        } else if (tower_ptr->IsMixerBlock()) {
            MixerBlockForward(batch_size, block, tower_ptr, mask_buf[0]);
        } else if (tower_ptr->IsTransformerBlock()) {
            TransformerBlockForward(batch_size, block, tower_ptr, mask_buf[0]);
        }

        bool module_skip = false;
        if (tower_ptr->apply_se) {
            // squeeze-and-excitation module
            void* se_skip = cuda_conv_op_[0];
            void* se_outs = cuda_conv_op_[0];

            block.se_module.Forward(
                batch_size, se_outs, cuda_conv_op_[3], se_skip, mask_buf[0], mask_buf[1]);
            module_skip = true;
        }

        if (!module_skip) {
            std::swap(cuda_conv_op_[3], cuda_conv_op_[0]);
        }
    }

    // policy head
    const auto policy_head_channels = weights_->policy_head_channels;

    graph_->p_hd_conv.Forward(batch_size,
                              cuda_pol_op_[0],
                              cuda_conv_op_[0],
                              nullptr,
                              mask_buf[0],
                              cuda_scratch_op_[0],
                              cuda_scratch_op_[1],
                              scratch_size_);

    if (weights_->policy_head_type == PolicyHeadType::kRepLK) {
        graph_->p_dw_conv.Forward(
            batch_size, cuda_pol_op_[1], cuda_pol_op_[0], nullptr, mask_buf[0]);
        graph_->p_pt_conv.Forward(batch_size,
                                  cuda_pol_op_[0],
                                  cuda_pol_op_[1],
                                  nullptr,
                                  mask_buf[0],
                                  cuda_scratch_op_[0],
                                  cuda_scratch_op_[1],
                                  scratch_size_);
    }

    graph_->p_pool.Forward(batch_size, cuda_pol_op_[1], cuda_pol_op_[0], mask_buf[0], mask_buf[1]);

    graph_->p_inter.Forward(batch_size, cuda_pol_op_[2], cuda_pol_op_[1]);

    void* null_op = nullptr;
    cuda::AddSpatial(handles_.fp16,
                     cuda_pol_op_[0],
                     cuda_pol_op_[2],
                     null_op,
                     mask_buf[0],
                     batch_size * policy_head_channels,
                     batch_size,
                     policy_head_channels,
                     num_intersections,
                     Activation::kIdentity,
                     handles_.stream);

    graph_->p_prob.Forward(batch_size,
                           cuda_output_prob_,
                           cuda_pol_op_[0],
                           nullptr,
                           nullptr,
                           cuda_scratch_op_[0],
                           cuda_scratch_op_[1],
                           scratch_size_);

    graph_->p_prob_pass.Forward(batch_size, cuda_output_prob_pass_, cuda_pol_op_[2]);

    // value head
    graph_->v_hd_conv.Forward(batch_size,
                              cuda_val_op_[0],
                              cuda_conv_op_[0],
                              nullptr,
                              mask_buf[0],
                              cuda_scratch_op_[0],
                              cuda_scratch_op_[1],
                              scratch_size_);

    graph_->v_pool.Forward(batch_size, cuda_val_op_[1], cuda_val_op_[0], mask_buf[0], mask_buf[1]);

    graph_->v_inter.Forward(batch_size, cuda_val_op_[2], cuda_val_op_[1]);

    graph_->v_ownership.Forward(batch_size,
                                cuda_output_ownership_,
                                cuda_val_op_[0],
                                nullptr,
                                nullptr,
                                cuda_scratch_op_[0],
                                cuda_scratch_op_[1],
                                scratch_size_);

    graph_->v_misc.Forward(batch_size, cuda_output_val_, cuda_val_op_[2]);

    const auto probabilities_channels = weights_->probabilities_channels;
    const auto pass_probability_outputs = weights_->pass_probability_outputs;
    const auto value_misc_outputs = weights_->value_misc_outputs;
    const auto ownership_channels = weights_->ownership_channels;

    auto batch_prob = std::vector<float>(batch_size * probabilities_channels * num_intersections);
    auto batch_prob_pass = std::vector<float>(batch_size * pass_probability_outputs);
    auto batch_value_misc = std::vector<float>(batch_size * value_misc_outputs);
    auto batch_ownership = std::vector<float>(batch_size * ownership_channels * num_intersections);

    // Copy results on this graph's stream, then wait once before decoding.
    cuda::CopyToHostOpAsync(
        handles_.fp16, batch_prob, &cuda_output_prob_, handles_.stream, &host_output_prob_);
    cuda::CopyToHostOpAsync(handles_.fp16,
                            batch_prob_pass,
                            &cuda_output_prob_pass_,
                            handles_.stream,
                            &host_output_prob_pass_);
    cuda::CopyToHostOpAsync(
        handles_.fp16, batch_value_misc, &cuda_output_val_, handles_.stream, &host_output_val_);
    cuda::CopyToHostOpAsync(handles_.fp16,
                            batch_ownership,
                            &cuda_output_ownership_,
                            handles_.stream,
                            &host_output_ownership_);

    auto batch_output_result = std::vector<OutputResult>(batch_size);

    FillOutputs(batch_prob,
                batch_prob_pass,
                batch_value_misc,
                batch_ownership,
                batch_input,
                batch_output_result);
    return batch_output_result;
}

void CudaForwardPipe::NNGraph::ResidualBlockForward(const int batch_size,
                                                    Block& block,
                                                    BlockBasic* tower_ptr,
                                                    void* mask) {
    // 1st conv layer
    block.conv1.Forward(batch_size,
                        cuda_conv_op_[1],
                        cuda_conv_op_[0],
                        nullptr,
                        mask,
                        cuda_scratch_op_[0],
                        cuda_scratch_op_[1],
                        scratch_size_);

    // 2nd conv layer
    void* second_skip = tower_ptr->apply_se ? nullptr : cuda_conv_op_[0];
    block.conv2.Forward(batch_size,
                        cuda_conv_op_[3],
                        cuda_conv_op_[1],
                        second_skip,
                        mask,
                        cuda_scratch_op_[0],
                        cuda_scratch_op_[1],
                        scratch_size_);
}

void CudaForwardPipe::NNGraph::BottleneckBlockForward(const int batch_size,
                                                      Block& block,
                                                      BlockBasic* tower_ptr,
                                                      void* mask) {
    // pre-bottleneck
    block.pre_btl_conv.Forward(batch_size,
                               cuda_conv_op_[1],
                               cuda_conv_op_[0],
                               nullptr,
                               mask,
                               cuda_scratch_op_[0],
                               cuda_scratch_op_[1],
                               scratch_size_);

    // 1st conv layer
    block.conv1.Forward(batch_size,
                        cuda_conv_op_[2],
                        cuda_conv_op_[1],
                        nullptr,
                        mask,
                        cuda_scratch_op_[0],
                        cuda_scratch_op_[1],
                        scratch_size_);

    // 2nd conv layer
    block.conv2.Forward(batch_size,
                        cuda_conv_op_[1],
                        cuda_conv_op_[2],
                        nullptr,
                        mask,
                        cuda_scratch_op_[0],
                        cuda_scratch_op_[1],
                        scratch_size_);

    // post-bottleneck
    void* btl_skip = tower_ptr->apply_se ? nullptr : cuda_conv_op_[0];
    block.post_btl_conv.Forward(batch_size,
                                cuda_conv_op_[3],
                                cuda_conv_op_[1],
                                btl_skip,
                                mask,
                                cuda_scratch_op_[0],
                                cuda_scratch_op_[1],
                                scratch_size_);
}

void CudaForwardPipe::NNGraph::NestedBottleneckBlockForward(const int batch_size,
                                                            Block& block,
                                                            BlockBasic* tower_ptr,
                                                            void* mask) {
    // pre-bottleneck
    block.pre_btl_conv.Forward(batch_size,
                               cuda_conv_op_[1],
                               cuda_conv_op_[0],
                               nullptr,
                               mask,
                               cuda_scratch_op_[0],
                               cuda_scratch_op_[1],
                               scratch_size_);

    // 1st conv layer (1st block)
    block.conv1.Forward(batch_size,
                        cuda_conv_op_[2],
                        cuda_conv_op_[1],
                        nullptr,
                        mask,
                        cuda_scratch_op_[0],
                        cuda_scratch_op_[1],
                        scratch_size_);

    // 2nd conv layer (1st block)
    void* block1_skip = cuda_conv_op_[1];
    block.conv2.Forward(batch_size,
                        cuda_conv_op_[3],
                        cuda_conv_op_[2],
                        block1_skip,
                        mask,
                        cuda_scratch_op_[0],
                        cuda_scratch_op_[1],
                        scratch_size_);

    std::swap(cuda_conv_op_[3], cuda_conv_op_[1]);

    // 3rd conv layer (2nd block)
    block.conv3.Forward(batch_size,
                        cuda_conv_op_[2],
                        cuda_conv_op_[1],
                        nullptr,
                        mask,
                        cuda_scratch_op_[0],
                        cuda_scratch_op_[1],
                        scratch_size_);

    // 4th conv layer (2nd block)
    void* block2_skip = cuda_conv_op_[1];
    block.conv4.Forward(batch_size,
                        cuda_conv_op_[3],
                        cuda_conv_op_[2],
                        block2_skip,
                        mask,
                        cuda_scratch_op_[0],
                        cuda_scratch_op_[1],
                        scratch_size_);

    std::swap(cuda_conv_op_[3], cuda_conv_op_[1]);

    // post-bottleneck
    void* btl_skip = tower_ptr->apply_se ? nullptr : cuda_conv_op_[0];
    block.post_btl_conv.Forward(batch_size,
                                cuda_conv_op_[3],
                                cuda_conv_op_[1],
                                btl_skip,
                                mask,
                                cuda_scratch_op_[0],
                                cuda_scratch_op_[1],
                                scratch_size_);
}

void CudaForwardPipe::NNGraph::MixerBlockForward(const int batch_size,
                                                 Block& block,
                                                 BlockBasic* tower_ptr,
                                                 void* mask) {
    // dw conv layer
    block.dw_conv.Forward(batch_size, cuda_conv_op_[3], cuda_conv_op_[0], cuda_conv_op_[0], mask);

    std::swap(cuda_conv_op_[3], cuda_conv_op_[0]);

    // 1st ffn conv layer
    block.conv1.Forward(batch_size,
                        cuda_conv_op_[1],
                        cuda_conv_op_[0],
                        nullptr,
                        mask,
                        cuda_scratch_op_[0],
                        cuda_scratch_op_[1],
                        scratch_size_);

    // 2nd ffn conv layer
    void* ffn_skip = tower_ptr->apply_se ? nullptr : cuda_conv_op_[0];
    block.conv2.Forward(batch_size,
                        cuda_conv_op_[3],
                        cuda_conv_op_[1],
                        ffn_skip,
                        mask,
                        cuda_scratch_op_[0],
                        cuda_scratch_op_[1],
                        scratch_size_);
}

void CudaForwardPipe::NNGraph::TransformerBlockForward(const int batch_size,
                                                       Block& block,
                                                       BlockBasic* tower_ptr,
                                                       void* mask) {
    const auto num_intersections = board_size_ * board_size_;

    // self-attention sublayer
    block.attn_norm.Forward(batch_size, cuda_conv_op_[1], cuda_conv_op_[0]);

    block.q_proj.Forward(batch_size,
                         cuda_conv_op_[2],
                         cuda_conv_op_[1],
                         nullptr,
                         mask,
                         cuda_scratch_op_[0],
                         cuda_scratch_op_[1],
                         scratch_size_);
    block.k_proj.Forward(batch_size,
                         cuda_conv_op_[3],
                         cuda_conv_op_[1],
                         nullptr,
                         mask,
                         cuda_scratch_op_[0],
                         cuda_scratch_op_[1],
                         scratch_size_);
    block.v_proj.Forward(batch_size,
                         cuda_attn_op_[0],
                         cuda_conv_op_[1],
                         nullptr,
                         mask,
                         cuda_scratch_op_[0],
                         cuda_scratch_op_[1],
                         scratch_size_);

    block.attention.Forward(
        batch_size, cuda_conv_op_[2], cuda_conv_op_[3], cuda_attn_op_[0], cuda_attn_op_[1], mask);

    block.out_proj.Forward(batch_size,
                           cuda_conv_op_[3],
                           cuda_conv_op_[2],
                           cuda_conv_op_[0],
                           mask,
                           cuda_scratch_op_[0],
                           cuda_scratch_op_[1],
                           scratch_size_);

    std::swap(cuda_conv_op_[3], cuda_conv_op_[0]);

    // gated feedforward sublayer
    const auto feedforwards = tower_ptr->feedforward_channels;
    block.ffn_norm.Forward(batch_size, cuda_conv_op_[1], cuda_conv_op_[0]);

    block.ffn_fc1.Forward(batch_size,
                          cuda_conv_op_[2],
                          cuda_conv_op_[1],
                          nullptr,
                          mask,
                          cuda_scratch_op_[0],
                          cuda_scratch_op_[1],
                          scratch_size_);
    block.ffn_gate.Forward(batch_size,
                           cuda_conv_op_[3],
                           cuda_conv_op_[1],
                           nullptr,
                           mask,
                           cuda_scratch_op_[0],
                           cuda_scratch_op_[1],
                           scratch_size_);
    cuda::MulVectors(handles_.fp16,
                     cuda_conv_op_[2],
                     cuda_conv_op_[2],
                     cuda_conv_op_[3],
                     batch_size * feedforwards * num_intersections,
                     handles_.stream);

    block.ffn_fc2.Forward(batch_size,
                          cuda_conv_op_[3],
                          cuda_conv_op_[2],
                          cuda_conv_op_[0],
                          mask,
                          cuda_scratch_op_[0],
                          cuda_scratch_op_[1],
                          scratch_size_);
}

void CudaForwardPipe::NNGraph::FillOutputs(const std::vector<float>& batch_prob,
                                           const std::vector<float>& batch_prob_pass,
                                           const std::vector<float>& batch_value_misc,
                                           const std::vector<float>& batch_ownership,
                                           const std::vector<InputData>& batch_input,
                                           std::vector<OutputResult>& batch_output_result) {
    const int batch_size = batch_output_result.size();
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
                int pol_index =
                    b * pol_offset + (int)PolicyBufferOffset::kNormal * num_intersections + idx;
                int own_index = b * own_offset + 0 * num_intersections + idx;
                output_result.probabilities[idx] = batch_prob[pol_index];
                output_result.ownership[idx] = batch_ownership[own_index];
            }
            output_result.pass_probability = batch_prob_pass[b * pass_probability_outputs + 0];

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
                int pol_index = b * pol_offset + (int)input.offset * num_intersections + idx;
                int own_index = b * own_offset + 0 * num_intersections + idx;
                output_result.probabilities[idx] = batch_prob[pol_index];
                output_result.ownership[idx] = batch_ownership[own_index];
            }
            output_result.pass_probability = batch_prob_pass[b * pass_probability_outputs + 0];

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

void CudaForwardPipe::NNGraph::DestroyGraph() {
    if (graph_ == nullptr) {
        return;
    }
    cuda::SetDevice(handles_.gpu_id);

    cuda::ReportCUDAErrors(cudaFree(cuda_scratch_op_[0]));
    cuda::ReportCUDAErrors(cudaFree(cuda_scratch_op_[1]));

    cuda::ReportCUDAErrors(cudaFree(cuda_conv_op_[0]));
    cuda::ReportCUDAErrors(cudaFree(cuda_conv_op_[1]));
    cuda::ReportCUDAErrors(cudaFree(cuda_conv_op_[2]));
    cuda::ReportCUDAErrors(cudaFree(cuda_conv_op_[3]));

    cuda::ReportCUDAErrors(cudaFree(cuda_pol_op_[0]));
    cuda::ReportCUDAErrors(cudaFree(cuda_pol_op_[1]));
    cuda::ReportCUDAErrors(cudaFree(cuda_pol_op_[2]));

    cuda::ReportCUDAErrors(cudaFree(cuda_val_op_[0]));
    cuda::ReportCUDAErrors(cudaFree(cuda_val_op_[1]));
    cuda::ReportCUDAErrors(cudaFree(cuda_val_op_[2]));

    cuda::ReportCUDAErrors(cudaFree(cuda_mask_op_[0]));
    cuda::ReportCUDAErrors(cudaFree(cuda_mask_op_[1]));

    cuda::ReportCUDAErrors(cudaFree(cuda_attn_op_[0]));
    cuda::ReportCUDAErrors(cudaFree(cuda_attn_op_[1]));
    cuda_attn_op_ = {nullptr, nullptr};

    cuda::ReportCUDAErrors(cudaFree(cuda_input_planes_));
    cuda::ReportCUDAErrors(cudaFree(cuda_output_prob_));
    cuda::ReportCUDAErrors(cudaFree(cuda_output_prob_pass_));
    cuda::ReportCUDAErrors(cudaFree(cuda_output_val_));
    cuda::ReportCUDAErrors(cudaFree(cuda_output_ownership_));

    cuda::ReportCUDAErrors(cudaFreeHost(host_mask_op_[0]));
    cuda::ReportCUDAErrors(cudaFreeHost(host_mask_op_[1]));

    cuda::ReportCUDAErrors(cudaFreeHost(host_input_planes_));
    cuda::ReportCUDAErrors(cudaFreeHost(host_output_prob_));
    cuda::ReportCUDAErrors(cudaFreeHost(host_output_prob_pass_));
    cuda::ReportCUDAErrors(cudaFreeHost(host_output_val_));
    cuda::ReportCUDAErrors(cudaFreeHost(host_output_ownership_));

    handles_.Release();

    graph_.reset();
    graph_ = nullptr;
}

CudaForwardPipe::NNGraph::~NNGraph() {
    DestroyGraph();
}

#endif
