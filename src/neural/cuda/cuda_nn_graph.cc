#ifdef USE_CUDA

#include "neural/cuda/cuda_nn_graph.h"

#include <stdexcept>

#include "neural/cuda/cuda_kernels.h"
#include "neural/encoder.h"
#include "utils/log.h"
#include "utils/option.h"

namespace cuda {

void NNGraph::ConstructGraph(bool dump_gpu_info,
                             const int gpu,
                             const int max_batch_size,
                             const int board_size,
                             std::shared_ptr<DNNWeights> weights) {
    if (graph_ != nullptr) {
        return;
    }

    graph_ = std::make_unique<Graph>();
    weights_ = weights;

    SetDevice(gpu);
    handles_.ApplyOnCurrentDevice();

    SetComputationMode(&handles_);
    if (dump_gpu_info) {
        LOGGING << GetCurrentDeviceInfo(&handles_);
    }

    board_size_ = board_size;
    scratch_size_ = 0;
    max_batch_ = max_batch_size;

    // Build the graph first.
    const auto input_channels = weights_->input_channels;
    const auto output_channels = weights_->residual_channels;
    const auto default_act = weights_->default_act;

    // input layer
    graph_->input_conv = Convolution(&handles_,
                                     max_batch_,      // max batch size
                                     board_size_,     // board size
                                     3,               // kernel size
                                     input_channels,  // input channels
                                     output_channels, // output channels
                                     default_act      // activation
    );

    // block tower
    const auto blocks = weights_->residual_blocks;
    auto peaks = PeakSizes{};
    for (int i = 0; i < blocks; ++i) {
        const auto tower_ptr = weights_->tower[i].get();
        auto layers = std::unique_ptr<BlockLayers>{nullptr};
        if (auto* block = tower_ptr->Cast<ResidualBlock>()) {
            layers = ConstructResidualBlock(block, peaks);
        } else if (auto* block = tower_ptr->Cast<BottleneckBlock>()) {
            layers = ConstructBottleneckBlock(block, peaks);
        } else if (auto* block = tower_ptr->Cast<NestedBottleneckBlock>()) {
            layers = ConstructNestedBottleneckBlock(block, peaks);
        } else if (auto* block = tower_ptr->Cast<MixerBlock>()) {
            layers = ConstructMixerBlock(block, peaks);
        } else if (auto* block = tower_ptr->Cast<TransformerBlock>()) {
            layers = ConstructTransformerBlock(block, peaks);
        } else {
            throw std::runtime_error("Unknown block type for CUDA backend.");
        }
        if (tower_ptr->apply_se) {
            ConstructSEUnit(*layers, tower_ptr);
        }
        graph_->tower.emplace_back(std::move(layers));
    }
    // policy head
    const auto policy_head_channels = weights_->policy_head_channels;
    const auto probabilities_channels = weights_->probabilities_channels;
    const auto pass_probability_outputs = weights_->pass_probability_outputs;
    graph_->p_hd_conv = Convolution(&handles_,
                                    max_batch_,           // max batch size
                                    board_size_,          // board size
                                    1,                    // kernel size
                                    output_channels,      // input channels
                                    policy_head_channels, // output channels
                                    default_act           // activation
    );
    if (weights_->policy_head_type == PolicyHeadType::kRepLK) {
        const auto filters = weights_->p_dw_conv.GetFilter();
        graph_->p_dw_conv = DepthwiseConvolution(&handles_,
                                                 max_batch_,           // max batch size
                                                 board_size_,          // board size
                                                 filters,              // kernel size
                                                 policy_head_channels, // input channels
                                                 default_act           // activation
        );
        graph_->p_pt_conv = Convolution(&handles_,
                                        max_batch_,           // max batch size
                                        board_size_,          // board size
                                        1,                    // kernel size
                                        policy_head_channels, // input channels
                                        policy_head_channels, // output channels
                                        default_act           // activation
        );
    }
    graph_->p_pool = GlobalPooling(&handles_,
                                   false,
                                   max_batch_,          // max batch size
                                   board_size_,         // board size
                                   policy_head_channels // input channels
    );
    graph_->p_inter = FullyConnect(&handles_,
                                   max_batch_,               // max batch size
                                   3 * policy_head_channels, // input sizes
                                   policy_head_channels,     // outpur size
                                   default_act               // activation
    );
    graph_->p_prob = Convolution(&handles_,
                                 max_batch_,             // max batch size
                                 board_size_,            // board size
                                 1,                      // kernel size
                                 policy_head_channels,   // input channels
                                 probabilities_channels, // output channels
                                 Activation::kIdentity   // activation
    );
    graph_->p_prob_pass = FullyConnect(&handles_,
                                       max_batch_,               // max batch size
                                       policy_head_channels,     // input sizes
                                       pass_probability_outputs, // outpur size
                                       Activation::kIdentity     // activation
    );

    // value head
    const auto value_head_channels = weights_->value_head_channels;
    const auto ownership_channels = weights_->ownership_channels;
    const auto value_misc_outputs = weights_->value_misc_outputs;
    graph_->v_hd_conv = Convolution(&handles_,
                                    max_batch_,          // max batch size
                                    board_size_,         // board size
                                    1,                   // kernel size
                                    output_channels,     // input channels
                                    value_head_channels, // output channels
                                    default_act          // activation
    );
    graph_->v_pool = GlobalPooling(&handles_,
                                   true,
                                   max_batch_,         // max batch size
                                   board_size_,        // board size
                                   value_head_channels // input channels
    );
    graph_->v_inter = FullyConnect(&handles_,
                                   max_batch_,              // max batch size
                                   3 * value_head_channels, // input sizes
                                   3 * value_head_channels, // outpur size
                                   default_act              // activation
    );
    graph_->v_ownership = Convolution(&handles_,
                                      max_batch_,           // max batch size
                                      board_size_,          // board size
                                      1,                    // kernel size
                                      value_head_channels,  // input channels
                                      ownership_channels,   // output channels
                                      Activation::kIdentity // activation
    );
    graph_->v_misc = FullyConnect(&handles_,
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
    const size_t factor = max_batch_ * GetCudaTypeSize(handles_.fp16);
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

    ReportCUDAErrors(cudaMalloc(&cuda_scratch_op_[0], scratch_size_));
    ReportCUDAErrors(cudaMalloc(&cuda_scratch_op_[1], scratch_size_));

    ReportCUDAErrors(cudaMalloc(&cuda_conv_op_[0], conv_op_size));
    ReportCUDAErrors(cudaMalloc(&cuda_conv_op_[1], conv_op_size));
    ReportCUDAErrors(cudaMalloc(&cuda_conv_op_[2], conv_op_size));
    ReportCUDAErrors(cudaMalloc(&cuda_conv_op_[3], conv_op_size));

    ReportCUDAErrors(cudaMalloc(&cuda_pol_op_[0], pol_op1_size));
    ReportCUDAErrors(cudaMalloc(&cuda_pol_op_[1], pol_op2_size));
    ReportCUDAErrors(cudaMalloc(&cuda_pol_op_[2], pol_op3_size));

    ReportCUDAErrors(cudaMalloc(&cuda_val_op_[0], val_op1_size));
    ReportCUDAErrors(cudaMalloc(&cuda_val_op_[1], val_op2_size));
    ReportCUDAErrors(cudaMalloc(&cuda_val_op_[2], val_op3_size));

    ReportCUDAErrors(cudaMalloc(&cuda_mask_op_[0], mask_op1_size));
    ReportCUDAErrors(cudaMalloc(&cuda_mask_op_[1], mask_op2_size));

    cuda_attn_op_ = {nullptr, nullptr};
    if (peaks.attn_channels > 0) {
        const size_t attn_values_size = factor * peaks.attn_channels * num_intersections;
        const size_t attn_scores_size =
            factor * peaks.attn_heads * num_intersections * num_intersections;

        ReportCUDAErrors(cudaMalloc(&cuda_attn_op_[0], attn_values_size));
        ReportCUDAErrors(cudaMalloc(&cuda_attn_op_[1], attn_scores_size));
    }

    ReportCUDAErrors(cudaMalloc(&cuda_input_planes_, planes_size));
    ReportCUDAErrors(cudaMalloc(&cuda_output_prob_pass_, pass_size));
    ReportCUDAErrors(cudaMalloc(&cuda_output_prob_, pol_size));
    ReportCUDAErrors(cudaMalloc(&cuda_output_ownership_, ownership_size));
    ReportCUDAErrors(cudaMalloc(&cuda_output_val_, val_size));

    // Locked-page memory.
    ReportCUDAErrors(cudaMallocHost(&host_mask_op_[0], mask_op1_size));
    ReportCUDAErrors(cudaMallocHost(&host_mask_op_[1], mask_op2_size));

    ReportCUDAErrors(cudaMallocHost(&host_input_planes_, planes_size));
    ReportCUDAErrors(cudaMallocHost(&host_output_prob_pass_, pass_size));
    ReportCUDAErrors(cudaMallocHost(&host_output_prob_, pol_size));
    ReportCUDAErrors(cudaMallocHost(&host_output_ownership_, ownership_size));
    ReportCUDAErrors(cudaMallocHost(&host_output_val_, val_size));
}

std::unique_ptr<NNGraph::ResidualLayers> NNGraph::ConstructResidualBlock(ResidualBlock* block,
                                                                         PeakSizes& peaks) {
    const auto default_act = weights_->default_act;
    const bool winograd = weights_->winograd;

    const auto channels = weights_->residual_channels;
    const auto last_act = block->apply_se ? Activation::kIdentity : default_act;

    auto layers = std::make_unique<ResidualLayers>();

    layers->conv1 = Convolution(&handles_,
                                max_batch_,  // max batch size
                                board_size_, // board size
                                3,           // kernel size
                                channels,    // input channels
                                channels,    // output channels
                                default_act  // activation
    );
    layers->conv2 = Convolution(&handles_,
                                max_batch_,  // max batch size
                                board_size_, // board size
                                3,           // kernel size
                                channels,    // input channels
                                channels,    // output channels
                                last_act     // activation
    );
    peaks.channels = std::max(peaks.channels, channels);

    layers->conv1.LoadWeights(
        block->conv1.GetWeights(), block->conv1.GetBiases(), scratch_size_, winograd);
    layers->conv2.LoadWeights(
        block->conv2.GetWeights(), block->conv2.GetBiases(), scratch_size_, winograd);

    return layers;
}

std::unique_ptr<NNGraph::BottleneckLayers> NNGraph::ConstructBottleneckBlock(BottleneckBlock* block,
                                                                             PeakSizes& peaks) {
    const auto default_act = weights_->default_act;
    const bool winograd = weights_->winograd;

    const auto outer_channels = weights_->residual_channels;
    const auto inner_channels = block->bottleneck_channels;
    const auto last_act = block->apply_se ? Activation::kIdentity : default_act;

    auto layers = std::make_unique<BottleneckLayers>();

    layers->pre_btl_conv = Convolution(&handles_,
                                       max_batch_,     // max batch size
                                       board_size_,    // board size
                                       1,              // kernel size
                                       outer_channels, // input channels
                                       inner_channels, // output channels
                                       default_act     // activation
    );
    layers->conv1 = Convolution(&handles_,
                                max_batch_,     // max batch size
                                board_size_,    // board size
                                3,              // kernel size
                                inner_channels, // input channels
                                inner_channels, // output channels
                                default_act     // activation
    );
    layers->conv2 = Convolution(&handles_,
                                max_batch_,     // max batch size
                                board_size_,    // board size
                                3,              // kernel size
                                inner_channels, // input channels
                                inner_channels, // output channels
                                default_act     // activation
    );
    layers->post_btl_conv = Convolution(&handles_,
                                        max_batch_,     // max batch size
                                        board_size_,    // board size
                                        1,              // kernel size
                                        inner_channels, // input channels
                                        outer_channels, // output channels
                                        last_act        // activation
    );
    peaks.channels = std::max({peaks.channels, inner_channels, outer_channels});

    layers->pre_btl_conv.LoadWeights(
        block->pre_btl_conv.GetWeights(), block->pre_btl_conv.GetBiases(), scratch_size_, winograd);
    layers->conv1.LoadWeights(
        block->conv1.GetWeights(), block->conv1.GetBiases(), scratch_size_, winograd);
    layers->conv2.LoadWeights(
        block->conv2.GetWeights(), block->conv2.GetBiases(), scratch_size_, winograd);
    layers->post_btl_conv.LoadWeights(block->post_btl_conv.GetWeights(),
                                      block->post_btl_conv.GetBiases(),
                                      scratch_size_,
                                      winograd);

    return layers;
}

std::unique_ptr<NNGraph::NestedBottleneckLayers>
NNGraph::ConstructNestedBottleneckBlock(NestedBottleneckBlock* block, PeakSizes& peaks) {
    const auto default_act = weights_->default_act;
    const bool winograd = weights_->winograd;

    const auto outer_channels = weights_->residual_channels;
    const auto inner_channels = block->bottleneck_channels;
    const auto last_act = block->apply_se ? Activation::kIdentity : default_act;

    auto layers = std::make_unique<NestedBottleneckLayers>();

    layers->pre_btl_conv = Convolution(&handles_,
                                       max_batch_,     // max batch size
                                       board_size_,    // board size
                                       1,              // kernel size
                                       outer_channels, // input channels
                                       inner_channels, // output channels
                                       default_act     // activation
    );
    layers->conv1 = Convolution(&handles_,
                                max_batch_,     // max batch size
                                board_size_,    // board size
                                3,              // kernel size
                                inner_channels, // input channels
                                inner_channels, // output channels
                                default_act     // activation
    );
    layers->conv2 = Convolution(&handles_,
                                max_batch_,     // max batch size
                                board_size_,    // board size
                                3,              // kernel size
                                inner_channels, // input channels
                                inner_channels, // output channels
                                default_act     // activation
    );
    layers->conv3 = Convolution(&handles_,
                                max_batch_,     // max batch size
                                board_size_,    // board size
                                3,              // kernel size
                                inner_channels, // input channels
                                inner_channels, // output channels
                                default_act     // activation
    );
    layers->conv4 = Convolution(&handles_,
                                max_batch_,     // max batch size
                                board_size_,    // board size
                                3,              // kernel size
                                inner_channels, // input channels
                                inner_channels, // output channels
                                default_act     // activation
    );
    layers->post_btl_conv = Convolution(&handles_,
                                        max_batch_,     // max batch size
                                        board_size_,    // board size
                                        1,              // kernel size
                                        inner_channels, // input channels
                                        outer_channels, // output channels
                                        last_act        // activation
    );
    peaks.channels = std::max({peaks.channels, inner_channels, outer_channels});

    layers->pre_btl_conv.LoadWeights(
        block->pre_btl_conv.GetWeights(), block->pre_btl_conv.GetBiases(), scratch_size_, winograd);
    layers->conv1.LoadWeights(
        block->conv1.GetWeights(), block->conv1.GetBiases(), scratch_size_, winograd);
    layers->conv2.LoadWeights(
        block->conv2.GetWeights(), block->conv2.GetBiases(), scratch_size_, winograd);
    layers->conv3.LoadWeights(
        block->conv3.GetWeights(), block->conv3.GetBiases(), scratch_size_, winograd);
    layers->conv4.LoadWeights(
        block->conv4.GetWeights(), block->conv4.GetBiases(), scratch_size_, winograd);
    layers->post_btl_conv.LoadWeights(block->post_btl_conv.GetWeights(),
                                      block->post_btl_conv.GetBiases(),
                                      scratch_size_,
                                      winograd);

    return layers;
}

std::unique_ptr<NNGraph::MixerLayers> NNGraph::ConstructMixerBlock(MixerBlock* block,
                                                                   PeakSizes& peaks) {
    const auto default_act = weights_->default_act;
    const bool winograd = weights_->winograd;

    const auto channels = weights_->residual_channels;
    const auto feedforwards = block->feedforward_channels;
    const auto filters = block->dw_conv.GetFilter();
    const auto last_act = block->apply_se ? Activation::kIdentity : default_act;

    auto layers = std::make_unique<MixerLayers>();

    layers->dw_conv = DepthwiseConvolution(&handles_,
                                           max_batch_,  // max batch size
                                           board_size_, // board size
                                           filters,     // kernel size
                                           channels,    // input channels
                                           default_act  // activation
    );
    layers->conv1 = Convolution(&handles_,
                                max_batch_,   // max batch size
                                board_size_,  // board size
                                1,            // kernel size
                                channels,     // input channels
                                feedforwards, // output channels
                                default_act   // activation
    );
    layers->conv2 = Convolution(&handles_,
                                max_batch_,   // max batch size
                                board_size_,  // board size
                                1,            // kernel size
                                feedforwards, // input channels
                                channels,     // output channels
                                last_act      // activation
    );
    peaks.channels = std::max({peaks.channels, channels, feedforwards});

    layers->dw_conv.LoadWeights(block->dw_conv.GetWeights(), block->dw_conv.GetBiases());
    layers->conv1.LoadWeights(
        block->conv1.GetWeights(), block->conv1.GetBiases(), scratch_size_, winograd);
    layers->conv2.LoadWeights(
        block->conv2.GetWeights(), block->conv2.GetBiases(), scratch_size_, winograd);

    return layers;
}

std::unique_ptr<NNGraph::TransformerLayers>
NNGraph::ConstructTransformerBlock(TransformerBlock* block, PeakSizes& peaks) {
    const auto default_act = weights_->default_act;
    const bool winograd = weights_->winograd;

    const auto channels = weights_->residual_channels;
    const auto num_heads = block->num_heads;
    const auto head_dim = block->head_dim;
    const auto attn_channels = num_heads * head_dim;
    const auto feedforwards = block->feedforward_channels;

    auto layers = std::make_unique<TransformerLayers>();

    layers->attn_norm = RMSNorm(&handles_,
                                max_batch_,  // max batch size
                                board_size_, // board size
                                channels     // channels
    );
    layers->q_proj = Convolution(&handles_,
                                 max_batch_,           // max batch size
                                 board_size_,          // board size
                                 1,                    // kernel size
                                 channels,             // input channels
                                 attn_channels,        // output channels
                                 Activation::kIdentity // activation
    );
    layers->k_proj = Convolution(&handles_,
                                 max_batch_,           // max batch size
                                 board_size_,          // board size
                                 1,                    // kernel size
                                 channels,             // input channels
                                 attn_channels,        // output channels
                                 Activation::kIdentity // activation
    );
    layers->v_proj = Convolution(&handles_,
                                 max_batch_,           // max batch size
                                 board_size_,          // board size
                                 1,                    // kernel size
                                 channels,             // input channels
                                 attn_channels,        // output channels
                                 Activation::kIdentity // activation
    );
    layers->attention = MultiHeadAttention(&handles_,
                                           max_batch_,  // max batch size
                                           board_size_, // board size
                                           num_heads,   // number of heads
                                           head_dim     // head size
    );
    layers->out_proj = Convolution(&handles_,
                                   max_batch_,           // max batch size
                                   board_size_,          // board size
                                   1,                    // kernel size
                                   attn_channels,        // input channels
                                   channels,             // output channels
                                   Activation::kIdentity // activation
    );
    layers->ffn_norm = RMSNorm(&handles_,
                               max_batch_,  // max batch size
                               board_size_, // board size
                               channels     // channels
    );
    layers->ffn_fc1 = Convolution(&handles_,
                                  max_batch_,   // max batch size
                                  board_size_,  // board size
                                  1,            // kernel size
                                  channels,     // input channels
                                  feedforwards, // output channels
                                  default_act   // activation
    );
    layers->ffn_gate = Convolution(&handles_,
                                   max_batch_,           // max batch size
                                   board_size_,          // board size
                                   1,                    // kernel size
                                   channels,             // input channels
                                   feedforwards,         // output channels
                                   Activation::kIdentity // activation
    );
    layers->ffn_fc2 = Convolution(&handles_,
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

    layers->attn_norm.LoadWeights(block->attn_norm.GetGammas());
    layers->q_proj.LoadWeights(
        block->q_proj.GetWeights(), block->q_proj.GetBiases(), scratch_size_, winograd);
    layers->k_proj.LoadWeights(
        block->k_proj.GetWeights(), block->k_proj.GetBiases(), scratch_size_, winograd);
    layers->v_proj.LoadWeights(
        block->v_proj.GetWeights(), block->v_proj.GetBiases(), scratch_size_, winograd);
    layers->attention.LoadWeights(block->rope.GetFrequencies());
    layers->out_proj.LoadWeights(
        block->out_proj.GetWeights(), block->out_proj.GetBiases(), scratch_size_, winograd);
    layers->ffn_norm.LoadWeights(block->ffn_norm.GetGammas());
    layers->ffn_fc1.LoadWeights(
        block->ffn_fc1.GetWeights(), block->ffn_fc1.GetBiases(), scratch_size_, winograd);
    layers->ffn_gate.LoadWeights(
        block->ffn_gate.GetWeights(), block->ffn_gate.GetBiases(), scratch_size_, winograd);
    layers->ffn_fc2.LoadWeights(
        block->ffn_fc2.GetWeights(), block->ffn_fc2.GetBiases(), scratch_size_, winograd);

    return layers;
}

void NNGraph::ConstructSEUnit(BlockLayers& layers, BlockBasic* tower_ptr) {
    const auto default_act = weights_->default_act;

    const auto channels = weights_->residual_channels;
    const size_t se_size = tower_ptr->se_size;

    layers.se_module = SEUnit(&handles_,
                              max_batch_,  // max batch size
                              board_size_, // board size
                              channels,    // channels
                              se_size,     // SE size
                              default_act  // activation
    );

    layers.se_module.LoadWeights(tower_ptr->squeeze.GetWeights(),
                                 tower_ptr->squeeze.GetBiases(),
                                 tower_ptr->excite.GetWeights(),
                                 tower_ptr->excite.GetBiases());
}

void NNGraph::SetComputationMode(CudaHandles* handles) {
    cudaDeviceProp dev_prop = GetDeviceProp();

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
        ReportCUBLASErrors(cublasSetMathMode(handles->cublas_handle, CUBLAS_TENSOR_OP_MATH));
    }
}

bool NNGraph::ApplyMask(const std::vector<InputData>& batch_input) {
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
        CopyToCudaOpAsync(
            handles_.fp16, &(cuda_mask_op_[0]), spat_mask, handles_.stream, &(host_mask_op_[0]));
        CopyToCudaOpAsync(
            handles_.fp16, &(cuda_mask_op_[1]), sqrt_mask, handles_.stream, &(host_mask_op_[1]));
    }

    return should_apply_mask;
}

std::vector<OutputResult> NNGraph::BatchForward(const std::vector<InputData>& batch_input) {
    const auto batch_size = (int)batch_input.size();

    assert(max_batch_ >= batch_size);
    SetDevice(handles_.gpu_id);

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
    CopyToCudaOpAsync(
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

        // The layers are built by the same dispatch in ConstructGraph().
        auto& layers = *graph_->tower[i];
        const auto tower_ptr = weights_->tower[i].get();
        if (auto* block = tower_ptr->Cast<ResidualBlock>()) {
            ResidualBlockForward(
                batch_size, static_cast<ResidualLayers&>(layers), block, mask_buf[0]);
        } else if (auto* block = tower_ptr->Cast<BottleneckBlock>()) {
            BottleneckBlockForward(
                batch_size, static_cast<BottleneckLayers&>(layers), block, mask_buf[0]);
        } else if (auto* block = tower_ptr->Cast<NestedBottleneckBlock>()) {
            NestedBottleneckBlockForward(
                batch_size, static_cast<NestedBottleneckLayers&>(layers), block, mask_buf[0]);
        } else if (auto* block = tower_ptr->Cast<MixerBlock>()) {
            MixerBlockForward(batch_size, static_cast<MixerLayers&>(layers), block, mask_buf[0]);
        } else if (auto* block = tower_ptr->Cast<TransformerBlock>()) {
            TransformerBlockForward(
                batch_size, static_cast<TransformerLayers&>(layers), block, mask_buf[0]);
        }

        bool module_skip = false;
        if (tower_ptr->apply_se) {
            // squeeze-and-excitation module
            void* se_skip = cuda_conv_op_[0];
            void* se_outs = cuda_conv_op_[0];

            layers.se_module.Forward(
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
    AddSpatial(handles_.fp16,
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
    CopyToHostOpAsync(
        handles_.fp16, batch_prob, &cuda_output_prob_, handles_.stream, &host_output_prob_);
    CopyToHostOpAsync(handles_.fp16,
                      batch_prob_pass,
                      &cuda_output_prob_pass_,
                      handles_.stream,
                      &host_output_prob_pass_);
    CopyToHostOpAsync(
        handles_.fp16, batch_value_misc, &cuda_output_val_, handles_.stream, &host_output_val_);
    CopyToHostOpAsync(handles_.fp16,
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

void NNGraph::ResidualBlockForward(const int batch_size,
                                   ResidualLayers& layers,
                                   ResidualBlock* block,
                                   void* mask) {
    // 1st conv layer
    layers.conv1.Forward(batch_size,
                         cuda_conv_op_[1],
                         cuda_conv_op_[0],
                         nullptr,
                         mask,
                         cuda_scratch_op_[0],
                         cuda_scratch_op_[1],
                         scratch_size_);

    // 2nd conv layer
    void* second_skip = block->apply_se ? nullptr : cuda_conv_op_[0];
    layers.conv2.Forward(batch_size,
                         cuda_conv_op_[3],
                         cuda_conv_op_[1],
                         second_skip,
                         mask,
                         cuda_scratch_op_[0],
                         cuda_scratch_op_[1],
                         scratch_size_);
}

void NNGraph::BottleneckBlockForward(const int batch_size,
                                     BottleneckLayers& layers,
                                     BottleneckBlock* block,
                                     void* mask) {
    // pre-bottleneck
    layers.pre_btl_conv.Forward(batch_size,
                                cuda_conv_op_[1],
                                cuda_conv_op_[0],
                                nullptr,
                                mask,
                                cuda_scratch_op_[0],
                                cuda_scratch_op_[1],
                                scratch_size_);

    // 1st conv layer
    layers.conv1.Forward(batch_size,
                         cuda_conv_op_[2],
                         cuda_conv_op_[1],
                         nullptr,
                         mask,
                         cuda_scratch_op_[0],
                         cuda_scratch_op_[1],
                         scratch_size_);

    // 2nd conv layer
    layers.conv2.Forward(batch_size,
                         cuda_conv_op_[1],
                         cuda_conv_op_[2],
                         nullptr,
                         mask,
                         cuda_scratch_op_[0],
                         cuda_scratch_op_[1],
                         scratch_size_);

    // post-bottleneck
    void* btl_skip = block->apply_se ? nullptr : cuda_conv_op_[0];
    layers.post_btl_conv.Forward(batch_size,
                                 cuda_conv_op_[3],
                                 cuda_conv_op_[1],
                                 btl_skip,
                                 mask,
                                 cuda_scratch_op_[0],
                                 cuda_scratch_op_[1],
                                 scratch_size_);
}

void NNGraph::NestedBottleneckBlockForward(const int batch_size,
                                           NestedBottleneckLayers& layers,
                                           NestedBottleneckBlock* block,
                                           void* mask) {
    // pre-bottleneck
    layers.pre_btl_conv.Forward(batch_size,
                                cuda_conv_op_[1],
                                cuda_conv_op_[0],
                                nullptr,
                                mask,
                                cuda_scratch_op_[0],
                                cuda_scratch_op_[1],
                                scratch_size_);

    // 1st conv layer (1st block)
    layers.conv1.Forward(batch_size,
                         cuda_conv_op_[2],
                         cuda_conv_op_[1],
                         nullptr,
                         mask,
                         cuda_scratch_op_[0],
                         cuda_scratch_op_[1],
                         scratch_size_);

    // 2nd conv layer (1st block)
    void* block1_skip = cuda_conv_op_[1];
    layers.conv2.Forward(batch_size,
                         cuda_conv_op_[3],
                         cuda_conv_op_[2],
                         block1_skip,
                         mask,
                         cuda_scratch_op_[0],
                         cuda_scratch_op_[1],
                         scratch_size_);

    std::swap(cuda_conv_op_[3], cuda_conv_op_[1]);

    // 3rd conv layer (2nd block)
    layers.conv3.Forward(batch_size,
                         cuda_conv_op_[2],
                         cuda_conv_op_[1],
                         nullptr,
                         mask,
                         cuda_scratch_op_[0],
                         cuda_scratch_op_[1],
                         scratch_size_);

    // 4th conv layer (2nd block)
    void* block2_skip = cuda_conv_op_[1];
    layers.conv4.Forward(batch_size,
                         cuda_conv_op_[3],
                         cuda_conv_op_[2],
                         block2_skip,
                         mask,
                         cuda_scratch_op_[0],
                         cuda_scratch_op_[1],
                         scratch_size_);

    std::swap(cuda_conv_op_[3], cuda_conv_op_[1]);

    // post-bottleneck
    void* btl_skip = block->apply_se ? nullptr : cuda_conv_op_[0];
    layers.post_btl_conv.Forward(batch_size,
                                 cuda_conv_op_[3],
                                 cuda_conv_op_[1],
                                 btl_skip,
                                 mask,
                                 cuda_scratch_op_[0],
                                 cuda_scratch_op_[1],
                                 scratch_size_);
}

void NNGraph::MixerBlockForward(const int batch_size,
                                MixerLayers& layers,
                                MixerBlock* block,
                                void* mask) {
    // dw conv layer
    layers.dw_conv.Forward(batch_size, cuda_conv_op_[3], cuda_conv_op_[0], cuda_conv_op_[0], mask);

    std::swap(cuda_conv_op_[3], cuda_conv_op_[0]);

    // 1st ffn conv layer
    layers.conv1.Forward(batch_size,
                         cuda_conv_op_[1],
                         cuda_conv_op_[0],
                         nullptr,
                         mask,
                         cuda_scratch_op_[0],
                         cuda_scratch_op_[1],
                         scratch_size_);

    // 2nd ffn conv layer
    void* ffn_skip = block->apply_se ? nullptr : cuda_conv_op_[0];
    layers.conv2.Forward(batch_size,
                         cuda_conv_op_[3],
                         cuda_conv_op_[1],
                         ffn_skip,
                         mask,
                         cuda_scratch_op_[0],
                         cuda_scratch_op_[1],
                         scratch_size_);
}

void NNGraph::TransformerBlockForward(const int batch_size,
                                      TransformerLayers& layers,
                                      TransformerBlock* block,
                                      void* mask) {
    const auto num_intersections = board_size_ * board_size_;

    // self-attention sublayer
    layers.attn_norm.Forward(batch_size, cuda_conv_op_[1], cuda_conv_op_[0]);

    layers.q_proj.Forward(batch_size,
                          cuda_conv_op_[2],
                          cuda_conv_op_[1],
                          nullptr,
                          mask,
                          cuda_scratch_op_[0],
                          cuda_scratch_op_[1],
                          scratch_size_);
    layers.k_proj.Forward(batch_size,
                          cuda_conv_op_[3],
                          cuda_conv_op_[1],
                          nullptr,
                          mask,
                          cuda_scratch_op_[0],
                          cuda_scratch_op_[1],
                          scratch_size_);
    layers.v_proj.Forward(batch_size,
                          cuda_attn_op_[0],
                          cuda_conv_op_[1],
                          nullptr,
                          mask,
                          cuda_scratch_op_[0],
                          cuda_scratch_op_[1],
                          scratch_size_);

    layers.attention.Forward(
        batch_size, cuda_conv_op_[2], cuda_conv_op_[3], cuda_attn_op_[0], cuda_attn_op_[1], mask);

    layers.out_proj.Forward(batch_size,
                            cuda_conv_op_[3],
                            cuda_conv_op_[2],
                            cuda_conv_op_[0],
                            mask,
                            cuda_scratch_op_[0],
                            cuda_scratch_op_[1],
                            scratch_size_);

    std::swap(cuda_conv_op_[3], cuda_conv_op_[0]);

    // gated feedforward sublayer
    const auto feedforwards = block->feedforward_channels;
    layers.ffn_norm.Forward(batch_size, cuda_conv_op_[1], cuda_conv_op_[0]);

    layers.ffn_fc1.Forward(batch_size,
                           cuda_conv_op_[2],
                           cuda_conv_op_[1],
                           nullptr,
                           mask,
                           cuda_scratch_op_[0],
                           cuda_scratch_op_[1],
                           scratch_size_);
    layers.ffn_gate.Forward(batch_size,
                            cuda_conv_op_[3],
                            cuda_conv_op_[1],
                            nullptr,
                            mask,
                            cuda_scratch_op_[0],
                            cuda_scratch_op_[1],
                            scratch_size_);
    MulVectors(handles_.fp16,
               cuda_conv_op_[2],
               cuda_conv_op_[2],
               cuda_conv_op_[3],
               batch_size * feedforwards * num_intersections,
               handles_.stream);

    layers.ffn_fc2.Forward(batch_size,
                           cuda_conv_op_[3],
                           cuda_conv_op_[2],
                           cuda_conv_op_[0],
                           mask,
                           cuda_scratch_op_[0],
                           cuda_scratch_op_[1],
                           scratch_size_);
}

void NNGraph::FillOutputs(const std::vector<float>& batch_prob,
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

void NNGraph::DestroyGraph() {
    if (graph_ == nullptr) {
        return;
    }
    SetDevice(handles_.gpu_id);

    ReportCUDAErrors(cudaFree(cuda_scratch_op_[0]));
    ReportCUDAErrors(cudaFree(cuda_scratch_op_[1]));

    ReportCUDAErrors(cudaFree(cuda_conv_op_[0]));
    ReportCUDAErrors(cudaFree(cuda_conv_op_[1]));
    ReportCUDAErrors(cudaFree(cuda_conv_op_[2]));
    ReportCUDAErrors(cudaFree(cuda_conv_op_[3]));

    ReportCUDAErrors(cudaFree(cuda_pol_op_[0]));
    ReportCUDAErrors(cudaFree(cuda_pol_op_[1]));
    ReportCUDAErrors(cudaFree(cuda_pol_op_[2]));

    ReportCUDAErrors(cudaFree(cuda_val_op_[0]));
    ReportCUDAErrors(cudaFree(cuda_val_op_[1]));
    ReportCUDAErrors(cudaFree(cuda_val_op_[2]));

    ReportCUDAErrors(cudaFree(cuda_mask_op_[0]));
    ReportCUDAErrors(cudaFree(cuda_mask_op_[1]));

    ReportCUDAErrors(cudaFree(cuda_attn_op_[0]));
    ReportCUDAErrors(cudaFree(cuda_attn_op_[1]));

    ReportCUDAErrors(cudaFree(cuda_input_planes_));
    ReportCUDAErrors(cudaFree(cuda_output_prob_));
    ReportCUDAErrors(cudaFree(cuda_output_prob_pass_));
    ReportCUDAErrors(cudaFree(cuda_output_val_));
    ReportCUDAErrors(cudaFree(cuda_output_ownership_));

    ReportCUDAErrors(cudaFreeHost(host_mask_op_[0]));
    ReportCUDAErrors(cudaFreeHost(host_mask_op_[1]));

    ReportCUDAErrors(cudaFreeHost(host_input_planes_));
    ReportCUDAErrors(cudaFreeHost(host_output_prob_));
    ReportCUDAErrors(cudaFreeHost(host_output_prob_pass_));
    ReportCUDAErrors(cudaFreeHost(host_output_val_));
    ReportCUDAErrors(cudaFreeHost(host_output_ownership_));

    handles_.Release();

    graph_.reset();
    graph_ = nullptr;
}

NNGraph::~NNGraph() {
    DestroyGraph();
}

} // namespace cuda

#endif
