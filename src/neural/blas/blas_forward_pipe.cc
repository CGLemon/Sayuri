#include "neural/blas/blas_forward_pipe.h"
#include "neural/blas/convolution.h"
#include "neural/blas/batchnorm.h" // not used
#include "neural/blas/se_unit.h"
#include "neural/blas/fullyconnect.h"
#include "neural/blas/biases.h"
#include "neural/blas/winograd_convolution3.h"
#include "neural/winograd_helper.h"
#include "utils/option.h"

#include <algorithm>

void BlasForwardPipe::Initialize(std::shared_ptr<DNNWeights> weights) {
    Load(weights);
    InitWinograd();
}

void BlasForwardPipe::InitWinograd() {
    if (weights_ == nullptr ||
            !weights_->winograd ||
            weights_->winograd_initialized) {
        return;
    }

    const auto residual_channels = weights_->residual_channels;

    // The input layer.
    weights_->input_conv.GetWeights() =
        WinogradTransformF(weights_->input_conv.GetWeights(),
                               residual_channels, kInputChannels);

    // The block tower.
    for (auto &block : weights_->tower) {
        if (block->IsResidualBlock()) {
            int channels = residual_channels;
            block->conv1.GetWeights() = WinogradTransformF(
                block->conv1.GetWeights(), channels, channels);
            block->conv2.GetWeights() = WinogradTransformF(
                block->conv2.GetWeights(), channels, channels);
        } else if (block->IsBottleneckBlock()) {
            int channels = block->bottleneck_channels;
            block->conv1.GetWeights() = WinogradTransformF(
                block->conv1.GetWeights(), channels, channels);
            block->conv2.GetWeights() = WinogradTransformF(
                block->conv2.GetWeights(), channels, channels);
        }
    }
    weights_->winograd_initialized = true;
}

void BlasForwardPipe::Load(std::shared_ptr<DNNWeights> weights) {
    weights_ = weights;
}

void BlasForwardPipe::ResidualBlockForward(int board_size,
                                           BlockBasic * tower_ptr,
                                           bool use_winograd,
                                           std::vector<float> &residual,
                                           std::vector<float> &conv_in,
                                           std::vector<float> &conv_out,
                                           std::vector<float> &workspace0,
                                           std::vector<float> &workspace1) {
    using Convolution3 = Convolution<3>;
    const auto zero_vec = std::vector<float>{};
    const auto channels = weights_->residual_channels;
    const auto default_act = weights_->default_act;

    // 1st conv3
    if (use_winograd) {
            WinogradConvolution3::Forward(
                board_size, channels, channels,
                conv_in,
                tower_ptr->conv1.GetWeights(),
                workspace0, workspace1, conv_out);
    } else {
        Convolution3::Forward(
            board_size, channels, channels,
            conv_in,
            tower_ptr->conv1.GetWeights(),
            workspace0, conv_out);
    }
    AddSpatialBiases::Forward(
        board_size, channels,
        conv_out,
        tower_ptr->conv1.GetBiases(), default_act);

    std::swap(conv_in, residual);
    std::swap(conv_out, conv_in);

    // 2nd conv3
    if (use_winograd) {
        WinogradConvolution3::Forward(
            board_size, channels, channels,
            conv_in,
            tower_ptr->conv2.GetWeights(),
            workspace0, workspace1, conv_out);
    } else {
        Convolution3::Forward(
            board_size, channels, channels,
            conv_in,
            tower_ptr->conv2.GetWeights(),
            workspace0, conv_out);
    }

    auto &last_skip = tower_ptr->apply_se ? zero_vec : residual;
    auto last_act = tower_ptr->apply_se ? Activation::kIdentity : default_act;

    AddSpatialBiases::Forward(
        board_size, channels,
        conv_out,
        tower_ptr->conv2.GetBiases(),
        last_skip, last_act);
}

void BlasForwardPipe::BottleneckBlockForward(int board_size,
                                             BlockBasic * tower_ptr,
                                             bool use_winograd,
                                             std::vector<float> &residual,
                                             std::vector<float> &conv_in,
                                             std::vector<float> &conv_out,
                                             std::vector<float> &workspace0,
                                             std::vector<float> &workspace1) {
    using Convolution3 = Convolution<3>;
    const auto zero_vec = std::vector<float>{};
    const auto outer_channels = weights_->residual_channels;
    const auto inner_channels = tower_ptr->bottleneck_channels;
    const auto default_act = weights_->default_act;

    // The pre-bottleneck conv1.
    Convolution1::Forward(
        board_size, outer_channels, inner_channels,
        conv_in,
        tower_ptr->pre_btl_conv.GetWeights(),
        workspace0, conv_out);
    AddSpatialBiases::Forward(
        board_size, inner_channels,
        conv_out,
        tower_ptr->pre_btl_conv.GetBiases(), default_act);

    std::swap(conv_in, residual);
    std::swap(conv_out, conv_in);

    // 1st conv3
    if (use_winograd) {
        WinogradConvolution3::Forward(
            board_size, inner_channels, inner_channels,
            conv_in,
            tower_ptr->conv1.GetWeights(),
            workspace0, workspace1, conv_out);
    } else {
        Convolution3::Forward(
            board_size, inner_channels, inner_channels,
            conv_in,
            tower_ptr->conv1.GetWeights(),
            workspace0, conv_out);
    }
    AddSpatialBiases::Forward(
        board_size, inner_channels,
        conv_out,
        tower_ptr->conv1.GetBiases(), default_act);

    std::swap(conv_out, conv_in);

    // 2nd conv3
    if (use_winograd) {
        WinogradConvolution3::Forward(
            board_size, inner_channels, inner_channels,
            conv_in,
            tower_ptr->conv2.GetWeights(),
            workspace0, workspace1, conv_out);
    } else {
        Convolution3::Forward(
            board_size, inner_channels, inner_channels,
            conv_in,
            tower_ptr->conv2.GetWeights(),
            workspace0, conv_out);
    }
    AddSpatialBiases::Forward(
        board_size, inner_channels,
        conv_out,
        tower_ptr->conv2.GetBiases(), default_act);

    std::swap(conv_out, conv_in);

    // The post-bottleneck conv1.
    Convolution1::Forward(
        board_size, inner_channels, outer_channels,
        conv_in,
        tower_ptr->post_btl_conv.GetWeights(),
        workspace0, conv_out);

    auto &last_skip = tower_ptr->apply_se ? zero_vec : residual;
    auto last_act = tower_ptr->apply_se ? Activation::kIdentity : default_act;

    AddSpatialBiases::Forward(
        board_size, outer_channels,
        conv_out,
        tower_ptr->post_btl_conv.GetBiases(),
        last_skip, last_act);
}

void BlasForwardPipe::MixerBlockForward(int board_size,
                                        BlockBasic * tower_ptr,
                                        std::vector<float> &residual,
                                        std::vector<float> &conv_in,
                                        std::vector<float> &conv_out) {
    auto zero_vec = std::vector<float>{};
    const auto channels = weights_->residual_channels;
    const auto ffn_channels = tower_ptr->feedforward_channels;
    const auto filter_size =  tower_ptr->dw_conv.GetFilter();
    const auto default_act = weights_->default_act;

    // dw conv layer
    DepthwiseConvolution::Forward(
        board_size, filter_size, channels,
        conv_in, tower_ptr->dw_conv.GetWeights(), conv_out);
    AddSpatialBiasesPost::Forward(
        board_size, channels,
        conv_out, tower_ptr->dw_conv.GetBiases(), default_act, conv_in);

    std::swap(conv_out, conv_in);

    // 1st ffn conv layer
    Convolution1::Forward(
        board_size, channels, ffn_channels,
        conv_in,
        tower_ptr->conv1.GetWeights(),
        zero_vec, conv_out);
    AddSpatialBiases::Forward(
        board_size, ffn_channels,
        conv_out,
        tower_ptr->conv1.GetBiases(), default_act);

    std::swap(conv_in, residual);
    std::swap(conv_out, conv_in);

    // 2nd ffn conv layer
    Convolution1::Forward(
        board_size, ffn_channels, channels,
        conv_in,
        tower_ptr->conv2.GetWeights(),
        zero_vec, conv_out);

    auto &last_skip = tower_ptr->apply_se ? zero_vec : residual;
    auto last_act = tower_ptr->apply_se ? Activation::kIdentity : default_act;

    AddSpatialBiases::Forward(
        board_size, channels,
        conv_out,
        tower_ptr->conv2.GetBiases(),
        last_skip, last_act);
}

OutputResult BlasForwardPipe::Forward(const InputData &inpnts) {
    using Convolution3 = Convolution<3>;

    // Some useful information for network.
    int tower_peak_channels = weights_->residual_channels;
    for (int i = 0; i < weights_->residual_blocks; ++i) {
        auto tower_ptr = weights_->tower[i].get();

        if (tower_ptr->IsResidualBlock()) {
            // The block peak channels is residual channels.
        } else if (tower_ptr->IsBottleneckBlock()) {
            tower_peak_channels = std::max(
                tower_peak_channels, tower_ptr->bottleneck_channels);
        } else if (tower_ptr->IsMixerBlock()) {
            tower_peak_channels = std::max(
                tower_peak_channels, tower_ptr->feedforward_channels);
        }
    }
    const auto board_size = inpnts.board_size;
    const auto num_intersections = board_size * board_size;
    const auto residual_channels = weights_->residual_channels;
    const auto max_channels = std::max({kInputChannels,
                                        tower_peak_channels,
                                        weights_->policy_extract_channels,
                                        weights_->value_extract_channels});
    const auto plane_size = kInputChannels * num_intersections;
    const auto max_intermediates = std::max(weights_->policy_extract_channels,
                                                weights_->value_extract_channels);
    const auto default_act = weights_->default_act;


    // Allocate the forward pipe buffers.
    bool use_winograd = weights_->winograd;
    int workspace0_size = 0;
    int workspace1_size = 0;

    if (use_winograd) {
        workspace0_size =
            workspace1_size =
            WinogradConvolution3::GetWorkspaceSize(board_size, max_channels);
    } else {
        workspace0_size =
            Convolution3::GetWorkspaceSize(board_size, max_channels);
        workspace1_size = 1; // not used.
    }

    auto workspace0 = std::vector<float>(workspace0_size);
    auto workspace1 = std::vector<float>(workspace1_size);

    auto conv_out = std::vector<float>(tower_peak_channels * num_intersections);
    auto conv_in = std::vector<float>(tower_peak_channels * num_intersections);
    auto residual = std::vector<float>(tower_peak_channels * num_intersections);
    auto intermediate = std::vector<float>(3 * max_intermediates);
    auto pooling = std::vector<float>(3 * max_intermediates);

    // Copy input plane to buffer.
    auto planes = std::vector<float>(plane_size);
    std::copy(std::begin(inpnts.planes),
                  std::begin(inpnts.planes) + plane_size,
                  std::begin(planes));

    // Allocate the output buffers.
    auto output_prob = std::vector<float>(kOuputProbabilitiesChannels * num_intersections);
    auto output_pass = std::vector<float>(kOuputPassProbability);
    auto output_ownership = std::vector<float>(kOuputOwnershipChannels * num_intersections);
    auto output_misc = std::vector<float>(kOuputValueMisc);

    // The input Layers.
    if (use_winograd) {
        WinogradConvolution3::Forward(
            board_size, kInputChannels, residual_channels,
            planes,
            weights_->input_conv.GetWeights(),
            workspace0, workspace1, conv_out);
    } else {
        Convolution3::Forward(
            board_size, kInputChannels, residual_channels,
            planes,
            weights_->input_conv.GetWeights(),
            workspace0, conv_out);
    }

    AddSpatialBiases::Forward(
        board_size, residual_channels,
        conv_out,
        weights_->input_conv.GetBiases(), default_act);

    // The block tower.
    for (int i = 0; i < weights_->residual_blocks; ++i) {
        auto tower_ptr = weights_->tower[i].get();
        std::swap(conv_out, conv_in);

        if (tower_ptr->IsResidualBlock()) {
            ResidualBlockForward(
                board_size, tower_ptr, use_winograd,
                residual, conv_in, conv_out, workspace0, workspace1);
        } else if (tower_ptr->IsBottleneckBlock()) {
            BottleneckBlockForward(
                board_size, tower_ptr, use_winograd,
                residual, conv_in, conv_out, workspace0, workspace1);
        } else if (tower_ptr->IsMixerBlock()) {
            MixerBlockForward(
                board_size, tower_ptr,
                residual, conv_in, conv_out);
        }

        // The SE process.
        if (tower_ptr->apply_se) {
            auto &se_skip = residual;
            const auto se_size = tower_ptr->se_size;
            SEUnit::Forward(
                board_size, residual_channels, se_size,
                conv_out, se_skip,
                tower_ptr->squeeze.GetWeights(),
                tower_ptr->squeeze.GetBiases(),
                tower_ptr->excite.GetWeights(),
                tower_ptr->excite.GetBiases(),
                default_act);
        }
    }

    // The policy head.
    const auto policy_extract_channels = weights_->policy_extract_channels;
    auto policy_conv = std::vector<float>(policy_extract_channels * num_intersections);

    Convolution1::Forward(
        board_size, residual_channels, policy_extract_channels,
        conv_out,
        weights_->p_ex_conv.GetWeights(),
        workspace0, policy_conv);

    AddSpatialBiases::Forward(
        board_size, policy_extract_channels,
        policy_conv,
        weights_->p_ex_conv.GetBiases(), default_act);

    GlobalPooling<false>::Forward(
        board_size, policy_extract_channels,
        policy_conv, pooling);

    FullyConnect::Forward(
        3 * policy_extract_channels, policy_extract_channels,
        pooling,
        weights_->p_inter_fc.GetWeights(),
        weights_->p_inter_fc.GetBiases(),
        intermediate, default_act);

    AddSpatialBiases::Forward(
        board_size, policy_extract_channels,
        policy_conv,
        intermediate, Activation::kIdentity);

    // The policy outs.
    Convolution1::Forward(
        board_size, policy_extract_channels, kOuputProbabilitiesChannels,
        policy_conv,
        weights_->prob_conv.GetWeights(),
        workspace0, output_prob);

    AddSpatialBiases::Forward(
        board_size, kOuputProbabilitiesChannels,
        output_prob,
        weights_->prob_conv.GetBiases(), Activation::kIdentity);

    FullyConnect::Forward(
        policy_extract_channels, kOuputPassProbability,
        intermediate,
        weights_->pass_fc.GetWeights(),
        weights_->pass_fc.GetBiases(),
        output_pass, Activation::kIdentity);

    // The value head.
    const auto value_extract_channels = weights_->value_extract_channels;
    auto value_conv = std::vector<float>(value_extract_channels * num_intersections);

    Convolution1::Forward(
        board_size, residual_channels, value_extract_channels,
        conv_out,
        weights_->v_ex_conv.GetWeights(),
        workspace0, value_conv);

    AddSpatialBiases::Forward(
        board_size, value_extract_channels,
        value_conv,
        weights_->v_ex_conv.GetBiases(), default_act);

    GlobalPooling<true>::Forward(
        board_size, value_extract_channels,
        value_conv, pooling);

    FullyConnect::Forward(
        3 * value_extract_channels, 3 * value_extract_channels,
        pooling,
        weights_->v_inter_fc.GetWeights(),
        weights_->v_inter_fc.GetBiases(),
        intermediate, default_act);

    // The value outs.
    Convolution1::Forward(
        board_size, value_extract_channels, kOuputOwnershipChannels,
        value_conv,
        weights_->v_ownership.GetWeights(),
        workspace0, output_ownership);

    AddSpatialBiases::Forward(
        board_size, kOuputOwnershipChannels,
        output_ownership,
        weights_->v_ownership.GetBiases(), Activation::kIdentity);

    FullyConnect::Forward(
        3 * value_extract_channels, kOuputValueMisc,
        intermediate,
        weights_->v_misc.GetWeights(),
        weights_->v_misc.GetBiases(),
        output_misc, Activation::kIdentity);

    // Now copy the result.
    auto result = OutputResult{};

    result.offset = inpnts.offset;
    result.fp16 = false;
    result.board_size = board_size;
    result.komi = inpnts.komi;
    result.wdl[0] = output_misc[0];
    result.wdl[1] = output_misc[1];
    result.wdl[2] = output_misc[2];
    result.stm_winrate = output_misc[3];
    result.final_score = output_misc[8];
    result.q_error = output_misc[13];
    result.score_error = output_misc[14];

    result.pass_probability = output_pass[0];

    auto pol_it = std::begin(output_prob) +
                      (int)inpnts.offset * num_intersections;
    std::copy(
        pol_it,
        pol_it + num_intersections,
        std::begin(result.probabilities));
    std::copy(std::begin(output_ownership),
        std::begin(output_ownership) + num_intersections,
        std::begin(result.ownership));

    return result;
}

bool BlasForwardPipe::Valid() {
    return weights_ != nullptr;
}

void BlasForwardPipe::Release() {}

void BlasForwardPipe::Destroy() {}

void BlasForwardPipe::Reload(int) {}
