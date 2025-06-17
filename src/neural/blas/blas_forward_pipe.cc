#include "neural/blas/blas_forward_pipe.h"
#include "neural/blas/convolution.h"
#include "neural/blas/batchnorm.h" // not used
#include "neural/blas/se_unit.h"
#include "neural/blas/fullyconnect.h"
#include "neural/blas/biases.h"
#include "neural/blas/winograd_convolution3.h"
#include "neural/winograd_helper.h"
#include "neural/encoder.h"

#include <algorithm>

void BlasForwardPipe::Initialize(std::shared_ptr<DNNWeights> weights) {
    Construct(ForwardPipeParameters::Get(), weights);
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
                               residual_channels, weights_->input_channels);

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
        } else if (block->IsNestedBottleneckBlock()) {
            int channels = block->bottleneck_channels;
            block->conv1.GetWeights() = WinogradTransformF(
                block->conv1.GetWeights(), channels, channels);
            block->conv2.GetWeights() = WinogradTransformF(
                block->conv2.GetWeights(), channels, channels);
            block->conv3.GetWeights() = WinogradTransformF(
                block->conv3.GetWeights(), channels, channels);
            block->conv4.GetWeights() = WinogradTransformF(
                block->conv4.GetWeights(), channels, channels);
        }
    }
    weights_->winograd_initialized = true;
}

void BlasForwardPipe::Convolution3Forward(const int board_size,
                                          const int input_channels,
                                          const int output_channels,
                                          const std::vector<float> &input,
                                          const std::vector<float> &weights,
                                          std::vector<float> &workspace0,
                                          std::vector<float> &workspace1,
                                          std::vector<float> &output) {
    if (weights_->winograd) {
        WinogradConvolution3::Forward(
            board_size, input_channels, output_channels,
            input, weights,
            workspace0, workspace1, output);
    } else {
        Convolution<3>::Forward(
            board_size, input_channels, output_channels,
            input, weights,
            workspace0, output);
    }
}

void BlasForwardPipe::ResidualBlockForward(const int board_size,
                                           BlockBasic * tower_ptr,
                                           std::vector<float> &residual,
                                           std::vector<float> &conv_in,
                                           std::vector<float> &conv_out,
                                           std::vector<float> &workspace0,
                                           std::vector<float> &workspace1) {
    const auto zero_vec = std::vector<float>{};
    const auto channels = weights_->residual_channels;
    const auto default_act = weights_->default_act;

    // 1st conv3x3
    Convolution3Forward(
        board_size,
        channels, channels,
        conv_in,
        tower_ptr->conv1.GetWeights(),
        workspace0, workspace1, conv_out);

    AddSpatialBiases::Forward(
        board_size, channels,
        conv_out,
        tower_ptr->conv1.GetBiases(), default_act);

    std::swap(conv_in, residual);
    std::swap(conv_out, conv_in);

    // 2nd conv3x3
    Convolution3Forward(
        board_size,
        channels, channels,
        conv_in,
        tower_ptr->conv2.GetWeights(),
        workspace0, workspace1, conv_out);

    auto &last_skip = tower_ptr->apply_se ? zero_vec : residual;
    auto last_act = tower_ptr->apply_se ? Activation::kIdentity : default_act;

    AddSpatialBiases::Forward(
        board_size, channels,
        conv_out,
        tower_ptr->conv2.GetBiases(),
        last_skip, last_act);
}

void BlasForwardPipe::BottleneckBlockForward(const int board_size,
                                             BlockBasic * tower_ptr,
                                             std::vector<float> &residual,
                                             std::vector<float> &conv_in,
                                             std::vector<float> &conv_out,
                                             std::vector<float> &workspace0,
                                             std::vector<float> &workspace1) {
    const auto zero_vec = std::vector<float>{};
    const auto outer_channels = weights_->residual_channels;
    const auto inner_channels = tower_ptr->bottleneck_channels;
    const auto default_act = weights_->default_act;

    // The pre-bottleneck conv1x1.
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

    // 1st conv3x3
    Convolution3Forward(
        board_size,
        inner_channels, inner_channels,
        conv_in,
        tower_ptr->conv1.GetWeights(),
        workspace0, workspace1, conv_out);
    AddSpatialBiases::Forward(
        board_size, inner_channels,
        conv_out,
        tower_ptr->conv1.GetBiases(), default_act);

    std::swap(conv_out, conv_in);

    // 2nd conv3x3
    Convolution3Forward(
        board_size,
        inner_channels, inner_channels,
        conv_in,
        tower_ptr->conv2.GetWeights(),
        workspace0, workspace1, conv_out);
    AddSpatialBiases::Forward(
        board_size, inner_channels,
        conv_out,
        tower_ptr->conv2.GetBiases(), default_act);

    std::swap(conv_out, conv_in);

    // The post-bottleneck conv1x1.
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

void BlasForwardPipe::NestedBottleneckBlockForward(const int board_size,
                                                   BlockBasic * tower_ptr,
                                                   std::vector<float> &residual0,
                                                   std::vector<float> &residual1,
                                                   std::vector<float> &conv_in,
                                                   std::vector<float> &conv_out,
                                                   std::vector<float> &workspace0,
                                                   std::vector<float> &workspace1) {
    const auto zero_vec = std::vector<float>{};
    const auto outer_channels = weights_->residual_channels;
    const auto inner_channels = tower_ptr->bottleneck_channels;
    const auto default_act = weights_->default_act;

    // The pre-bottleneck conv1x1.
    Convolution1::Forward(
        board_size, outer_channels, inner_channels,
        conv_in,
        tower_ptr->pre_btl_conv.GetWeights(),
        workspace0, conv_out);
    AddSpatialBiases::Forward(
        board_size, inner_channels,
        conv_out,
        tower_ptr->pre_btl_conv.GetBiases(), default_act);

    std::swap(conv_in, residual0);
    std::swap(conv_out, conv_in);

    // 1st conv3x3
    Convolution3Forward(
        board_size,
        inner_channels, inner_channels,
        conv_in,
        tower_ptr->conv1.GetWeights(),
        workspace0, workspace1, conv_out);
    AddSpatialBiases::Forward(
        board_size, inner_channels,
        conv_out,
        tower_ptr->conv1.GetBiases(), default_act);
    std::swap(conv_in, residual1);
    std::swap(conv_out, conv_in);

    // 2nd conv3x3
    Convolution3Forward(
        board_size,
        inner_channels, inner_channels,
        conv_in,
        tower_ptr->conv2.GetWeights(),
        workspace0, workspace1, conv_out);
    AddSpatialBiases::Forward(
        board_size, inner_channels,
        conv_out,
        tower_ptr->conv2.GetBiases(),
        residual1, default_act);
    std::swap(conv_out, conv_in);

    // 3rd conv3x3
    Convolution3Forward(
        board_size,
        inner_channels, inner_channels,
        conv_in,
        tower_ptr->conv3.GetWeights(),
        workspace0, workspace1, conv_out);
    AddSpatialBiases::Forward(
        board_size, inner_channels,
        conv_out,
        tower_ptr->conv3.GetBiases(), default_act);
    std::swap(conv_in, residual1);
    std::swap(conv_out, conv_in);

    // 4th conv3x3
    Convolution3Forward(
        board_size,
        inner_channels, inner_channels,
        conv_in,
        tower_ptr->conv4.GetWeights(),
        workspace0, workspace1, conv_out);
    AddSpatialBiases::Forward(
        board_size, inner_channels,
        conv_out,
        tower_ptr->conv4.GetBiases(),
        residual1, default_act);
    std::swap(conv_out, conv_in);

    // The post-bottleneck conv1x1.
    Convolution1::Forward(
        board_size, inner_channels, outer_channels,
        conv_in,
        tower_ptr->post_btl_conv.GetWeights(),
        workspace0, conv_out);

    auto &last_skip = tower_ptr->apply_se ? zero_vec : residual0;
    auto last_act = tower_ptr->apply_se ? Activation::kIdentity : default_act;

    AddSpatialBiases::Forward(
        board_size, outer_channels,
        conv_out,
        tower_ptr->post_btl_conv.GetBiases(),
        last_skip, last_act);
}

void BlasForwardPipe::MixerBlockForward(const int board_size,
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
    const auto max_channels = std::max({weights_->input_channels,
                                        tower_peak_channels,
                                        weights_->policy_head_channels,
                                        weights_->value_head_channels});
    const auto plane_size = weights_->input_channels * num_intersections;
    const auto max_intermediates = std::max(weights_->policy_head_channels,
                                                weights_->value_head_channels);
    const auto default_act = weights_->default_act;

    // Allocate the forward pipe buffers.
    int workspace0_size = 0;
    int workspace1_size = 0;

    if (weights_->winograd) {
        workspace0_size =
            workspace1_size =
            WinogradConvolution3::GetWorkspaceSize(board_size, max_channels);
    } else {
        workspace0_size =
            Convolution<3>::GetWorkspaceSize(board_size, max_channels);
        workspace1_size = 1; // not used.
    }

    auto workspace0 = std::vector<float>(workspace0_size);
    auto workspace1 = std::vector<float>(workspace1_size);

    auto conv_out = std::vector<float>(tower_peak_channels * num_intersections);
    auto conv_in = std::vector<float>(tower_peak_channels * num_intersections);
    auto residual0 = std::vector<float>(tower_peak_channels * num_intersections);
    auto residual1 = std::vector<float>(tower_peak_channels * num_intersections);
    auto intermediate = std::vector<float>(3 * max_intermediates);
    auto pooling = std::vector<float>(3 * max_intermediates);

    // Copy input plane to buffer.
    auto planes = std::vector<float>(plane_size);
    std::copy(std::begin(inpnts.planes),
                  std::begin(inpnts.planes) + plane_size,
                  std::begin(planes));

    // Allocate the output buffers.
    auto output_prob = std::vector<float>(weights_->probabilities_channels * num_intersections);
    auto output_pass = std::vector<float>(weights_->pass_probability_outputs);
    auto output_ownership = std::vector<float>(weights_->ownership_channels * num_intersections);
    auto output_misc = std::vector<float>(weights_->value_misc_outputs);

    // The input Layers.
    Convolution3Forward(
        board_size,
        weights_->input_channels, residual_channels,
        planes,
        weights_->input_conv.GetWeights(),
        workspace0, workspace1, conv_out);
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
                board_size, tower_ptr, residual0,
                conv_in, conv_out, workspace0, workspace1);
        } else if (tower_ptr->IsBottleneckBlock()) {
            BottleneckBlockForward(
                board_size, tower_ptr, residual0,
                conv_in, conv_out, workspace0, workspace1);
        } else if (tower_ptr->IsNestedBottleneckBlock()) {
            NestedBottleneckBlockForward(
                board_size, tower_ptr, residual0, residual1,
                conv_in, conv_out, workspace0, workspace1);
        } else if (tower_ptr->IsMixerBlock()) {
            MixerBlockForward(
                board_size, tower_ptr, residual0,
                conv_in, conv_out);
        }

        // The SE process.
        if (tower_ptr->apply_se) {
            auto &se_skip = residual0;
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
    const auto policy_head_channels = weights_->policy_head_channels;
    auto policy_conv = std::vector<float>(policy_head_channels * num_intersections);

    Convolution1::Forward(
        board_size, residual_channels, policy_head_channels,
        conv_out,
        weights_->p_hd_conv.GetWeights(),
        workspace0, policy_conv);

    AddSpatialBiases::Forward(
        board_size, policy_head_channels,
        policy_conv,
        weights_->p_hd_conv.GetBiases(), default_act);

    if (weights_->policy_head_type == PolicyHeadType::kRepLK) {
        auto policy_conv_buf = std::vector<float>(policy_head_channels * num_intersections);
        const auto filter_size =  weights_->p_dw_conv.GetFilter();

        DepthwiseConvolution::Forward(
            board_size, filter_size, policy_head_channels,
            policy_conv, weights_->p_dw_conv.GetWeights(), policy_conv_buf);
        AddSpatialBiases::Forward(
            board_size, policy_head_channels,
            policy_conv_buf, weights_->p_dw_conv.GetBiases(), default_act);
        Convolution1::Forward(
            board_size, policy_head_channels, policy_head_channels,
            policy_conv_buf,
            weights_->p_pt_conv.GetWeights(),
            workspace0, policy_conv);
        AddSpatialBiases::Forward(
            board_size, policy_head_channels,
            policy_conv,
            weights_->p_pt_conv.GetBiases(), default_act);
    }

    GlobalPooling<false>::Forward(
        board_size, policy_head_channels,
        policy_conv, pooling);

    FullyConnect::Forward(
        3 * policy_head_channels, policy_head_channels,
        pooling,
        weights_->p_inter_fc.GetWeights(),
        weights_->p_inter_fc.GetBiases(),
        intermediate, default_act);

    AddSpatialBiases::Forward(
        board_size, policy_head_channels,
        policy_conv,
        intermediate, Activation::kIdentity);

    // The policy outs.
    Convolution1::Forward(
        board_size, policy_head_channels, weights_->probabilities_channels,
        policy_conv,
        weights_->prob_conv.GetWeights(),
        workspace0, output_prob);

    AddSpatialBiases::Forward(
        board_size, weights_->probabilities_channels,
        output_prob,
        weights_->prob_conv.GetBiases(), Activation::kIdentity);

    FullyConnect::Forward(
        policy_head_channels, weights_->pass_probability_outputs,
        intermediate,
        weights_->pass_fc.GetWeights(),
        weights_->pass_fc.GetBiases(),
        output_pass, Activation::kIdentity);

    // The value head.
    const auto value_head_channels = weights_->value_head_channels;
    auto value_conv = std::vector<float>(value_head_channels * num_intersections);

    Convolution1::Forward(
        board_size, residual_channels, value_head_channels,
        conv_out,
        weights_->v_hd_conv.GetWeights(),
        workspace0, value_conv);

    AddSpatialBiases::Forward(
        board_size, value_head_channels,
        value_conv,
        weights_->v_hd_conv.GetBiases(), default_act);

    GlobalPooling<true>::Forward(
        board_size, value_head_channels,
        value_conv, pooling);

    FullyConnect::Forward(
        3 * value_head_channels, 3 * value_head_channels,
        pooling,
        weights_->v_inter_fc.GetWeights(),
        weights_->v_inter_fc.GetBiases(),
        intermediate, default_act);

    // The value outs.
    Convolution1::Forward(
        board_size, value_head_channels, weights_->ownership_channels,
        value_conv,
        weights_->v_ownership.GetWeights(),
        workspace0, output_ownership);

    AddSpatialBiases::Forward(
        board_size, weights_->ownership_channels,
        output_ownership,
        weights_->v_ownership.GetBiases(), Activation::kIdentity);

    FullyConnect::Forward(
        3 * value_head_channels, weights_->value_misc_outputs,
        intermediate,
        weights_->v_misc.GetWeights(),
        weights_->v_misc.GetBiases(),
        output_misc, Activation::kIdentity);

    // Now copy the result.
    auto result = OutputResult{};

    FillOutputs(output_prob,
                output_pass,
                output_misc,
                output_ownership,
                inpnts, result);

    return result;
}

void BlasForwardPipe::FillOutputs(const std::vector<float> &output_prob,
                                  const std::vector<float> &output_pass,
                                  const std::vector<float> &output_misc,
                                  const std::vector<float> &output_ownership,
                                  const InputData &inpnts,
                                  OutputResult &output) {
    const auto board_size = inpnts.board_size;
    const auto num_intersections = board_size * board_size;
    const auto encoder_version = Encoder::GetEncoderVersion(weights_->version); 

    if (encoder_version == 1) {
        std::copy(
            std::begin(output_prob),
            std::begin(output_prob) + num_intersections,
            std::begin(output.probabilities));
        output.pass_probability = output_pass[0];

        output.wdl[0]      = output_misc[0];
        output.wdl[1]      = output_misc[1];
        output.wdl[2]      = output_misc[2];
        output.stm_winrate = output_misc[3];
        output.final_score = output_misc[4];
        output.q_error     = 0.0f;
        output.score_error = 0.0f;

        std::copy(std::begin(output_ownership),
            std::begin(output_ownership) + num_intersections,
            std::begin(output.ownership));

        output.offset = PolicyBufferOffset::kNormal;
        output.board_size = board_size;
        output.komi = inpnts.komi;
        output.fp16 = false;
    } else if (encoder_version == 2) {
        auto pol_it = std::begin(output_prob) +
                          (int)inpnts.offset * num_intersections;
        std::copy(
            pol_it,
            pol_it + num_intersections,
            std::begin(output.probabilities));
        output.pass_probability = output_pass[(int)inpnts.offset];

        output.wdl[0]      = output_misc[0];
        output.wdl[1]      = output_misc[1];
        output.wdl[2]      = output_misc[2];
        output.stm_winrate = output_misc[3];
        output.final_score = output_misc[8];
        output.q_error     = output_misc[13];
        output.score_error = output_misc[14];

        std::copy(std::begin(output_ownership),
            std::begin(output_ownership) + num_intersections,
            std::begin(output.ownership));

        output.offset = inpnts.offset;
        output.board_size = board_size;
        output.komi = inpnts.komi;
        output.fp16 = false;
    }
}

bool BlasForwardPipe::Valid() const {
    return weights_ != nullptr;
}

void BlasForwardPipe::Release() {}

void BlasForwardPipe::Destroy() {}

void BlasForwardPipe::Construct(ForwardPipeParameters /* param */,
                                std::shared_ptr<DNNWeights> weights) {
    if (weights) {
        weights_ = weights;
        InitWinograd();
    }
}
