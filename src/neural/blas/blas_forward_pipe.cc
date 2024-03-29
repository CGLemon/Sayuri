#include "neural/blas/blas_forward_pipe.h"
#include "neural/blas/convolution.h"
#include "neural/blas/batchnorm.h"
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
    use_optimistic_policy_ = GetOption<bool>("use_optimistic_policy");
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

    // The residual tower.
    for (auto &residual : weights_->tower) {
        const auto outer_channels = residual_channels;
        const auto inner_channels = residual.apply_btl ?
                                        outer_channels/2 :
                                        outer_channels;
        residual.conv1.GetWeights() =
            WinogradTransformF(residual.conv1.GetWeights(),
                                   inner_channels, inner_channels);

        residual.conv2.GetWeights() =
            WinogradTransformF(residual.conv2.GetWeights(),
                                   inner_channels, inner_channels);
    }
    weights_->winograd_initialized = true;
}

void BlasForwardPipe::Load(std::shared_ptr<DNNWeights> weights) {
    weights_ = weights;
}

OutputResult BlasForwardPipe::Forward(const InputData &inpnts) {

    using Convolution3 = Convolution<3>;

    // Some useful information for network.
    const auto board_size = inpnts.board_size;
    const auto num_intersections = board_size * board_size;
    const auto output_channels = weights_->residual_channels;
    const auto max_channels = std::max({kInputChannels,
                                        output_channels,
                                        weights_->policy_extract_channels,
                                        weights_->value_extract_channels});
    const auto plane_size = kInputChannels * num_intersections;
    const auto max_intermediates = std::max(weights_->policy_extract_channels,
                                                weights_->value_extract_channels);
    const auto zero_vec = std::vector<float>{};

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

    auto conv_out = std::vector<float>(output_channels * num_intersections);
    auto conv_in = std::vector<float>(output_channels * num_intersections);
    auto res = std::vector<float>(output_channels * num_intersections);
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
            board_size, kInputChannels, output_channels,
            planes,
            weights_->input_conv.GetWeights(),
            workspace0, workspace1, conv_out);
    } else {
        Convolution3::Forward(
            board_size, kInputChannels, output_channels,
            planes,
            weights_->input_conv.GetWeights(),
            workspace0, conv_out);
    }

    AddSpatialBiases::Forward(
        board_size, output_channels,
        conv_out,
        weights_->input_conv.GetBiases(), true);

    // The residual tower.
    const auto residuals =  weights_->residual_blocks;
    for (int i = 0; i < residuals; ++i) {
        const auto tower_ptr = weights_->tower.data() + i;
        const auto outer_channels = weights_->residual_channels;
        const auto inner_channels = tower_ptr->apply_btl ?
                                        outer_channels/2 :
                                        outer_channels;
        if (tower_ptr->apply_btl) {
            std::swap(conv_out, conv_in);

            // The pre-bottleneck conv1.
            Convolution1::Forward(
                board_size, outer_channels, inner_channels,
                conv_in,
                tower_ptr->pre_btl_conv.GetWeights(),
                workspace0, conv_out);
            AddSpatialBiases::Forward(
                board_size, inner_channels,
                conv_out,
                tower_ptr->pre_btl_conv.GetBiases(), true);

            std::swap(conv_in, res);
        }

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
            tower_ptr->conv1.GetBiases(), true);

        if (!(tower_ptr->apply_btl)) {
            std::swap(conv_in, res);
        }

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

        if (tower_ptr->apply_btl) {
            AddSpatialBiases::Forward(
                board_size, inner_channels,
                conv_out,
                tower_ptr->conv2.GetBiases(), true);

            std::swap(conv_out, conv_in);

            // The post-bottleneck conv1.
            Convolution1::Forward(
                board_size, inner_channels, outer_channels,
                conv_in,
                tower_ptr->post_btl_conv.GetWeights(),
                workspace0, conv_out);
        }

        auto &last_biases = tower_ptr->apply_btl ?
                                tower_ptr->post_btl_conv.GetBiases() :
                                tower_ptr->conv2.GetBiases();
        auto &last_skip = tower_ptr->apply_se ? zero_vec : res;
        bool last_relu = !(tower_ptr->apply_se);

        AddSpatialBiases::Forward(
            board_size, outer_channels,
            conv_out,
            last_biases,
            last_skip, last_relu);

        // The SE process.
        if (tower_ptr->apply_se) {
            auto &se_skip = res;

            const size_t se_size = tower_ptr->se_size;
            SEUnit::Forward(
                board_size, outer_channels, se_size,
                conv_out, se_skip,
                tower_ptr->squeeze.GetWeights(),
                tower_ptr->squeeze.GetBiases(),
                tower_ptr->excite.GetWeights(),
                tower_ptr->excite.GetBiases(), true);
        }
    }

    // The policy head.
    const auto policy_extract_channels = weights_->policy_extract_channels;
    auto policy_conv = std::vector<float>(policy_extract_channels * num_intersections);

    Convolution1::Forward(
        board_size, output_channels, policy_extract_channels,
        conv_out,
        weights_->p_ex_conv.GetWeights(),
        workspace0, policy_conv);

    AddSpatialBiases::Forward(
        board_size, policy_extract_channels,
        policy_conv,
        weights_->p_ex_conv.GetBiases(), true);

    GlobalPooling<false>::Forward(
        board_size, policy_extract_channels,
        policy_conv, pooling);

    FullyConnect::Forward(
        3 * policy_extract_channels, policy_extract_channels,
        pooling,
        weights_->p_inter_fc.GetWeights(),
        weights_->p_inter_fc.GetBiases(),
        intermediate, true);

    AddSpatialBiases::Forward(
        board_size, policy_extract_channels,
        policy_conv,
        intermediate, false);

    // The policy outs.
    Convolution1::Forward(
        board_size, policy_extract_channels, kOuputProbabilitiesChannels,
        policy_conv,
        weights_->prob_conv.GetWeights(),
        workspace0, output_prob);

    AddSpatialBiases::Forward(
        board_size, kOuputProbabilitiesChannels,
        output_prob,
        weights_->prob_conv.GetBiases(), false);

    FullyConnect::Forward(
        policy_extract_channels, kOuputPassProbability,
        intermediate,
        weights_->pass_fc.GetWeights(),
        weights_->pass_fc.GetBiases(),
        output_pass, false);

    // The value head.
    const auto value_extract_channels = weights_->value_extract_channels;
    auto value_conv = std::vector<float>(value_extract_channels * num_intersections);

    Convolution1::Forward(
        board_size, output_channels, value_extract_channels,
        conv_out,
        weights_->v_ex_conv.GetWeights(),
        workspace0, value_conv);

    AddSpatialBiases::Forward(
        board_size, value_extract_channels,
        value_conv,
        weights_->v_ex_conv.GetBiases(), true);

    GlobalPooling<true>::Forward(
        board_size, value_extract_channels,
        value_conv, pooling);

    FullyConnect::Forward(
        3 * value_extract_channels, 3 * value_extract_channels,
        pooling,
        weights_->v_inter_fc.GetWeights(),
        weights_->v_inter_fc.GetBiases(),
        intermediate, true);

    // The value outs.
    Convolution1::Forward(
        board_size, value_extract_channels, kOuputOwnershipChannels,
        value_conv,
        weights_->v_ownership.GetWeights(),
        workspace0, output_ownership);

    AddSpatialBiases::Forward(
        board_size, kOuputOwnershipChannels,
        output_ownership,
        weights_->v_ownership.GetBiases(), false);

    FullyConnect::Forward(
        3 * value_extract_channels, kOuputValueMisc,
        intermediate,
        weights_->v_misc.GetWeights(),
        weights_->v_misc.GetBiases(),
        output_misc, false);

    // Now copy the result.
    auto result = OutputResult{};

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

    auto pol_it = std::begin(output_prob);
    if (!use_optimistic_policy_) {
        std::copy(
            pol_it,
            pol_it + num_intersections,
            std::begin(result.probabilities));
    } else {
        pol_it += 4 * num_intersections;
        std::copy(
            pol_it,
            pol_it + num_intersections,
            std::begin(result.probabilities));
    }
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
