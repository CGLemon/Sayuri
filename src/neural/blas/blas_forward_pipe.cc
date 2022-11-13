#include "neural/blas/blas_forward_pipe.h"
#include "neural/blas/convolution.h"
#include "neural/blas/batchnorm.h"
#include "neural/blas/se_unit.h"
#include "neural/blas/fullyconnect.h"
#include "neural/blas/biases.h"
#include "neural/blas/winograd_convolution3.h"
#include "neural/winograd_helper.h"

#include <algorithm>
#include <iostream>

void BlasForwardPipe::Initialize(std::shared_ptr<DNNWeights> weights) {
    Load(weights);
    InitWinograd();
}


void BlasForwardPipe::InitWinograd() {
    if (!weights_->winograd) {
        return;
    }
    if (weights_->winograd_initialized) {
        return;
    }

    const auto residual_channels = weights_->residual_channels;

    // The input layer.
    weights_->input_conv.GetWeights() = 
        WinogradTransformF(weights_->input_conv.GetWeights(),
                               residual_channels, kInputChannels);

    // The residual tower.
    for (auto &residual : weights_->tower) {
        residual.conv1.GetWeights() =
            WinogradTransformF(residual.conv1.GetWeights(),
                                   residual_channels, residual_channels);

        residual.conv2.GetWeights() =
            WinogradTransformF(residual.conv2.GetWeights(),
                                   residual_channels, residual_channels);
    }
    weights_->winograd_initialized = true;
}

void BlasForwardPipe::Load(std::shared_ptr<DNNWeights> weights) {
    weights_ = weights;
}

OutputResult BlasForwardPipe::Forward(const InputData &inpnts) {

    using Convolution3 = Convolution<3>;
    using Convolution1 = Convolution<1>;

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
    auto output_prob = std::vector<float>(num_intersections);
    auto output_pass = std::vector<float>(kOuputPassProbability);
    auto output_ownership = std::vector<float>(num_intersections);
    auto output_misc = std::vector<float>(kOuputValueMisc);

    // The input Layers.
    if (use_winograd) {
        WinogradConvolution3::Forward(board_size, output_channels,
                                      planes,
                                      weights_->input_conv.GetWeights(),
                                      workspace0, workspace1, conv_out);
    } else {
        Convolution3::Forward(board_size, kInputChannels, output_channels,
                              planes,
                              weights_->input_conv.GetWeights(),
                              workspace0, conv_out);
    }

    Batchnorm::Forward(board_size, output_channels,
                       conv_out,
                       weights_->input_bn.GetMeans(),
                       weights_->input_bn.GetStddevs());

    // The residual tower.
    const auto residuals =  weights_->residual_blocks;
    for (int i = 0; i < residuals; ++i) {
        const auto tower_channels = weights_->residual_channels;
        const auto tower_ptr = weights_->tower.data() + i;
        std::swap(conv_in, conv_out);

        // The first conv3.
        if (use_winograd) {
            WinogradConvolution3::Forward(board_size, tower_channels,
                                          conv_in,
                                          tower_ptr->conv1.GetWeights(),
                                          workspace0, workspace1, conv_out);
        } else {
            Convolution3::Forward(board_size, tower_channels, tower_channels,
                                  conv_in,
                                  tower_ptr->conv1.GetWeights(),
                                  workspace0, conv_out);
        }
        Batchnorm::Forward(board_size, tower_channels,
                           conv_out,
                           tower_ptr->bn1.GetMeans(),
                           tower_ptr->bn1.GetStddevs());

        std::swap(conv_in, res);
        std::swap(conv_out, conv_in);

        // The second conv3.
        if (use_winograd) {
            WinogradConvolution3::Forward(board_size, tower_channels,
                                          conv_in,
                                          tower_ptr->conv2.GetWeights(),
                                          workspace0, workspace1, conv_out);
        } else {
            Convolution3::Forward(board_size, tower_channels, tower_channels,
                                  conv_in,
                                  tower_ptr->conv2.GetWeights(),
                                  workspace0, conv_out);
        }

        // The SE process.
        if (tower_ptr->apply_se) {
            Batchnorm::Forward(board_size, tower_channels,
                               conv_out,
                               tower_ptr->bn2.GetMeans(),
                               tower_ptr->bn2.GetStddevs(),
                               nullptr, false);

            const size_t se_size = tower_ptr->se_size;
            SEUnit::Forward(board_size, tower_channels, se_size,
                            conv_out, res,
                            tower_ptr->squeeze.GetWeights(),
                            tower_ptr->squeeze.GetBiases(),
                            tower_ptr->excite.GetWeights(),
                            tower_ptr->excite.GetBiases());
        
        } else {
             Batchnorm::Forward(board_size, tower_channels,
                                conv_out,
                                tower_ptr->bn2.GetMeans(),
                                tower_ptr->bn2.GetStddevs(),
                                res.data());
        }
    }

    // The policy head.
    const auto policy_extract_channels = weights_->policy_extract_channels;
    auto policy_conv = std::vector<float>(policy_extract_channels * num_intersections);

    Convolution1::Forward(board_size, output_channels, policy_extract_channels,
                          conv_out,
                          weights_->p_ex_conv.GetWeights(),
                          workspace0, policy_conv);

    Batchnorm::Forward(board_size, policy_extract_channels,
                       policy_conv,
                       weights_->p_ex_bn.GetMeans(),
                       weights_->p_ex_bn.GetStddevs());


    GlobalPooling<false>::Forward(board_size, policy_extract_channels,
                                  policy_conv, pooling);

    FullyConnect::Forward(3 * policy_extract_channels, policy_extract_channels,
                          pooling,
                          weights_->p_inter_fc.GetWeights(),
                          weights_->p_inter_fc.GetBiases(),
                          intermediate, true);

    AddSpatialBiases::Forward(board_size, policy_extract_channels,
                              policy_conv,
                              intermediate, false);    

    // The policy outs.
    Convolution1::Forward(board_size, policy_extract_channels, kOuputProbabilitiesChannels,
                          policy_conv,
                          weights_->prob_conv.GetWeights(),
                          workspace0, output_prob);

    AddSpatialBiases::Forward(board_size, kOuputProbabilitiesChannels,
                              output_prob,
                              weights_->prob_conv.GetBiases(), false);

    FullyConnect::Forward(policy_extract_channels, kOuputPassProbability,
                          intermediate,
                          weights_->pass_fc.GetWeights(),
                          weights_->pass_fc.GetBiases(),
                          output_pass, false);

    // The value head.
    const auto value_extract_channels = weights_->value_extract_channels;
    auto value_conv = std::vector<float>(value_extract_channels * num_intersections);
    
    Convolution1::Forward(board_size, output_channels, value_extract_channels,
                          conv_out,
                          weights_->v_ex_conv.GetWeights(),
                          workspace0, value_conv);

    Batchnorm::Forward(board_size, value_extract_channels,
                       value_conv,
                       weights_->v_ex_bn.GetMeans(),
                       weights_->v_ex_bn.GetStddevs());

    GlobalPooling<true>::Forward(board_size, value_extract_channels,
                                 value_conv, pooling);

    FullyConnect::Forward(3 * value_extract_channels, 3 * value_extract_channels,
                          pooling,
                          weights_->v_inter_fc.GetWeights(),
                          weights_->v_inter_fc.GetBiases(),
                          intermediate, true);

    // The value outs.
    Convolution1::Forward(board_size, value_extract_channels, kOuputOwnershipChannels,
                          value_conv,
                          weights_->v_ownership.GetWeights(),
                          workspace0, output_ownership);

    AddSpatialBiases::Forward(board_size, kOuputOwnershipChannels,
                              output_ownership,
                              weights_->v_ownership.GetBiases(), false);

    FullyConnect::Forward(3 * value_extract_channels, kOuputValueMisc,
                          intermediate,
                          weights_->v_misc.GetWeights(),
                          weights_->v_misc.GetBiases(),
                          output_misc, false);
    // Now copy the result.
    auto result = OutputResult{};

    result.board_size = board_size;
    result.komi = inpnts.komi;
    result.wdl[0] = output_misc[0];
    result.wdl[1] = output_misc[1];
    result.wdl[2] = output_misc[2];
    result.stm_winrate = output_misc[3];
    result.final_score = output_misc[4];
    result.pass_probability = output_pass[0];

    std::copy(std::begin(output_prob), std::end(output_prob), std::begin(result.probabilities));
    std::copy(std::begin(output_ownership), std::end(output_ownership), std::begin(result.ownership));

    return result;
}

bool BlasForwardPipe::Valid() {
    return weights_ != nullptr;
}

void BlasForwardPipe::Release() {}

void BlasForwardPipe::Destroy() {}

void BlasForwardPipe::Reload(int) {}
