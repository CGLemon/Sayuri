#include "neural/blas/blas_forward_pipe.h"
#include "neural/blas/convolution.h"
#include "neural/blas/batchnorm.h"
#include "neural/blas/se_unit.h"
#include "neural/blas/fullyconnect.h"
#include "neural/blas/biases.h"

#include <algorithm>
#include <iostream>
void BlasForwardPipe::Initialize(std::shared_ptr<DNNWeights> weights) {
    Load(weights);
}

void BlasForwardPipe::Load(std::shared_ptr<DNNWeights> weights) {
    weights_ = weights;
}

OutputResult BlasForwardPipe::Forward(const InputData &inpnts) {

    using Convolution3 = Convolution<3>;
    using Convolution1 = Convolution<1>;
    using Pooling = GlobalAvgPool;

    // Some useful information for network.
    const auto board_size = inpnts.board_size;
    const auto num_intersections = board_size * board_size;
    const auto output_channels = weights_->residual_channels;
    const auto max_channels = std::max({kInputChannels,
                                        output_channels,
                                        weights_->policy_extract_channels});
    const auto plane_size = kInputChannels * num_intersections;


    // Allocate the forward pipe buffers. 
    auto workspace = std::vector<float>(Convolution3::GetWorkspaceSize(board_size, max_channels));
    auto conv_out = std::vector<float>(output_channels * num_intersections);
    auto conv_in = std::vector<float>(output_channels * num_intersections);
    auto res = std::vector<float>(output_channels * num_intersections);

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

    // input Layers

    Convolution3::Forward(board_size, kInputChannels, output_channels,
                          planes,
                          weights_->input_conv.GetWeights(),
                          workspace, conv_out);

    Batchnorm::Forward(board_size, output_channels,
                       conv_out,
                       weights_->input_bn.GetMeans(),
                       weights_->input_bn.GetStddevs());

    // residual tower
    const auto residuals =  weights_->residual_blocks;
    for (int i = 0; i < residuals; ++i) {
        const auto tower_channels = weights_->residual_channels;
        const auto tower_ptr = weights_->tower.data() + i;

        std::swap(conv_in, conv_out);

        Convolution3::Forward(board_size, tower_channels, tower_channels,
                              conv_in,
                              tower_ptr->conv1.GetWeights(),
                              workspace, conv_out);

        Batchnorm::Forward(board_size, tower_channels,
                           conv_out,
                           tower_ptr->bn1.GetMeans(),
                           tower_ptr->bn1.GetStddevs());

        std::swap(conv_in, res);
        std::swap(conv_out, conv_in);

        Convolution3::Forward(board_size, tower_channels, tower_channels,
                              conv_in,
                              tower_ptr->conv2.GetWeights(),
                              workspace, conv_out);

        if (tower_ptr->apply_se) {
            Batchnorm::Forward(board_size, tower_channels,
                               conv_out,
                               tower_ptr->bn2.GetMeans(),
                               tower_ptr->bn2.GetStddevs(),
                               nullptr, false);

            const size_t se_size = tower_ptr->se_size;
            SEUnit::Forward(board_size, tower_channels, se_size,
                            conv_out, res,
                            tower_ptr->extend.GetWeights(),
                            tower_ptr->extend.GetBiases(),
                            tower_ptr->squeeze.GetWeights(),
                            tower_ptr->squeeze.GetBiases());
        
        } else {
             Batchnorm::Forward(board_size, tower_channels,
                                conv_out,
                                tower_ptr->bn2.GetMeans(),
                                tower_ptr->bn2.GetStddevs(),
                                res.data());
        }
    }

    // policy head
    const auto policy_extract_channels = weights_->policy_extract_channels;
    auto policy_conv = std::vector<float>(policy_extract_channels * num_intersections);
    auto policy_layer = std::vector<float>(policy_extract_channels);

    Convolution1::Forward(board_size, output_channels, policy_extract_channels,
                          conv_out,
                          weights_->p_ex_conv.GetWeights(),
                          workspace, policy_conv);

    Batchnorm::Forward(board_size, policy_extract_channels,
                       policy_conv,
                       weights_->p_ex_bn.GetMeans(),
                       weights_->p_ex_bn.GetStddevs());

    Convolution1::Forward(board_size, policy_extract_channels, 1,
                          policy_conv,
                          weights_->prob_conv.GetWeights(),
                          workspace, output_prob);

    AddSpatialBiases::Forward(board_size, 1,
                              output_prob,
                              weights_->prob_conv.GetBiases(), false);
    Pooling::Forward(board_size, policy_extract_channels, policy_conv, policy_layer);

    FullyConnect::Forward(policy_extract_channels, kOuputPassProbability,
                          policy_layer,
                          weights_->pass_fc.GetWeights(),
                          weights_->pass_fc.GetBiases(),
                          output_pass, false);

    // value head
    const auto value_extract_channels = weights_->value_extract_channels;
    auto value_conv = std::vector<float>(value_extract_channels * num_intersections);
    auto value_layer = std::vector<float>(value_extract_channels);
    
    Convolution1::Forward(board_size, output_channels, value_extract_channels,
                          conv_out,
                          weights_->v_ex_conv.GetWeights(),
                          workspace, value_conv);
    
    Batchnorm::Forward(board_size, value_extract_channels,
                       value_conv,
                       weights_->v_ex_bn.GetMeans(),
                       weights_->v_ex_bn.GetStddevs());

    Convolution1::Forward(board_size, value_extract_channels, 1,
                          value_conv,
                          weights_->v_ownership.GetWeights(),
                          workspace, output_ownership);
    
    AddSpatialBiases::Forward(board_size, 1,
                                  output_ownership,
                                  weights_->v_ownership.GetBiases(), false);


    Pooling::Forward(board_size, value_extract_channels, value_conv, value_layer);

    FullyConnect::Forward(value_extract_channels, kOuputValueMisc,
                          value_layer,
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
