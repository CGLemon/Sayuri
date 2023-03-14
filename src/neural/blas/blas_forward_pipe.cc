#include "neural/blas/blas_forward_pipe.h"
#include "neural/blas/convolution.h"
#include "neural/blas/batchnorm.h"
#include "neural/blas/se_unit.h"
#include "neural/blas/fullyconnect.h"
#include "neural/blas/biases.h"
#include "neural/blas/winograd_convolution3.h"
#include "neural/winograd_helper.h"

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

        // The SE process.
        if (tower_ptr->apply_se) {
            if (tower_ptr->apply_btl) {
                AddSpatialBiases::Forward(
                    board_size, outer_channels,
                    conv_out,
                    tower_ptr->post_btl_conv.GetBiases(), false);
            } else {
                // The 'outer_channels' is equal to 'inner_channels'.
                AddSpatialBiases::Forward(
                    board_size, outer_channels,
                    conv_out,
                    tower_ptr->conv2.GetBiases(), false);
            }
            const size_t se_size = tower_ptr->se_size;
            SEUnit::Forward(
                board_size, outer_channels, se_size,
                conv_out, res,
                tower_ptr->squeeze.GetWeights(),
                tower_ptr->squeeze.GetBiases(),
                tower_ptr->excite.GetWeights(),
                tower_ptr->excite.GetBiases());
        
        } else {
            if (tower_ptr->apply_btl) {
                AddSpatialBiases::Forward(
                    board_size, outer_channels,
                    conv_out,
                    tower_ptr->post_btl_conv.GetBiases(),
                    res, true);
            } else {
                // The 'outer_channels' is equal to 'inner_channels'.
                AddSpatialBiases::Forward(
                    board_size, outer_channels,
                    conv_out,
                    tower_ptr->conv2.GetBiases(),
                    res, true);
            }
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

/*
===================================input
0.317133 0.817466 0.817466 0.817466 0.817466 0.817466 0.817466 0.817466 0.817466 0.817466 0.817466 0.817466 0.817466 0.817466 0.817466 0.817466 0.817466 0.817466 0.882624 0.372575 0.768428 0.768428 0.768428 0.768428 0.768428 0.768428 0.768428 0.768428 0.768428 0.768428 0.768428 0.768428 0.768428 0.768428 0.768428 0.768428 0.768428 0.659478 0.372575 0.768428 0.768428 0.768428 0.768428 0.768428 0.768428 0.768428 0.768428 0.768428 0.768428 0.768428 0.768428 0.768428 0.768428 0.768428 0.768428 0.768428 0.659478 0.372575 0.768428 0.768428 0.768428 0.768428 0.768428 0.768428 0.768428 0.768428 0.768428 0.768428 0.768428 0.768428 0.768428 0.768428 0.768428 0.768428 0.768428 0.659478 0.372575 0.768428 0.768428 0.768428 0.768428 0.768428 0.768428 0.768428 0.768428 0.768428 0.768428 0.768428 0.768428 0.768428 0.768428 0.768428 0.768428 0.768428 0.659478 0.372575 0.768428 0.768428 0.768428 0.768428 

===================================1st
0.217549 -0.093031 -0.131262 -0.131262 -0.131262 -0.131262 -0.131262 -0.131262 -0.131262 -0.131262 -0.131262 -0.131262 -0.131262 -0.131262 -0.131262 -0.131262 -0.131262 0.091298 0.055421 -0.255990 -0.934063 -0.651144 -0.651144 -0.651144 -0.651144 -0.651144 -0.651144 -0.651144 -0.651144 -0.651144 -0.651144 -0.651144 -0.651144 -0.651144 -0.651144 -0.651144 -0.570679 -0.872654 -0.225737 -0.950486 -0.719699 -0.719699 -0.719699 -0.719699 -0.719699 -0.719699 -0.719699 -0.719699 -0.719699 -0.719699 -0.719699 -0.719699 -0.719699 -0.719699 -0.719699 -0.665872 -0.894532 -0.225737 -0.950486 -0.719699 -0.719699 -0.719699 -0.719699 -0.719699 -0.719699 -0.719699 -0.719699 -0.719699 -0.719699 -0.719699 -0.719699 -0.719699 -0.719699 -0.719699 -0.665872 -0.894532 -0.225737 -0.950486 -0.719699 -0.719699 -0.719699 -0.719699 -0.719699 -0.719699 -0.719699 -0.719699 -0.719699 -0.719699 -0.719699 -0.719699 -0.719699 -0.719699 -0.719699 -0.665872 -0.894532 -0.225737 -0.950486 -0.719699 -0.719699 -0.719699 

===================================2nd
-0.250877 -0.233126 -0.209352 -0.160949 -0.160949 -0.160949 -0.160949 -0.160949 -0.160949 -0.160949 -0.160949 -0.160949 -0.160949 -0.160949 -0.160949 -0.160949 -0.079717 -0.098287 0.340154 -0.867988 -0.802308 -0.834015 -0.854408 -0.854408 -0.854408 -0.854408 -0.854408 -0.854408 -0.854408 -0.854408 -0.854408 -0.854408 -0.854408 -0.854408 -0.854408 -0.783988 -0.844178 -0.103944 -0.883818 -1.067536 -0.889396 -0.931840 -0.931840 -0.931840 -0.931840 -0.931840 -0.931840 -0.931840 -0.931840 -0.931840 -0.931840 -0.931840 -0.931840 -0.931840 -0.849958 -0.935686 -0.160967 -0.850671 -1.040717 -0.900269 -0.913401 -0.913401 -0.913401 -0.913401 -0.913401 -0.913401 -0.913401 -0.913401 -0.913401 -0.913401 -0.913401 -0.913401 -0.913401 -0.842084 -0.935816 -0.167624 -0.850671 -1.040717 -0.900269 -0.913401 -0.913401 -0.913401 -0.913401 -0.913401 -0.913401 -0.913401 -0.913401 -0.913401 -0.913401 -0.913401 -0.913401 -0.913401 -0.842084 -0.935816 -0.167624 -0.850671 -1.040717 -0.900269 -0.913401 -0.913401 

===================================1st
0.368722 0.505630 0.755241 0.788560 0.766491 0.766491 0.766491 0.766491 0.766491 0.766491 0.766491 0.766491 0.766491 0.766491 0.766491 0.736004 0.721817 0.410139 -0.053901 0.565588 1.100939 1.453793 1.592035 1.580665 1.580665 1.580665 1.580665 1.580665 1.580665 1.580665 1.580665 1.580665 1.580665 1.580665 1.549081 1.458544 1.154300 -0.200909 0.704382 1.202250 1.516864 1.739158 1.735715 1.735715 1.735715 1.735715 1.735715 1.735715 1.735715 1.735715 1.735715 1.735715 1.735715 1.739142 1.627979 1.154726 -0.209557 0.714587 1.163516 1.458956 1.668523 1.652523 1.652523 1.652523 1.652523 1.652523 1.652523 1.652523 1.652523 1.652523 1.652523 1.652523 1.652533 1.596203 1.100696 -0.187797 0.686701 1.078135 1.359226 1.566576 1.547670 1.547670 1.547670 1.547670 1.547670 1.547670 1.547670 1.547670 1.547670 1.547670 1.547670 1.547601 1.520678 1.058223 -0.199585 0.686701 1.078135 1.359226 1.566576 1.547670 

===================================2nd
-0.491867 -0.299717 -0.069684 -0.210552 -0.375178 -0.388087 -0.388087 -0.388087 -0.388087 -0.388087 -0.388087 -0.388087 -0.388087 -0.388087 -0.418191 -0.388373 -0.330790 -0.232786 0.535085 -0.548967 -0.317512 -0.147224 -0.332692 -0.444034 -0.443607 -0.443607 -0.443607 -0.443607 -0.443607 -0.443607 -0.443607 -0.443607 -0.443607 -0.453843 -0.406547 -0.460718 -0.066615 0.703915 -0.372866 -0.098037 0.288179 0.105074 -0.075820 -0.082451 -0.082451 -0.082451 -0.082451 -0.082451 -0.082451 -0.082451 -0.082451 -0.082451 -0.082209 -0.043088 -0.236170 0.061353 0.680772 -0.492000 -0.147648 0.127297 -0.084667 -0.258092 -0.235564 -0.235564 -0.235564 -0.235564 -0.235564 -0.235564 -0.235564 -0.235564 -0.235564 -0.239684 -0.216541 -0.405731 -0.152053 0.570777 -0.424804 -0.119268 0.215264 -0.088116 -0.243416 -0.219284 -0.219284 -0.219284 -0.219284 -0.219284 -0.219284 -0.219284 -0.219284 -0.219284 -0.225942 -0.218715 -0.413057 -0.195284 0.531618 -0.441405 -0.154223 0.196512 -0.093485 -0.239784 

===================================1st
-1.175503 -1.222471 -0.972173 -1.039606 -0.982689 -0.976308 -0.979574 -0.979574 -0.979574 -0.979574 -0.979574 -0.979574 -0.979574 -0.983730 -0.985597 -0.910724 -0.806808 -0.511885 -0.557456 -1.090326 -1.242721 -0.994707 -0.987084 -0.938473 -0.924823 -0.937057 -0.937057 -0.937057 -0.937057 -0.937057 -0.937057 -0.937057 -0.939237 -0.936077 -0.897297 -0.708914 -0.359720 -0.700314 -1.444564 -1.527921 -1.230639 -1.224390 -1.234338 -1.234127 -1.245397 -1.245397 -1.245397 -1.245397 -1.245397 -1.245397 -1.245397 -1.251633 -1.256851 -1.221053 -0.935559 -0.609968 -0.586094 -1.349240 -1.494850 -1.166197 -1.145526 -1.169152 -1.177047 -1.188479 -1.188479 -1.188479 -1.188479 -1.188479 -1.188479 -1.188479 -1.193439 -1.203647 -1.201019 -0.917468 -0.593927 -0.580425 -1.354940 -1.438442 -1.090887 -1.077158 -1.086953 -1.089984 -1.098578 -1.098578 -1.098578 -1.098578 -1.098578 -1.098578 -1.098578 -1.102087 -1.115184 -1.110991 -0.818775 -0.475223 -0.442088 -1.352114 -1.425116 -1.084034 -1.075261 -1.082420 

===================================2nd
-0.554823 0.305975 0.227670 -0.018923 -0.023169 0.053761 0.048218 0.046035 0.046035 0.046035 0.046035 0.046035 0.043230 0.039372 0.037531 0.014930 0.045315 0.199181 0.711548 -0.242971 1.080872 0.547776 0.314749 0.472087 0.510535 0.508222 0.508204 0.508204 0.508204 0.508204 0.508204 0.501687 0.503053 0.551927 0.634629 0.462555 0.267278 0.420950 -0.454867 0.875750 0.600343 0.481272 0.644101 0.656520 0.629027 0.628488 0.628488 0.628488 0.628488 0.628488 0.621711 0.628726 0.696387 0.860664 0.688149 0.255887 0.736609 -0.662994 0.470039 0.262603 0.176451 0.444761 0.463902 0.437071 0.443681 0.443681 0.443681 0.443681 0.443681 0.438556 0.449068 0.552099 0.718365 0.591560 0.031591 0.614114 -0.691428 0.465808 0.278313 0.157691 0.427561 0.458443 0.438978 0.445445 0.445445 0.445445 0.445445 0.445445 0.440316 0.451088 0.580268 0.795824 0.730361 0.198823 0.756271 -0.812159 0.322148 0.085566 -0.039672 0.219136 

===================================1st
0.304159 1.078055 1.557661 1.655132 1.455541 1.447248 1.451437 1.461861 1.463749 1.463749 1.463749 1.463425 1.463753 1.455057 1.444665 1.369812 1.434665 1.364987 1.223403 0.850110 1.383435 2.095521 2.277566 2.205622 2.219342 2.241573 2.246176 2.247803 2.247803 2.247803 2.246439 2.249528 2.256244 2.333245 2.197320 2.153420 2.031728 1.139929 1.178812 1.578038 2.025673 2.062110 1.915355 1.878263 1.930519 1.944241 1.947196 1.947196 1.947196 1.946274 1.947134 1.956534 2.045661 1.871741 1.737078 1.614152 1.066726 1.038571 1.634417 1.978610 1.927486 1.837222 1.782882 1.822281 1.831076 1.835059 1.835059 1.835059 1.834994 1.837269 1.855236 1.924709 1.797137 1.772418 1.525108 1.054208 0.939856 1.619196 1.940305 1.813503 1.704779 1.684548 1.739761 1.744237 1.746530 1.746530 1.746530 1.746724 1.751070 1.773223 1.828817 1.703603 1.694171 1.530034 0.984082 0.945001 1.629344 1.922747 1.807990 1.719567 

===================================2nd
-1.679700 -1.780264 -2.157001 -2.709488 -2.747494 -2.709878 -2.674551 -2.688681 -2.687320 -2.686016 -2.685363 -2.684761 -2.687551 -2.660562 -2.637661 -2.510982 -2.491891 -1.499354 -0.740695 -0.601300 -0.936685 -1.570727 -1.864782 -1.593473 -1.562678 -1.565182 -1.589483 -1.591650 -1.591671 -1.591633 -1.589489 -1.587517 -1.572373 -1.508907 -1.522808 -1.148821 -0.408560 -0.525795 -0.937111 -0.978017 -1.812024 -1.821964 -1.625413 -1.619488 -1.632570 -1.641501 -1.639309 -1.638109 -1.638715 -1.636572 -1.627914 -1.591630 -1.507285 -1.339140 -1.198493 -0.764615 -0.908341 -0.959449 -0.955811 -1.973369 -2.206496 -2.053051 -2.020601 -2.042007 -2.075567 -2.074788 -2.073519 -2.073525 -2.073903 -2.066599 -2.033979 -1.893650 -1.736040 -1.389251 -1.104848 -0.935520 -1.050836 -1.032155 -2.200585 -2.352301 -2.228608 -2.106217 -2.084927 -2.114757 -2.113610 -2.113021 -2.112978 -2.112461 -2.111893 -2.066918 -1.959880 -1.854148 -1.566421 -1.082388 -0.818606 -1.103848 -0.968986 -2.121776 -2.319100 -2.133038 

===================================1st
-0.069681 -0.082338 -0.421423 -0.547509 -0.649031 -0.553279 -0.543912 -0.537935 -0.534911 -0.534995 -0.535330 -0.535809 -0.527194 -0.525432 -0.553659 -0.643448 -0.405778 -0.246554 -0.640166 0.361119 -0.492868 -1.009941 -1.031314 -1.232984 -1.160337 -1.131157 -1.132967 -1.133111 -1.133036 -1.132751 -1.132560 -1.135003 -1.152035 -1.171448 -1.294605 -1.122333 -0.844706 -1.459343 0.233727 -0.341395 -0.764580 -0.754586 -0.856066 -0.785497 -0.762152 -0.752358 -0.753705 -0.753279 -0.753852 -0.754229 -0.753056 -0.763154 -0.807125 -0.986594 -0.773679 -0.835411 -1.615859 0.215584 -0.227210 -0.567507 -0.543884 -0.636183 -0.547598 -0.516889 -0.509413 -0.509740 -0.509916 -0.510045 -0.508056 -0.502161 -0.494909 -0.599475 -0.917552 -0.692757 -0.684628 -1.481716 0.341734 -0.199118 -0.521221 -0.525544 -0.602938 -0.490169 -0.448167 -0.434190 -0.433272 -0.433952 -0.434459 -0.433694 -0.435597 -0.422730 -0.463627 -0.732098 -0.569769 -0.689846 -1.552940 0.337768 -0.163680 -0.452433 -0.457306 -0.549330 

===================================2nd
-1.620219 -2.312975 -3.066312 -2.674592 -2.516657 -2.521010 -2.530037 -2.516634 -2.504729 -2.503586 -2.504061 -2.502070 -2.493375 -2.496413 -2.552444 -2.549918 -2.191289 -1.807203 -1.088787 -1.526614 -2.344521 -3.166568 -2.803772 -2.528708 -2.572806 -2.617498 -2.601722 -2.594048 -2.594555 -2.594316 -2.592059 -2.589153 -2.619009 -2.774139 -2.596571 -2.368213 -1.324597 -0.491755 -1.653077 -2.431766 -2.971218 -2.446274 -2.266604 -2.251967 -2.291142 -2.310985 -2.302373 -2.296748 -2.298013 -2.294816 -2.310954 -2.268077 -2.362342 -2.113314 -2.098766 -1.286367 -0.683832 -1.759716 -2.435706 -3.100359 -2.385721 -2.210100 -2.266336 -2.339494 -2.391481 -2.381916 -2.375989 -2.376933 -2.385721 -2.409733 -2.408526 -2.383319 -2.099060 -2.178915 -1.377515 -0.534035 -1.766190 -2.528760 -3.316309 -2.426466 -2.122288 -2.251964 -2.368038 -2.431300 -2.420679 -2.417083 -2.419548 -2.424036 -2.433708 -2.450183 -2.409548 -2.206465 -2.251747 -1.344022 -0.431725 -1.755483 -2.490043 -3.276145 -2.367625 -2.166530 

===================================1st
0.671232 0.260660 -0.102131 -0.290864 -0.069233 -0.170390 -0.183015 -0.198857 -0.200472 -0.196720 -0.195986 -0.197839 -0.202649 -0.238830 -0.202663 -0.250603 -0.235084 0.078007 -0.661530 0.982234 0.806605 0.285722 0.600130 0.909098 0.952212 0.956192 0.919470 0.910844 0.909947 0.910432 0.908621 0.899610 0.867828 0.872725 0.965865 0.842773 1.279535 0.578595 0.436063 0.120728 0.391615 0.743006 1.082300 1.151407 1.153419 1.113345 1.102792 1.101928 1.102636 1.103105 1.098372 1.062005 0.983954 1.095903 1.103128 1.524999 0.958291 0.223211 0.134608 0.474588 0.973534 1.298200 1.300683 1.325895 1.272792 1.264544 1.266526 1.265469 1.257896 1.227214 1.186754 1.120984 1.108929 1.162923 1.407226 1.057682 0.530748 0.462721 0.834984 1.234650 1.696296 1.690068 1.701102 1.635526 1.619606 1.620791 1.621076 1.616500 1.588523 1.516865 1.430077 1.327548 1.238175 1.621858 1.169601 0.601230 0.526780 0.896986 1.246374 1.706793 

===================================2nd
-1.148906 -0.508867 -0.903412 -0.799040 -0.972600 -0.999017 -1.015383 -1.016230 -1.008135 -1.007976 -1.011078 -1.022786 -1.022916 -0.998132 -0.925089 -0.592852 -0.942717 -1.000463 -0.260284 -0.195864 0.824718 0.185224 -0.010882 -0.041345 -0.074025 -0.150778 -0.166146 -0.169399 -0.168024 -0.168901 -0.184193 -0.162612 -0.026199 0.157556 0.517542 -0.029334 -0.039699 0.577525 -0.160531 0.481270 0.078683 0.388104 0.241165 0.235171 0.172070 0.147930 0.142911 0.140941 0.137925 0.120752 0.131644 0.228371 0.506358 0.728768 0.260981 -0.006845 0.581522 -0.581341 -0.070278 -0.535725 -0.364777 -0.617105 -0.667233 -0.720953 -0.740917 -0.728628 -0.725426 -0.732967 -0.753522 -0.752637 -0.656861 -0.449912 -0.302445 -0.598319 -0.490336 0.281225 -0.484076 0.176000 -0.522325 -0.409008 -0.648839 -0.700296 -0.731908 -0.720083 -0.703632 -0.703577 -0.708895 -0.737940 -0.762391 -0.679541 -0.336521 -0.167174 -0.532665 -0.466845 0.238085 -0.387208 0.156172 -0.386227 -0.277257 -0.579008 
*/

