/*
    This file is part of ElephantArt.
    Copyright (C) 2021 Hung-Zhe Lin

    ElephantArt is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    ElephantArt is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with ElephantArt.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifdef USE_CUDA
#include "neural/cuda/cuda_forward_pipe.h"
#include "neural/cuda/cuda_common.h"

void CudaForwardPipe::Initialize(std::shared_ptr<DNNWeights> weights) {

}

OutputResult CudaForwardPipe::Forward(const InputData &inpnt) {
    return OutputResult{};
}

bool CudaForwardPipe::Valid() {
    return true;
}

void CudaForwardPipe::Load(std::shared_ptr<DNNWeights> weights) {

}

void CudaForwardPipe::Release() {

}

void CudaForwardPipe::Destroy() {

}

void CudaForwardPipe::NNGraph::BuildGraph(const int gpu,
                                          const int max_batch_size,
                                          const int board_size,
                                          std::shared_ptr<DNNWeights> weights) {
    if (graph_ != nullptr) {
        return;
    }
    graph_ = std::make_unique<Graph>();
    gpu_ = gpu;
    auto d = CUDA::get_device(gpu_);

    cudaSetDevice(d);

    weights_ = weights;
    handel_.apply(gpu_);

    board_size_ = board_size;
    scratch_size_ = 0;

    max_batch_ = max_batch_size;

    const auto output_channels = weights_->residual_channels;

    // Build the graph first.

    // input layer
    graph_->input_conv = CUDA::Convolution(
        max_batch_,          // max batch size
        board_size_,         // board size
        3,                   // kernel size
        kInputChannels,      // input channels
        output_channels      // output channels
    );
    graph_->input_bnorm = CUDA::Batchnorm(
        max_batch_,          // max batch size
        board_size_,         // board size
        output_channels,     // channels
        false                // relu
    );

    // residual tower
    const auto residuals = weights_->residual_blocks;
    for (int i = 0; i < residuals; ++i) {
        graph_->tower_conv.emplace_back(CUDA::Convolution{});
        graph_->tower_bnorm.emplace_back(CUDA::Batchnorm{});
        graph_->tower_conv.emplace_back(CUDA::Convolution{});
        graph_->tower_bnorm.emplace_back(CUDA::Batchnorm{});
        graph_->tower_se.emplace_back(CUDA::SEUnit{});
    }

    for (int i = 0; i < residuals; ++i) {
        const auto t_offset = 2 * i;
        const auto tower_channels = weights_->residual_channels;
        const auto tower_ptr = weights_->tower.data() + i;
    
        graph_->tower_conv[t_offset+0] = CUDA::Convolution(
            max_batch_,          // max batch size
            board_size_,         // board size
            3,                   // kernel size
            tower_channels,      // input channels
            tower_channels       // output channels
        );
        graph_->tower_bnorm[t_offset+0] = CUDA::Batchnorm(
            max_batch_,          // max batch size
            board_size_,         // board size
            tower_channels       // channels
        );

        graph_->tower_conv[t_offset+1] = CUDA::Convolution(
            max_batch_,          // max batch size
            board_size_,         // board size
            3,                   // kernel size
            tower_channels,      // input channels
            tower_channels       // output channels
        );
        graph_->tower_bnorm[t_offset+1] = CUDA::Batchnorm(
            max_batch_,          // max batch size
            board_size_,         // board size
            tower_channels,      // channels
            !tower_ptr->apply_se // relu
        );

        if (tower_ptr->apply_se) {
            const size_t se_size = tower_ptr->se_size;
            graph_->tower_se[i] = CUDA::SEUnit(
                max_batch_,      // max batch size
                board_size_,     // board size
                tower_channels,  // channels
                se_size          // SE size
            );
        }
    }

    // policy head
    const auto policy_extract_channels = weights_->policy_extract_channels;
    graph_->p_ex_conv = CUDA::Convolution(
        max_batch_,             // max batch size
        board_size_,            // board size
        1,                      // kernel size
        output_channels,        // input channels
        policy_extract_channels // output channels
    );
    graph_->p_ex_bnorm = CUDA::Batchnorm(
        max_batch_,             // max batch size
        board_size_,            // board size
        policy_extract_channels // channels
    );
    graph_->p_prob = CUDA::Convolution(
        max_batch_,             // max batch size
        board_size_,            // board size
        1,                      // kernel size
        policy_extract_channels,// input channels
        1                       // output channels
    );
    graph_->p_pool = CUDA::GlobalAvgPool(
        max_batch_,              // max batch size
        board_size_,             // board size
        policy_extract_channels  // input channels
    );
    graph_->p_prob_pass = CUDA::FullyConnect(
        max_batch_,              // max batch size
        policy_extract_channels, // input sizes
        1,                       // outpur size
        false
    );

    // value head
    const auto value_extract_channels = weights_->value_extract_channels;
    graph_->v_ex_conv = CUDA::Convolution(
        max_batch_,              // max batch size
        board_size_,             // board size
        1,                       // kernel size
        output_channels,         // input channels
        value_extract_channels   // output channels
    );
    graph_->v_ex_bnorm = CUDA::Batchnorm(
        max_batch_,              // max batch size
        board_size_,             // board size
        value_extract_channels   // channels
    );
    graph_->v_ownership  = CUDA::Convolution(
        max_batch_,              // max batch size
        board_size_,             // board size
        1,                       // kernel size
        value_extract_channels,  // input channels
        1                        // output channels
    ); 
    graph_->v_pool = CUDA::GlobalAvgPool(
        max_batch_,              // max batch size
        board_size_,             // board size
        value_extract_channels   // input channels
    );
    graph_->v_misc = CUDA::FullyConnect(
        max_batch_,              // max batch size
        value_extract_channels,  // input size
        kOuputValueMisc,         // output size
        false                    // relu
    );
    // Now fill the parameters.

    // input layer
    graph_->input_conv.LoadingWeight(
        weights_->input_conv.GetWeights(), scratch_size_, &handel_);

    graph_->input_bnorm.LoadingWeight(
        weights_->input_bn.GetMeans(), weights_->input_bn.GetStddevs());

    // residual tower
    for (int i = 0; i < residuals; ++i) {
        const auto t_offset = 2 * i;
        const auto tower_ptr = weights_->tower.data() + i;

        graph_->tower_conv[t_offset+0].LoadingWeight(
            tower_ptr->conv1.GetWeights(), scratch_size_, &handel_);

        graph_->tower_bnorm[t_offset+0].LoadingWeight(
            tower_ptr->bn1.GetMeans(), tower_ptr->bn1.GetStddevs());

        graph_->tower_conv[t_offset+1].LoadingWeight(
            tower_ptr->conv2.GetWeights(), scratch_size_, &handel_);

        graph_->tower_bnorm[t_offset+1].LoadingWeight(
            tower_ptr->bn2.GetMeans(), tower_ptr->bn2.GetStddevs());

        if (tower_ptr->apply_se) {
            graph_->tower_se[i].LoadingWeight(
                tower_ptr->extend.GetWeights(),
                tower_ptr->extend.GetBiases(),
                tower_ptr->squeeze.GetWeights(),
                tower_ptr->squeeze.GetBiases());
        }
    }

    // policy head
    graph_->p_ex_conv.LoadingWeight(
        weights->p_ex_conv.GetWeights(), scratch_size_, &handel_);

    graph_->p_ex_bnorm.LoadingWeight(
        weights->p_ex_bn.GetMeans(), weights_->p_ex_bn.GetStddevs());

    graph_->p_prob.LoadingWeight(
        weights->prob_conv.GetWeights(),
        weights_->prob_conv.GetBiases(),
        scratch_size_, &handel_);

    graph_->p_prob_pass.LoadingWeight(
        weights_->pass_fc.GetWeights(), weights_->pass_fc.GetBiases());

    // value head
    graph_->v_ex_conv.LoadingWeight(
        weights->v_ex_conv.GetWeights(), scratch_size_, &handel_);

    graph_->v_ex_bnorm.LoadingWeight(
        weights->v_ex_bn.GetMeans(), weights_->v_ex_bn.GetStddevs());

    graph_->v_ownership.LoadingWeight(
        weights->v_ownership.GetWeights(),
        weights_->v_ownership.GetBiases(),
        scratch_size_, &handel_);

    graph_->v_misc.LoadingWeight(
        weights_->v_misc.GetWeights(), weights_->v_misc.GetBiases());


    // Allocate some buffers.
    const size_t factor = max_batch_ * sizeof(float);
    const size_t num_intersections = board_size_ * board_size_;

    const size_t planes_size = factor * kInputChannels * num_intersections;
    const size_t spatia_size = factor * num_intersections;
    const size_t val_size = factor * kOuputValueMisc;

    const size_t conv_op_size = factor * weights_->residual_channels * num_intersections;

    const size_t pol_op1_size = factor * policy_extract_channels * num_intersections;
    const size_t pol_op2_size = factor * policy_extract_channels;

    const size_t val_op1_size = factor * value_extract_channels * num_intersections;
    const size_t val_op2_size = factor * value_extract_channels;
    const size_t val_op3_size = factor * kOuputValueMisc;

    CUDA::ReportCUDAErrors(cudaMalloc(&cuda_scratch_, scratch_size_));
    CUDA::ReportCUDAErrors(cudaMalloc(&cuda_input_planes_, planes_size));

    CUDA::ReportCUDAErrors(cudaMalloc(&cuda_conv_op_[0], conv_op_size));
    CUDA::ReportCUDAErrors(cudaMalloc(&cuda_conv_op_[1], conv_op_size));
    CUDA::ReportCUDAErrors(cudaMalloc(&cuda_conv_op_[2], conv_op_size));

    CUDA::ReportCUDAErrors(cudaMalloc(&cuda_pol_op_[0], pol_op1_size));
    CUDA::ReportCUDAErrors(cudaMalloc(&cuda_pol_op_[1], pol_op2_size));

    CUDA::ReportCUDAErrors(cudaMalloc(&cuda_val_op_[0], val_op1_size));
    CUDA::ReportCUDAErrors(cudaMalloc(&cuda_val_op_[1], val_op2_size));
    CUDA::ReportCUDAErrors(cudaMalloc(&cuda_val_op_[2], val_op3_size));

    CUDA::ReportCUDAErrors(cudaMalloc(&cuda_output_prob_pass_, factor));
    CUDA::ReportCUDAErrors(cudaMalloc(&cuda_output_prob_, spatia_size));
    CUDA::ReportCUDAErrors(cudaMalloc(&cuda_output_ownership_, spatia_size));
    CUDA::ReportCUDAErrors(cudaMalloc(&cuda_output_val_, val_size));
}


#endif
