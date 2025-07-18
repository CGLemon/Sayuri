#ifdef USE_CUDA

#include <sstream>
#include <stdexcept>

#include "config.h"
#include "neural/cuda/cuda_forward_pipe.h"
#include "neural/cuda/cuda_kernels.h"
#include "neural/encoder.h"
#include "utils/log.h"
#include "utils/format.h"
#include "utils/option.h"

void CudaForwardPipe::Initialize(std::shared_ptr<DNNWeights> weights) {
    LOGGING << cuda::GetBackendInfo();

    dump_gpu_info_ = true;

    group_ = std::make_unique<ThreadGroup<void>>(&ThreadPool::Get());

    auto option = ForwardPipeOption::Get().
                      SetBoardSize(GetOption<int>("defualt_boardsize")).
                      SetBatchSize(GetOption<int>("batch_size"));
    Construct(option, weights);

    AssignWorkers(); // Run the batch forwarding worker.
}

OutputResult CudaForwardPipe::Forward(const InputData &input) {
    OutputResult output;
    InputData reordered_input = input;

    // Reorder the inputs data.
    const int planes_bsize = input.board_size;
    const bool should_reorder = planes_bsize != board_size_;

    if (should_reorder) {
        // The input data's board size doesn't match the NN's expected
        // input size. We are reordering the original input data to conform
        // to the NN's input dimensions.
        for (int c = 0; c < weights_->input_channels; ++c) {
            int offset_r = c * board_size_ * board_size_; // data's ordering index
            int offset_p = c * planes_bsize * planes_bsize; // NN's ordering index

            for (int idx = 0; idx < board_size_ * board_size_; ++idx) {
                const int x = idx % board_size_;
                const int y = idx / board_size_;
                if (x < planes_bsize && y < planes_bsize) {
                    reordered_input.planes[offset_r++] = input.planes[offset_p++];
                } else {
                    reordered_input.planes[offset_r++] = 0.f;
                }
            }
        }
    }

    auto entry = std::make_shared<ForwawrdEntry>(reordered_input, output);
    std::unique_lock<std::mutex> lock(entry->mutex);
    {
        // Push the entry into queue.
        std::lock_guard<std::mutex> queue_lock(queue_mutex_);
        entry_queue_.emplace_back(entry);
    }

    if (static_cast<int>(entry_queue_.size()) >= forwarding_batch_per_nn_) {
        cv_.notify_one(); // Wake up one worker if there are enough batch size.
    }
    entry->cv.wait(lock); // Wait for batch forwarding worker.

    // Reorder the outputs data.
    OutputResult reordered_ouput = output;

    if (should_reorder) {
        // Reorder the NN's outputs data to fit the correct data format.
        int offset_r = 0; // data order index
        int offset_p = 0; // NN order index
        for (int idx = 0; idx < board_size_ * board_size_; ++idx) {
            const int x = idx % board_size_;
            const int y = idx / board_size_;
            if (x < planes_bsize && y < planes_bsize) {
                reordered_ouput.probabilities[offset_r] = output.probabilities[offset_p];
                reordered_ouput.ownership[offset_r] = output.ownership[offset_p];
                offset_r++;
                offset_p++;
            } else {
                offset_p++;
            }
        }
    }

    return reordered_ouput;
}

bool CudaForwardPipe::Valid() const {
    return weights_ != nullptr;
}

void CudaForwardPipe::Construct(ForwardPipeOption option,
                                std::shared_ptr<DNNWeights> weights) {
    // Construct the network with parameters (e.g., board_size) and weights.
    // If the current parameters are the same as the new ones, exit the function 
    // immediately.
    if (weights) {
        weights_ = weights;
    }
    if (weights_ == nullptr) {
        // use dummy backend
        return;
    }

    int board_size = option.IsValidBoardSize() ?
                         option.board_size : board_size_;
    int batch_size = option.IsValidBatchSize() ?
                         option.batch_size : max_batch_per_nn_;
    // Select the matched board size.
    board_size = std::max(board_size, GetOption<int>("fixed_nn_boardsize"));

    if (board_size == 0 || batch_size == 0) {
        LOGGING << "NN board size/batch size should be larger than zero.\n";
        return;
    }

    forwarding_batch_per_nn_ = batch_size;
    if (board_size_ == board_size &&
            batch_size <= max_batch_per_nn_) {
        return;
    }
    Release();

    board_size_ = board_size;
    max_batch_per_nn_ = batch_size;

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
        nngraphs_.emplace_back(std::make_unique<NNGraph>(io_mutex_));
    }

    max_batch_per_nn_ = batch_size;

    // Construct network graph for each valid GPU.
    for (auto i = size_t{0}; i < gpus_list.size(); ++i) {
        nngraphs_[i]->ConstructGraph(
            dump_gpu_info_, gpus_list[i], max_batch_per_nn_, board_size_, weights_);
    }

    dump_gpu_info_ = false; // don't show the GPU info next time.
}

void CudaForwardPipe::Release() {
    for (auto &g : nngraphs_) {
        g->DestroyGraph();
    }
    nngraphs_.clear();
}

void CudaForwardPipe::Destroy() {
    Release();
    QuitWorkers();
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
    graph_->input_conv = cuda::Convolution(
        &handles_,
        max_batch_,          // max batch size
        board_size_,         // board size
        3,                   // kernel size
        input_channels,      // input channels
        output_channels,     // output channels
        default_act          // activation
    );

    // block tower
    const auto blocks = weights_->residual_blocks;
    for (int i = 0; i < weights_->residual_blocks; ++i) {
        graph_->tower.emplace_back(NNGraph::Block{});
    }

    int peak_channels = 0;
    for (int i = 0; i < blocks; ++i) {
        const auto tower_ptr = weights_->tower[i].get();
        if (tower_ptr->IsResidualBlock()) {
            const auto channels = weights_->residual_channels;
            const auto last_act = tower_ptr->apply_se ? Activation::kIdentity : default_act;

            graph_->tower[i].conv1 = cuda::Convolution(
                &handles_,
                max_batch_,   // max batch size
                board_size_,  // board size
                3,            // kernel size
                channels,     // input channels
                channels,     // output channels
                default_act   // activation
            );
            graph_->tower[i].conv2 = cuda::Convolution(
                &handles_,
                max_batch_,   // max batch size
                board_size_,  // board size
                3,            // kernel size
                channels,     // input channels
                channels,     // output channels
                last_act      // activation
            );
            peak_channels = std::max(peak_channels, channels);
        } else if (tower_ptr->IsBottleneckBlock()) {
            const auto outer_channels = weights_->residual_channels;
            const auto inner_channels = tower_ptr->bottleneck_channels;
            const auto last_act = tower_ptr->apply_se ? Activation::kIdentity : default_act;

            graph_->tower[i].pre_btl_conv = cuda::Convolution(
                &handles_,
                max_batch_,     // max batch size
                board_size_,    // board size
                1,              // kernel size
                outer_channels, // input channels
                inner_channels, // output channels
                default_act     // activation
            );
            graph_->tower[i].conv1 = cuda::Convolution(
                &handles_,
                max_batch_,     // max batch size
                board_size_,    // board size
                3,              // kernel size
                inner_channels, // input channels
                inner_channels, // output channels
                default_act     // activation
            );
            graph_->tower[i].conv2 = cuda::Convolution(
                &handles_,
                max_batch_,     // max batch size
                board_size_,    // board size
                3,              // kernel size
                inner_channels, // input channels
                inner_channels, // output channels
                default_act     // activation
            );
            graph_->tower[i].post_btl_conv = cuda::Convolution(
                &handles_,
                max_batch_,     // max batch size
                board_size_,    // board size
                1,              // kernel size
                inner_channels, // input channels
                outer_channels, // output channels
                last_act        // activation
            );
            peak_channels = std::max({peak_channels, inner_channels, outer_channels});
        } else if (tower_ptr->IsNestedBottleneckBlock()) {
            const auto outer_channels = weights_->residual_channels;
            const auto inner_channels = tower_ptr->bottleneck_channels;
            const auto last_act = tower_ptr->apply_se ? Activation::kIdentity : default_act;

            graph_->tower[i].pre_btl_conv = cuda::Convolution(
                &handles_,
                max_batch_,     // max batch size
                board_size_,    // board size
                1,              // kernel size
                outer_channels, // input channels
                inner_channels, // output channels
                default_act     // activation
            );
            graph_->tower[i].conv1 = cuda::Convolution(
                &handles_,
                max_batch_,     // max batch size
                board_size_,    // board size
                3,              // kernel size
                inner_channels, // input channels
                inner_channels, // output channels
                default_act     // activation
            );
            graph_->tower[i].conv2 = cuda::Convolution(
                &handles_,
                max_batch_,     // max batch size
                board_size_,    // board size
                3,              // kernel size
                inner_channels, // input channels
                inner_channels, // output channels
                default_act     // activation
            );
            graph_->tower[i].conv3 = cuda::Convolution(
                &handles_,
                max_batch_,     // max batch size
                board_size_,    // board size
                3,              // kernel size
                inner_channels, // input channels
                inner_channels, // output channels
                default_act     // activation
            );
            graph_->tower[i].conv4 = cuda::Convolution(
                &handles_,
                max_batch_,     // max batch size
                board_size_,    // board size
                3,              // kernel size
                inner_channels, // input channels
                inner_channels, // output channels
                default_act     // activation
            );
            graph_->tower[i].post_btl_conv = cuda::Convolution(
                &handles_,
                max_batch_,     // max batch size
                board_size_,    // board size
                1,              // kernel size
                inner_channels, // input channels
                outer_channels, // output channels
                last_act        // activation
            );
            peak_channels = std::max({peak_channels, inner_channels, outer_channels});
        } else if (tower_ptr->IsMixerBlock()) {
            const auto channels = weights_->residual_channels;
            const auto feedforwards = tower_ptr->feedforward_channels;
            const auto filters = tower_ptr->dw_conv.GetFilter();
            const auto last_act = tower_ptr->apply_se ? Activation::kIdentity : default_act;

            graph_->tower[i].dw_conv = cuda::DepthwiseConvolution(
                &handles_,
                max_batch_,   // max batch size
                board_size_,  // board size
                filters,      // kernel size
                channels,     // input channels
                default_act   // activation
            );
            graph_->tower[i].conv1 = cuda::Convolution(
                &handles_,
                max_batch_,   // max batch size
                board_size_,  // board size
                1,            // kernel size
                channels,     // input channels
                feedforwards, // output channels
                default_act   // activation
            );
            graph_->tower[i].conv2 = cuda::Convolution(
                &handles_,
                max_batch_,   // max batch size
                board_size_,  // board size
                1,            // kernel size
                feedforwards, // input channels
                channels,     // output channels
                last_act      // activation
            );
            peak_channels = std::max({peak_channels, channels, feedforwards});
        }
        if (tower_ptr->apply_se) {
            const auto channels = weights_->residual_channels;
            const size_t se_size = tower_ptr->se_size;

            graph_->tower[i].se_module = cuda::SEUnit(
                &handles_,
                max_batch_,  // max batch size
                board_size_, // board size
                channels,    // channels
                se_size,     // SE size
                default_act  // activation
            );
        }
    }

    // policy head
    const auto policy_head_channels = weights_->policy_head_channels;
    const auto probabilities_channels = weights_->probabilities_channels;
    const auto pass_probability_outputs = weights_->pass_probability_outputs;
    graph_->p_hd_conv = cuda::Convolution(
        &handles_,
        max_batch_,              // max batch size
        board_size_,             // board size
        1,                       // kernel size
        output_channels,         // input channels
        policy_head_channels,    // output channels
        default_act              // activation
    );
    if (weights_->policy_head_type == PolicyHeadType::kRepLK) {
        const auto filters = weights_->p_dw_conv.GetFilter();
        graph_->p_dw_conv = cuda::DepthwiseConvolution(
            &handles_,
            max_batch_,              // max batch size
            board_size_,             // board size
            filters,                 // kernel size
            policy_head_channels,    // input channels
            default_act              // activation
        );
        graph_->p_pt_conv = cuda::Convolution(
            &handles_,
            max_batch_,              // max batch size
            board_size_,             // board size
            1,                       // kernel size
            policy_head_channels,    // input channels
            policy_head_channels,    // output channels
            default_act              // activation
        );
    }
    graph_->p_pool = cuda::GlobalPooling(
        &handles_,
        false,
        max_batch_,               // max batch size
        board_size_,              // board size
        policy_head_channels      // input channels
    );
    graph_->p_inter = cuda::FullyConnect(
        &handles_,
        max_batch_,               // max batch size
        3*policy_head_channels,   // input sizes
        policy_head_channels,     // outpur size
        default_act               // activation
    );
    graph_->p_prob = cuda::Convolution(
        &handles_,
        max_batch_,               // max batch size
        board_size_,              // board size
        1,                        // kernel size
        policy_head_channels,     // input channels
        probabilities_channels,   // output channels
        Activation::kIdentity     // activation
    );
    graph_->p_prob_pass = cuda::FullyConnect(
        &handles_,
        max_batch_,               // max batch size
        policy_head_channels,     // input sizes
        pass_probability_outputs, // outpur size
        Activation::kIdentity     // activation
    );

    // value head
    const auto value_head_channels = weights_->value_head_channels;
    const auto ownership_channels = weights_->ownership_channels;
    const auto value_misc_outputs = weights_->value_misc_outputs;
    graph_->v_hd_conv = cuda::Convolution(
        &handles_,
        max_batch_,               // max batch size
        board_size_,              // board size
        1,                        // kernel size
        output_channels,          // input channels
        value_head_channels,      // output channels
        default_act               // activation
    );
    graph_->v_pool = cuda::GlobalPooling(
        &handles_,
        true,
        max_batch_,               // max batch size
        board_size_,              // board size
        value_head_channels       // input channels
    );
    graph_->v_inter = cuda::FullyConnect(
        &handles_,
        max_batch_,               // max batch size
        3*value_head_channels,    // input sizes
        3*value_head_channels,    // outpur size
        default_act               // activation
    );
    graph_->v_ownership = cuda::Convolution(
        &handles_,
        max_batch_,               // max batch size
        board_size_,              // board size
        1,                        // kernel size
        value_head_channels,      // input channels
        ownership_channels,       // output channels
        Activation::kIdentity     // activation
    );
    graph_->v_misc = cuda::FullyConnect(
        &handles_,
        max_batch_,               // max batch size
        3*value_head_channels,    // input size
        value_misc_outputs,       // output size
        Activation::kIdentity     // relu
    );

    // Now push the weights.

    const bool winograd = weights_->winograd;

    // input layer
    graph_->input_conv.LoadWeights(
        weights_->input_conv.GetWeights(),
        weights_->input_conv.GetBiases(),
        scratch_size_, winograd);

    // block tower
    for (int i = 0; i < blocks; ++i) {
        const auto tower_ptr = weights_->tower[i].get();
        if (tower_ptr->IsResidualBlock()) {
            graph_->tower[i].conv1.LoadWeights(
                tower_ptr->conv1.GetWeights(),
                tower_ptr->conv1.GetBiases(),
                scratch_size_, winograd);
            graph_->tower[i].conv2.LoadWeights(
                tower_ptr->conv2.GetWeights(),
                tower_ptr->conv2.GetBiases(),
                scratch_size_, winograd);
        } else if (tower_ptr->IsBottleneckBlock()) {
            graph_->tower[i].pre_btl_conv.LoadWeights(
                tower_ptr->pre_btl_conv.GetWeights(),
                tower_ptr->pre_btl_conv.GetBiases(),
                scratch_size_, winograd);
            graph_->tower[i].conv1.LoadWeights(
                tower_ptr->conv1.GetWeights(),
                tower_ptr->conv1.GetBiases(),
                scratch_size_, winograd);
            graph_->tower[i].conv2.LoadWeights(
                tower_ptr->conv2.GetWeights(),
                tower_ptr->conv2.GetBiases(),
                scratch_size_, winograd);
            graph_->tower[i].post_btl_conv.LoadWeights(
                tower_ptr->post_btl_conv.GetWeights(),
                tower_ptr->post_btl_conv.GetBiases(),
                scratch_size_, winograd);
        } else if (tower_ptr->IsNestedBottleneckBlock()) {
            graph_->tower[i].pre_btl_conv.LoadWeights(
                tower_ptr->pre_btl_conv.GetWeights(),
                tower_ptr->pre_btl_conv.GetBiases(),
                scratch_size_, winograd);
            graph_->tower[i].conv1.LoadWeights(
                tower_ptr->conv1.GetWeights(),
                tower_ptr->conv1.GetBiases(),
                scratch_size_, winograd);
            graph_->tower[i].conv2.LoadWeights(
                tower_ptr->conv2.GetWeights(),
                tower_ptr->conv2.GetBiases(),
                scratch_size_, winograd);
            graph_->tower[i].conv3.LoadWeights(
                tower_ptr->conv3.GetWeights(),
                tower_ptr->conv3.GetBiases(),
                scratch_size_, winograd);
            graph_->tower[i].conv4.LoadWeights(
                tower_ptr->conv4.GetWeights(),
                tower_ptr->conv4.GetBiases(),
                scratch_size_, winograd);
            graph_->tower[i].post_btl_conv.LoadWeights(
                tower_ptr->post_btl_conv.GetWeights(),
                tower_ptr->post_btl_conv.GetBiases(),
                scratch_size_, winograd);
        } else if (tower_ptr->IsMixerBlock()) {
            graph_->tower[i].dw_conv.LoadWeights(
                tower_ptr->dw_conv.GetWeights(),
                tower_ptr->dw_conv.GetBiases());
            graph_->tower[i].conv1.LoadWeights(
                tower_ptr->conv1.GetWeights(),
                tower_ptr->conv1.GetBiases(),
                scratch_size_, winograd);
            graph_->tower[i].conv2.LoadWeights(
                tower_ptr->conv2.GetWeights(),
                tower_ptr->conv2.GetBiases(),
                scratch_size_, winograd);
        }
        if (tower_ptr->apply_se) {
            graph_->tower[i].se_module.LoadWeights(
                tower_ptr->squeeze.GetWeights(),
                tower_ptr->squeeze.GetBiases(),
                tower_ptr->excite.GetWeights(),
                tower_ptr->excite.GetBiases());
        }
    }

    // policy head
    graph_->p_hd_conv.LoadWeights(
        weights->p_hd_conv.GetWeights(),
        weights->p_hd_conv.GetBiases(),
        scratch_size_, winograd);

    if (weights_->policy_head_type == PolicyHeadType::kRepLK) {
        graph_->p_dw_conv.LoadWeights(
            weights->p_dw_conv.GetWeights(),
            weights->p_dw_conv.GetBiases());
        graph_->p_pt_conv.LoadWeights(
            weights->p_pt_conv.GetWeights(),
            weights->p_pt_conv.GetBiases(),
            scratch_size_, winograd);
    }

    graph_->p_inter.LoadWeights(
        weights_->p_inter_fc.GetWeights(), weights_->p_inter_fc.GetBiases());

    graph_->p_prob.LoadWeights(
        weights->prob_conv.GetWeights(),
        weights_->prob_conv.GetBiases(),
        scratch_size_, winograd);

    graph_->p_prob_pass.LoadWeights(
        weights_->pass_fc.GetWeights(), weights_->pass_fc.GetBiases());

    // value head
    graph_->v_hd_conv.LoadWeights(
        weights->v_hd_conv.GetWeights(),
        weights->v_hd_conv.GetBiases(),
        scratch_size_, winograd);

    graph_->v_inter.LoadWeights(
        weights_->v_inter_fc.GetWeights(), weights_->v_inter_fc.GetBiases());

    graph_->v_ownership.LoadWeights(
        weights->v_ownership.GetWeights(),
        weights_->v_ownership.GetBiases(),
        scratch_size_, winograd);

    graph_->v_misc.LoadWeights(
        weights_->v_misc.GetWeights(), weights_->v_misc.GetBiases());

    // Allocate all buffers.
    const size_t factor = max_batch_ * cuda::GetCudaTypeSize(handles_.fp16);
    const size_t num_intersections = board_size_ * board_size_;

    const size_t planes_size = factor * input_channels * num_intersections;
    const size_t spatia_size = factor * num_intersections;
    const size_t pol_size = spatia_size * probabilities_channels;
    const size_t pass_size = factor * pass_probability_outputs;
    const size_t val_size = factor * value_misc_outputs;
    const size_t ownership_size = spatia_size * ownership_channels;

    const size_t conv_op_size = factor * peak_channels * num_intersections;

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

void CudaForwardPipe::NNGraph::SetComputationMode(cuda::CudaHandles *handles) {
    cudaDeviceProp dev_prop = cuda::GetDeviceProp();

    if (dev_prop.major <= 6 ||
            !GetOption<bool>("fp16")) {
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
        cuda::ReportCUBLASErrors(cublasSetMathMode(
            handles->cublas_handle,
            CUBLAS_TENSOR_OP_MATH));
    }
}

bool CudaForwardPipe::NNGraph::ApplyMask(const std::vector<InputData> &batch_input) {
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

        // Copy the mask from host memory to device.
        {
            std::lock_guard<std::mutex> lock(io_mutex_);
            cuda::SetDevice(handles_.gpu_id);

            cuda::CopyToCudaOp(
                handles_.fp16, &(cuda_mask_op_[0]),
                spat_mask, &(host_mask_op_[0]));
            cuda::CopyToCudaOp(
                handles_.fp16, &(cuda_mask_op_[1]),
                sqrt_mask, &(host_mask_op_[1]));
        }
    }

    return should_apply_mask;
}

std::vector<OutputResult> CudaForwardPipe::NNGraph::BatchForward(const std::vector<InputData> &batch_input) {
    const auto batch_size = (int)batch_input.size();

    assert(max_batch_ >= batch_size);

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

    std::array<void *, 2> mask_buf = cuda_mask_op_;
    if (!should_apply_mask) {
        // Disable the mask.
        mask_buf[0] = mask_buf[1] = nullptr;
    }

    // copy the inputs data from host memory to device
    {
        std::lock_guard<std::mutex> lock(io_mutex_);
        cuda::SetDevice(handles_.gpu_id);
        cuda::CopyToCudaOp(
            handles_.fp16, &cuda_input_planes_,
            batch_planes, &host_input_planes_);
    }

    // input layer
    graph_->input_conv.Forward(
        batch_size,
        cuda_conv_op_[0], cuda_input_planes_,
        nullptr, mask_buf[0],
        cuda_scratch_op_[0], cuda_scratch_op_[1], scratch_size_);

    // block tower
    const auto blocks = weights_->residual_blocks;
    for (int i = 0; i < blocks; ++i) {
        // cuda_conv_op_[0] is input/residual.
        // cuda_conv_op_[1 ~ 2] are buffers.
        // cuda_conv_op_[3] is output before SE module.

        const auto tower_ptr = weights_->tower[i].get();
        if (tower_ptr->IsResidualBlock()) {
            // 1st conv layer
            graph_->tower[i].conv1.Forward(
                batch_size,
                cuda_conv_op_[1], cuda_conv_op_[0],
                nullptr, mask_buf[0],
                cuda_scratch_op_[0], cuda_scratch_op_[1], scratch_size_);

            // 2nd conv layer
            void *second_skip = tower_ptr->apply_se ?
                                    nullptr : cuda_conv_op_[0];
            graph_->tower[i].conv2.Forward(
                batch_size,
                cuda_conv_op_[3], cuda_conv_op_[1],
                second_skip, mask_buf[0],
                cuda_scratch_op_[0],  cuda_scratch_op_[1], scratch_size_);
        } else if (tower_ptr->IsBottleneckBlock()) {
            // pre-bottleneck
            graph_->tower[i].pre_btl_conv.Forward(
                batch_size,
                cuda_conv_op_[1], cuda_conv_op_[0],
                nullptr, mask_buf[0],
                cuda_scratch_op_[0], cuda_scratch_op_[1], scratch_size_);

            // 1st conv layer
            graph_->tower[i].conv1.Forward(
                batch_size,
                cuda_conv_op_[2], cuda_conv_op_[1],
                nullptr, mask_buf[0],
                cuda_scratch_op_[0], cuda_scratch_op_[1], scratch_size_);

            // 2nd conv layer
            graph_->tower[i].conv2.Forward(
                batch_size,
                cuda_conv_op_[1], cuda_conv_op_[2],
                nullptr, mask_buf[0],
                cuda_scratch_op_[0],  cuda_scratch_op_[1], scratch_size_);

            // post-bottleneck
            void *btl_skip = tower_ptr->apply_se ?
                                     nullptr : cuda_conv_op_[0];
            graph_->tower[i].post_btl_conv.Forward(
                batch_size,
                cuda_conv_op_[3], cuda_conv_op_[1],
                btl_skip, mask_buf[0],
                cuda_scratch_op_[0], cuda_scratch_op_[1], scratch_size_);
        } else if (tower_ptr->IsNestedBottleneckBlock()) {
            // pre-bottleneck
            graph_->tower[i].pre_btl_conv.Forward(
                batch_size,
                cuda_conv_op_[1], cuda_conv_op_[0],
                nullptr, mask_buf[0],
                cuda_scratch_op_[0], cuda_scratch_op_[1], scratch_size_);

            // 1st conv layer (1st block)
            graph_->tower[i].conv1.Forward(
                batch_size,
                cuda_conv_op_[2], cuda_conv_op_[1],
                nullptr, mask_buf[0],
                cuda_scratch_op_[0], cuda_scratch_op_[1], scratch_size_);

            // 2nd conv layer (1st block)
            void *block1_skip = cuda_conv_op_[1];
            graph_->tower[i].conv2.Forward(
                batch_size,
                cuda_conv_op_[3], cuda_conv_op_[2],
                block1_skip, mask_buf[0],
                cuda_scratch_op_[0],  cuda_scratch_op_[1], scratch_size_);

            std::swap(cuda_conv_op_[3], cuda_conv_op_[1]);

            // 3rd conv layer (2nd block)
            graph_->tower[i].conv3.Forward(
                batch_size,
                cuda_conv_op_[2], cuda_conv_op_[1],
                nullptr, mask_buf[0],
                cuda_scratch_op_[0], cuda_scratch_op_[1], scratch_size_);

            // 4th conv layer (2nd block)
            void *block2_skip = cuda_conv_op_[1];
            graph_->tower[i].conv4.Forward(
                batch_size,
                cuda_conv_op_[3], cuda_conv_op_[2],
                block2_skip, mask_buf[0],
                cuda_scratch_op_[0],  cuda_scratch_op_[1], scratch_size_);

            std::swap(cuda_conv_op_[3], cuda_conv_op_[1]);

            // post-bottleneck
            void *btl_skip = tower_ptr->apply_se ?
                                     nullptr : cuda_conv_op_[0];
            graph_->tower[i].post_btl_conv.Forward(
                batch_size,
                cuda_conv_op_[3], cuda_conv_op_[1],
                btl_skip, mask_buf[0],
                cuda_scratch_op_[0], cuda_scratch_op_[1], scratch_size_);
        } else if (tower_ptr->IsMixerBlock()) {
            // dw conv layer
            graph_->tower[i].dw_conv.Forward(
                batch_size,
                cuda_conv_op_[3], cuda_conv_op_[0],
                cuda_conv_op_[0], mask_buf[0]);

            std::swap(cuda_conv_op_[3], cuda_conv_op_[0]);

            // 1st ffn conv layer
            graph_->tower[i].conv1.Forward(
                batch_size,
                cuda_conv_op_[1], cuda_conv_op_[0],
                nullptr, mask_buf[0],
                cuda_scratch_op_[0], cuda_scratch_op_[1], scratch_size_);

            // 2nd ffn conv layer
            void *ffn_skip = tower_ptr->apply_se ?
                                     nullptr : cuda_conv_op_[0];
            graph_->tower[i].conv2.Forward(
                batch_size,
                cuda_conv_op_[3], cuda_conv_op_[1],
                ffn_skip, mask_buf[0],
                cuda_scratch_op_[0],  cuda_scratch_op_[1], scratch_size_);
        }

        bool module_skip = false;
        if (tower_ptr->apply_se) {
            // squeeze-and-excitation module
            void *se_skip = cuda_conv_op_[0];
            void *se_outs = cuda_conv_op_[0];

            graph_->tower[i].se_module.Forward(
                batch_size,
                se_outs, cuda_conv_op_[3],
                se_skip, mask_buf[0], mask_buf[1]);
            module_skip = true;
        }

        if (!module_skip) {
            std::swap(cuda_conv_op_[3], cuda_conv_op_[0]);
        }
    }

    // policy head
    const auto policy_head_channels = weights_->policy_head_channels;

    graph_->p_hd_conv.Forward(
        batch_size,
        cuda_pol_op_[0], cuda_conv_op_[0],
        nullptr, mask_buf[0],
        cuda_scratch_op_[0], cuda_scratch_op_[1], scratch_size_);

    if (weights_->policy_head_type == PolicyHeadType::kRepLK) {
        graph_->p_dw_conv.Forward(
            batch_size,
            cuda_pol_op_[1], cuda_pol_op_[0],
            nullptr, mask_buf[0]);
        graph_->p_pt_conv.Forward(
            batch_size,
            cuda_pol_op_[0], cuda_pol_op_[1],
            nullptr, mask_buf[0],
            cuda_scratch_op_[0], cuda_scratch_op_[1], scratch_size_);
    }

    graph_->p_pool.Forward(
        batch_size,
        cuda_pol_op_[1], cuda_pol_op_[0],
        mask_buf[0], mask_buf[1]);

    graph_->p_inter.Forward(
        batch_size, cuda_pol_op_[2], cuda_pol_op_[1]);

    void *null_op = nullptr;
    cuda::AddSpatial(
        handles_.fp16, cuda_pol_op_[0], cuda_pol_op_[2],
        null_op, mask_buf[0],
        batch_size * policy_head_channels,
        batch_size, policy_head_channels, num_intersections,
        Activation::kIdentity, handles_.stream);

    graph_->p_prob.Forward(
        batch_size,
        cuda_output_prob_, cuda_pol_op_[0],
        nullptr, nullptr,
        cuda_scratch_op_[0], cuda_scratch_op_[1], scratch_size_);

    graph_->p_prob_pass.Forward(
        batch_size, cuda_output_prob_pass_, cuda_pol_op_[2]);

    // value head
    graph_->v_hd_conv.Forward(
        batch_size,
        cuda_val_op_[0], cuda_conv_op_[0],
        nullptr, mask_buf[0],
        cuda_scratch_op_[0], cuda_scratch_op_[1], scratch_size_);

    graph_->v_pool.Forward(
        batch_size,
        cuda_val_op_[1], cuda_val_op_[0],
        mask_buf[0], mask_buf[1]);

    graph_->v_inter.Forward(
        batch_size, cuda_val_op_[2], cuda_val_op_[1]);

    graph_->v_ownership.Forward(
        batch_size,
        cuda_output_ownership_, cuda_val_op_[0],
        nullptr, nullptr,
        cuda_scratch_op_[0], cuda_scratch_op_[1], scratch_size_);

    graph_->v_misc.Forward(
        batch_size, cuda_output_val_, cuda_val_op_[2]);

    const auto probabilities_channels = weights_->probabilities_channels;
    const auto pass_probability_outputs = weights_->pass_probability_outputs;
    const auto value_misc_outputs = weights_->value_misc_outputs;
    const auto ownership_channels = weights_->ownership_channels;

    auto batch_prob = std::vector<float>(batch_size * probabilities_channels * num_intersections);
    auto batch_prob_pass = std::vector<float>(batch_size * pass_probability_outputs);
    auto batch_value_misc = std::vector<float>(batch_size * value_misc_outputs);
    auto batch_ownership = std::vector<float>(batch_size * ownership_channels * num_intersections);

    cuda::WaitToFinish(handles_.stream);

    {
        std::lock_guard<std::mutex> lock(io_mutex_);

        // copy the results to host memory
        cuda::SetDevice(handles_.gpu_id);

        cuda::CopyToHostOp(
            handles_.fp16, batch_prob,
            &cuda_output_prob_, &host_output_prob_);
        cuda::CopyToHostOp(
            handles_.fp16, batch_prob_pass,
            &cuda_output_prob_pass_, &host_output_prob_pass_);
        cuda::CopyToHostOp(
            handles_.fp16, batch_value_misc,
            &cuda_output_val_, &host_output_val_);
        cuda::CopyToHostOp(
            handles_.fp16, batch_ownership,
            &cuda_output_ownership_, &host_output_ownership_);
    }

    auto batch_output_result = std::vector<OutputResult>(batch_size);

    FillOutputs(batch_prob,
                batch_prob_pass,
                batch_value_misc,
                batch_ownership,
                batch_input,
                batch_output_result);
    return batch_output_result;
}

void CudaForwardPipe::NNGraph::FillOutputs(const std::vector<float> &batch_prob,
                                           const std::vector<float> &batch_prob_pass,
                                           const std::vector<float> &batch_value_misc,
                                           const std::vector<float> &batch_ownership,
                                           const std::vector<InputData> &batch_input,
                                           std::vector<OutputResult> &batch_output_result) {
    const int batch_size = batch_output_result.size();
    const auto num_intersections = board_size_ * board_size_;
    const auto encoder_version = Encoder::GetEncoderVersion(weights_->version); 
    const auto probabilities_channels = weights_->probabilities_channels;
    const auto pass_probability_outputs = weights_->pass_probability_outputs;
    const auto value_misc_outputs = weights_->value_misc_outputs;
    const auto ownership_channels = weights_->ownership_channels;

    if (encoder_version == 1) {
        for (int b = 0; b < batch_size; ++b) {
            auto &output_result = batch_output_result[b];
            const auto &input = batch_input[b];
            const int pol_offset = probabilities_channels * num_intersections;
            const int own_offset = ownership_channels * num_intersections;
            for (int idx = 0; idx < num_intersections; ++idx) {
                int pol_index = b * pol_offset + (int)PolicyBufferOffset::kNormal * num_intersections + idx;
                int own_index = b * own_offset + 0 * num_intersections + idx;
                output_result.probabilities[idx] = batch_prob[pol_index];
                output_result.ownership[idx] = batch_ownership[own_index];
            }
            output_result.pass_probability = batch_prob_pass[b * pass_probability_outputs + 0];

            output_result.wdl[0]      = batch_value_misc[b * value_misc_outputs + 0];
            output_result.wdl[1]      = batch_value_misc[b * value_misc_outputs + 1];
            output_result.wdl[2]      = batch_value_misc[b * value_misc_outputs + 2];
            output_result.stm_winrate = batch_value_misc[b * value_misc_outputs + 3];
            output_result.final_score = batch_value_misc[b * value_misc_outputs + 4];
            output_result.q_error     = 0.0f;
            output_result.score_error = 0.0f;

            output_result.offset = PolicyBufferOffset::kNormal;
            output_result.board_size = input.board_size;
            output_result.komi = input.komi;
            output_result.fp16 = handles_.fp16;
        }
    } else if (encoder_version == 2) {
        for (int b = 0; b < batch_size; ++b) {
            auto &output_result = batch_output_result[b];
            const auto &input = batch_input[b];
            const int pol_offset = probabilities_channels * num_intersections;
            const int own_offset = ownership_channels * num_intersections;
            for (int idx = 0; idx < num_intersections; ++idx) {
                int pol_index = b * pol_offset + (int)input.offset * num_intersections + idx;
                int own_index = b * own_offset + 0 * num_intersections + idx;
                output_result.probabilities[idx] = batch_prob[pol_index];
                output_result.ownership[idx] = batch_ownership[own_index];
            }
            output_result.pass_probability = batch_prob_pass[b * pass_probability_outputs + 0];

            output_result.wdl[0]      = batch_value_misc[b * value_misc_outputs + 0];
            output_result.wdl[1]      = batch_value_misc[b * value_misc_outputs + 1];
            output_result.wdl[2]      = batch_value_misc[b * value_misc_outputs + 2];
            output_result.stm_winrate = batch_value_misc[b * value_misc_outputs + 3];
            output_result.final_score = batch_value_misc[b * value_misc_outputs + 8];
            output_result.q_error     = batch_value_misc[b * value_misc_outputs + 13];
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

void CudaForwardPipe::AssignWorkers() {
    worker_running_.store(true);
    waittime_.store(GetOption<int>("gpu_waittime"), std::memory_order_relaxed);

    ThreadPool::Get("cuda-forward-pipe", nngraphs_.size());
    if (group_->FutureEmpty()) {
        for (int gpu = 0; gpu < (int)nngraphs_.size(); ++gpu) {
            group_->AddTask([g=gpu, this](){ Worker(g); });
        }
    }
}

void CudaForwardPipe::Worker(int gpu) {
    const auto GatherBatches = [this](int gpu_waittime) {
        auto entries = std::vector<std::shared_ptr<ForwawrdEntry>>{};

        // Running the loop until there are enough entries in the queue or time out,
        // then breaking the loop.
        {
            std::unique_lock<std::mutex> lock(worker_mutex_);
            while(true) {
                if (!worker_running_.load(std::memory_order_relaxed)) {
                    return entries;
                }
                if (static_cast<int>(entry_queue_.size()) >= forwarding_batch_per_nn_) {
                    break;
                }

                bool timeout = false;
                if (waittime_.load(std::memory_order_relaxed) != 0) {
                    // Wait for some time to avoid busy waiting.
                    timeout = !cv_.wait_for(
                        lock, std::chrono::milliseconds(waittime_.load(std::memory_order_relaxed)),
                        [this]() {
                            return !worker_running_.load(std::memory_order_relaxed) ||
                                       static_cast<int>(entry_queue_.size()) >= forwarding_batch_per_nn_; });
                }

                if (entry_queue_.empty()) {
                    // No any entry in the queue. In this case, we can not
                    // expect next forwarding time. Keep increasing waiting
                    // time.
                    auto last_waittime = waittime_.fetch_add(1, std::memory_order_relaxed);
                    if (last_waittime >= gpu_waittime) {
                        waittime_.store(gpu_waittime, std::memory_order_relaxed);
                    }
                } else {
                    if (timeout) {
                        // May be CPU-bound time. Boost forwarding.
                        waittime_.store(0, std::memory_order_relaxed);
                    }
                    break;
                }
            }
        }

        // Gather the entries and return.
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            auto count = std::min(static_cast<int>(entry_queue_.size()), forwarding_batch_per_nn_);
            auto end = std::begin(entry_queue_);
            std::advance(end, count);
            std::move(std::begin(entry_queue_), end, std::back_inserter(entries));
            entry_queue_.erase(std::begin(entry_queue_), end);
        }
        return entries;
    };

    const auto gpu_waittime_base = GetOption<int>("gpu_waittime");
    while (true) {
        if (!worker_running_.load(std::memory_order_relaxed)) return;

        auto entries = GatherBatches(gpu_waittime_base);
        const auto batch_size = entries.size();

        if (batch_size == 0) {
            continue;
        }

        // Gather batch data.
        auto inputs = std::vector<InputData>(batch_size);
        for (auto b = size_t{0}; b < batch_size; ++b) {
            inputs[b] = entries[b]->input;
        }

        // Forwarding...
        auto outputs = nngraphs_[gpu]->BatchForward(inputs);

        for (auto b = size_t{0}; b < batch_size; ++b) {
            entries[b]->output = outputs[b];
            {
                // Be sure the condition variable of current entry is ready.
                std::unique_lock<std::mutex> lk(entries[b]->mutex);
            }
            entries[b]->cv.notify_all();
        }
    }
}

void CudaForwardPipe::QuitWorkers() {
    worker_running_.store(false);
    cv_.notify_all();
    group_->WaitToJoin();
}

#endif
