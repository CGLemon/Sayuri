#ifdef USE_CUDA

#include <sstream>
#include <stdexcept>

#include "config.h"
#include "neural/cuda/cuda_forward_pipe.h"
#include "neural/cuda/cuda_kernels.h"
#include "utils/log.h"
#include "utils/format.h"
#include "utils/option.h"

void CudaForwardPipe::Initialize(std::shared_ptr<DNNWeights> weights) {
    LOGGING << cuda::GetBackendInfo();

    dump_gpu_info_ = true;

    Load(weights); // Will select max batch size.

    PrepareWorkers(); // Run the batch forwarding worker.
}

OutputResult CudaForwardPipe::Forward(const InputData &input) {
    OutputResult output;
    InputData reordered_input = input;

    // Reorder the inputs data.
    const int planes_bsize = input.board_size;
    const bool should_reorder = planes_bsize != board_size_;

    if (should_reorder) {
        for (int c = 0; c < kInputChannels; ++c) {
            int offset_r = c * board_size_ * board_size_;
            int offset_p = c * planes_bsize * planes_bsize;

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
        // Push the entry.
        std::lock_guard<std::mutex> queue_lock(queue_mutex_);
        entry_queue_.emplace_back(entry);
    }

    if (entry_queue_.size() >= (size_t)max_batch_) {
        cv_.notify_one(); // Wake up one worker if there are enough batch size.
    }
    entry->cv.wait(lock); // Wait for batch forwarding worker.
    entry->done.store(true, std::memory_order_relaxed);

    // Reorder the outputs data.
    OutputResult reordered_ouput = output;

    if (should_reorder) {
        int offset_r = 0;
        int offset_p = 0;
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

bool CudaForwardPipe::Valid() {
    return weights_ != nullptr;
}

void CudaForwardPipe::Load(std::shared_ptr<DNNWeights> weights) {
    weights_ = weights;
    Reload(GetOption<int>("defualt_boardsize"));
}

void CudaForwardPipe::Reload(int board_size) {
    if (board_size_ == board_size) {
        return;
    }

    Release();

    if (weights_ == nullptr) {
        return;
    }

    // Select the matched size.
    const int fixed_nn_boardsize =
                  std::max(board_size, GetOption<int>("fixed_nn_boardsize"));
    if (fixed_nn_boardsize > 0) {
        board_size_ = fixed_nn_boardsize;
    } else {
        board_size_ = board_size;
    }
    max_batch_ = GetOption<int>("batch_size");
    const auto d_cnt = cuda::GetDeviceCount();

    auto already_set_gpu = !IsOptionDefault("gpus");
    auto gpus_list = std::vector<int>{};

    if (!already_set_gpu) {
         for (int i = 0; i < d_cnt; ++i) {
             gpus_list.emplace_back(i);
         } 
    } else {
        auto gpus_cnt = GetOptionCount("gpus");
        for (int idx = 0; idx < gpus_cnt; ++idx) {
            auto gpu_id = GetOption<int>("gpus", idx);
            if (gpu_id < d_cnt) {
                gpus_list.emplace_back(gpu_id);
            } else {
                LOGGING << Format("Not found GPU device %d.\n", gpu_id);
            }
        }
    }

    if (gpus_list.empty()) {
        LOGGING << "Not found any GPU device! Now assign the GPU(s) automatically.\n";
        for (int i = 0; i < d_cnt; ++i) {
            gpus_list.emplace_back(i);
        }

        if (gpus_list.empty()) {
            throw std::runtime_error("No executable GPU device!");
        }
    }

    for (size_t i = 0; i < gpus_list.size(); ++i) {
        nngraphs_.emplace_back(std::make_unique<NNGraph>(io_mutex_));
    }

    // TODO: Assign different batch size by device computing capability.

    if (gpus_list.size() >= 2) {
        // Assign the the batch for each netork.
        const int num_gpus = gpus_list.size();
        max_batch_ = (max_batch_ / num_gpus) + bool(max_batch_ % num_gpus);
        max_batch_ = std::max(max_batch_, 1);
    }

    for (auto i = size_t{0}; i < gpus_list.size(); ++i) {
        nngraphs_[i]->BuildGraph(
            dump_gpu_info_, gpus_list[i], max_batch_, board_size_, weights_);
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

void CudaForwardPipe::NNGraph::BuildGraph(bool dump_gpu_info,
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

    use_optimistic_policy_ = GetOption<bool>("use_optimistic_policy");

    board_size_ = board_size;
    scratch_size_ = 0;
    max_batch_ = max_batch_size;

    // Build the graph first.
    const auto output_channels = weights_->residual_channels;

    // input layer
    graph_->input_conv = cuda::Convolution(
        &handles_,
        max_batch_,          // max batch size
        board_size_,         // board size
        3,                   // kernel size
        kInputChannels,      // input channels
        output_channels      // output channels
    );

    // residual tower
    const auto residuals = weights_->residual_blocks;
    for (int i = 0; i < residuals; ++i) {
        graph_->btl_conv.emplace_back(cuda::Convolution{});
        graph_->btl_conv.emplace_back(cuda::Convolution{});
        graph_->tower_conv.emplace_back(cuda::Convolution{});
        graph_->tower_conv.emplace_back(cuda::Convolution{});
        graph_->tower_se.emplace_back(cuda::SEUnit{});
    }

    for (int i = 0; i < residuals; ++i) {
        const auto t_offset = 2 * i;
        const auto tower_ptr = weights_->tower.data() + i;
        const auto outer_channels = weights_->residual_channels;
        const auto inner_channels = tower_ptr->apply_btl ?
                                        outer_channels/2 :
                                        outer_channels;
    
        graph_->tower_conv[t_offset+0] = cuda::Convolution(
            &handles_,
            max_batch_,     // max batch size
            board_size_,    // board size
            3,              // kernel size
            inner_channels, // input channels
            inner_channels  // output channels
        );

        const bool second_use_relu =
                       tower_ptr->apply_btl ||
                       !(tower_ptr->apply_se);
        graph_->tower_conv[t_offset+1] = cuda::Convolution(
            &handles_,
            max_batch_,     // max batch size
            board_size_,    // board size
            3,              // kernel size
            inner_channels, // input channels
            inner_channels, // output channels
            second_use_relu // relu
        );

        if (tower_ptr->apply_btl) {
            graph_->btl_conv[t_offset+0] = cuda::Convolution(
                &handles_,
                max_batch_,     // max batch size
                board_size_,    // board size
                1,              // kernel size
                outer_channels, // input channels
                inner_channels  // output channels
            );

            const bool post_use_relu =
                           !(tower_ptr->apply_se);
            graph_->btl_conv[t_offset+1] = cuda::Convolution(
                &handles_,
                max_batch_,     // max batch size
                board_size_,    // board size
                1,              // kernel size
                inner_channels, // input channels
                outer_channels, // output channels
                post_use_relu   // relu
            );
        } 

        if (tower_ptr->apply_se) {
            const size_t se_size = tower_ptr->se_size;
            const bool se_use_relu = true;
            graph_->tower_se[i] = cuda::SEUnit(
                &handles_,
                max_batch_,      // max batch size
                board_size_,     // board size
                outer_channels,  // channels
                se_size,         // SE size
                se_use_relu      // relu
            );
        }
    }

    // policy head
    const auto policy_extract_channels = weights_->policy_extract_channels;
    graph_->p_ex_conv = cuda::Convolution(
        &handles_,
        max_batch_,              // max batch size
        board_size_,             // board size
        1,                       // kernel size
        output_channels,         // input channels
        policy_extract_channels  // output channels
    );
    graph_->p_pool = cuda::GlobalPooling(
        &handles_,
        false,
        max_batch_,               // max batch size
        board_size_,              // board size
        policy_extract_channels   // input channels
    );
    graph_->p_inter = cuda::FullyConnect(
        &handles_,
        max_batch_,               // max batch size
        3*policy_extract_channels,// input sizes
        policy_extract_channels,  // outpur size
        true
    );
    graph_->p_prob = cuda::Convolution(
        &handles_,
        max_batch_,                  // max batch size
        board_size_,                 // board size
        1,                           // kernel size
        policy_extract_channels,     // input channels
        kOuputProbabilitiesChannels, // output channels
        false                        // relu
    );
    graph_->p_prob_pass = cuda::FullyConnect(
        &handles_,
        max_batch_,               // max batch size
        policy_extract_channels,  // input sizes
        kOuputPassProbability,    // outpur size
        false                     // relu
    );

    // value head
    const auto value_extract_channels = weights_->value_extract_channels;
    graph_->v_ex_conv = cuda::Convolution(
        &handles_,
        max_batch_,               // max batch size
        board_size_,              // board size
        1,                        // kernel size
        output_channels,          // input channels
        value_extract_channels    // output channels
    );
    graph_->v_pool = cuda::GlobalPooling(
        &handles_,
        true,
        max_batch_,               // max batch size
        board_size_,              // board size
        value_extract_channels    // input channels
    );
    graph_->v_inter = cuda::FullyConnect(
        &handles_,
        max_batch_,               // max batch size
        3*value_extract_channels, // input sizes
        3*value_extract_channels, // outpur size
        true
    );
    graph_->v_ownership = cuda::Convolution(
        &handles_,
        max_batch_,               // max batch size
        board_size_,              // board size
        1,                        // kernel size
        value_extract_channels,   // input channels
        kOuputOwnershipChannels,  // output channels
        false                     // relu
    );
    graph_->v_misc = cuda::FullyConnect(
        &handles_,
        max_batch_,               // max batch size
        3*value_extract_channels, // input size
        kOuputValueMisc,          // output size
        false                     // relu
    );

    // Now push the weights.

    const bool winograd = weights_->winograd;

    // input layer
    graph_->input_conv.LoadWeights(
        weights_->input_conv.GetWeights(),
        weights_->input_conv.GetBiases(),
        scratch_size_, winograd);

    // residual tower
    for (int i = 0; i < residuals; ++i) {
        const auto t_offset = 2 * i;
        const auto tower_ptr = weights_->tower.data() + i;

        graph_->tower_conv[t_offset+0].LoadWeights(
            tower_ptr->conv1.GetWeights(),
            tower_ptr->conv1.GetBiases(),
            scratch_size_, winograd);

        graph_->tower_conv[t_offset+1].LoadWeights(
            tower_ptr->conv2.GetWeights(),
            tower_ptr->conv2.GetBiases(),
            scratch_size_, winograd);

        if (tower_ptr->apply_btl) {
            graph_->btl_conv[t_offset+0].LoadWeights(
                tower_ptr->pre_btl_conv.GetWeights(),
                tower_ptr->pre_btl_conv.GetBiases(),
                scratch_size_, winograd);

            graph_->btl_conv[t_offset+1].LoadWeights(
                tower_ptr->post_btl_conv.GetWeights(),
                tower_ptr->post_btl_conv.GetBiases(),
                scratch_size_, winograd);
        }
        if (tower_ptr->apply_se) {
            graph_->tower_se[i].LoadWeights(
                tower_ptr->squeeze.GetWeights(),
                tower_ptr->squeeze.GetBiases(),
                tower_ptr->excite.GetWeights(),
                tower_ptr->excite.GetBiases());
        }
    }

    // policy head
    graph_->p_ex_conv.LoadWeights(
        weights->p_ex_conv.GetWeights(),
        weights->p_ex_conv.GetBiases(),
        scratch_size_, winograd);

    graph_->p_inter.LoadWeights(
        weights_->p_inter_fc.GetWeights(), weights_->p_inter_fc.GetBiases());

    graph_->p_prob.LoadWeights(
        weights->prob_conv.GetWeights(),
        weights_->prob_conv.GetBiases(),
        scratch_size_, winograd);

    graph_->p_prob_pass.LoadWeights(
        weights_->pass_fc.GetWeights(), weights_->pass_fc.GetBiases());

    // value head
    graph_->v_ex_conv.LoadWeights(
        weights->v_ex_conv.GetWeights(),
        weights->v_ex_conv.GetBiases(),
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

    const size_t planes_size = factor * kInputChannels * num_intersections;
    const size_t spatia_size = factor * num_intersections;
    const size_t pol_size = spatia_size * kOuputProbabilitiesChannels;
    const size_t pass_size = factor * kOuputPassProbability;
    const size_t val_size = factor * kOuputValueMisc;
    const size_t ownership_size = spatia_size * kOuputOwnershipChannels;

    const size_t conv_op_size = factor * weights_->residual_channels * num_intersections;

    const size_t pol_op1_size = factor * policy_extract_channels * num_intersections;
    const size_t pol_op2_size = factor * policy_extract_channels * 3;
    const size_t pol_op3_size = factor * policy_extract_channels;

    const size_t val_op1_size = factor * value_extract_channels * num_intersections;
    const size_t val_op2_size = factor * value_extract_channels * 3;
    const size_t val_op3_size = factor * value_extract_channels * 3;

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

    if (dev_prop.major <= 5 ||
            !GetOption<bool>("fp16")) {
        // The device is too old. Disable the 
        // FP16 computation.
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

bool CudaForwardPipe::NNGraph::ApplyMask(const std::vector<InputData> &inputs) {
    const int batch_size = inputs.size();
    if (batch_size == 0) {
        return false;
    }

    const int num_intersections = board_size_ * board_size_;
    bool should_apply_mask = false;

    for (int b = 0; b < batch_size; ++b) {
        if (board_size_ != inputs[b].board_size) {
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
            const int planes_bsize = inputs[b].board_size;
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

        // Copy the mask to device.
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

std::vector<OutputResult> CudaForwardPipe::NNGraph::BatchForward(const std::vector<InputData> &inputs) {
    const auto batch_size = (int)inputs.size();

    assert(max_batch_ >= batch_size);

    const auto should_apply_mask = ApplyMask(inputs);
    const auto num_intersections = board_size_ * board_size_;
    auto batch_planes = std::vector<float>(batch_size * kInputChannels * num_intersections);

    for (int b = 0; b < batch_size; ++b) {
        const auto& input = inputs[b];
        for (int idx = 0; idx < kInputChannels * num_intersections; ++idx) {
            batch_planes[b * kInputChannels * num_intersections + idx] = input.planes[idx];
        }
    }

    std::array<void *, 2> mask_buf = cuda_mask_op_;
    if (!should_apply_mask) {
        // Disable the mask.
        mask_buf[0] = mask_buf[1] = nullptr;
    }

    // copy the inputs to device
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

    //   The Residual tower. The forwarding order of
    //   each block is
    // [
    //      (pre-bottleneck)
    //   -> 1st conv layer
    //   -> 2nd conv layer
    //   -> (post-bottleneck)
    //   -> (squeeze-and-excitation module)
    //   -> (spatial attention module)
    // ]
    const auto residuals = weights_->residual_blocks;
    for (int i = 0; i < residuals; ++i) {
        // TODO: Remove one of cuda_conv_op_. Make it more
        //       clear.
        const auto t_offset = 2 * i;
        const auto tower_ptr = weights_->tower.data() + i;

        if (tower_ptr->apply_btl) {
            // pre-bottleneck
            graph_->btl_conv[t_offset+0].Forward(
                batch_size,
                cuda_conv_op_[1], cuda_conv_op_[0],
                nullptr, mask_buf[0],
                cuda_scratch_op_[0], cuda_scratch_op_[1], scratch_size_);
        }

        // 1st conv layer
        void *first_in = tower_ptr->apply_btl ?
                             cuda_conv_op_[1] : cuda_conv_op_[0];
        graph_->tower_conv[t_offset+0].Forward(
            batch_size,
            cuda_conv_op_[2], first_in,
            nullptr, mask_buf[0],
            cuda_scratch_op_[0], cuda_scratch_op_[1], scratch_size_);

        // 2nd conv layer
        void *second_skip = (tower_ptr->apply_se ||
                                 tower_ptr->apply_btl) ?
                                     nullptr : cuda_conv_op_[0];
        graph_->tower_conv[t_offset+1].Forward(
            batch_size,
            cuda_conv_op_[3], cuda_conv_op_[2],
            second_skip, mask_buf[0],
            cuda_scratch_op_[0],  cuda_scratch_op_[1], scratch_size_);

        if (tower_ptr->apply_btl) {
            std::swap(cuda_conv_op_[2], cuda_conv_op_[3]);

            // post-bottleneck
            void *btl_skip = tower_ptr->apply_se ?
                                     nullptr : cuda_conv_op_[0];
            graph_->btl_conv[t_offset+1].Forward(
                batch_size,
                cuda_conv_op_[3], cuda_conv_op_[2],
                btl_skip, mask_buf[0],
                cuda_scratch_op_[0], cuda_scratch_op_[1], scratch_size_);
        }

        bool module_skip = false;
        if (tower_ptr->apply_se) {
            // squeeze-and-excitation module
            void *se_skip = cuda_conv_op_[0];
            void *se_outs = cuda_conv_op_[0];

            graph_->tower_se[i].Forward(
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
    const auto policy_extract_channels = weights_->policy_extract_channels;

    graph_->p_ex_conv.Forward(
        batch_size,
        cuda_pol_op_[0], cuda_conv_op_[0],
        nullptr, mask_buf[0],
        cuda_scratch_op_[0], cuda_scratch_op_[1], scratch_size_);

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
        batch_size * policy_extract_channels,
        batch_size, policy_extract_channels, num_intersections,
        false, handles_.stream);

    graph_->p_prob.Forward(
        batch_size,
        cuda_output_prob_, cuda_pol_op_[0],
        nullptr, nullptr,
        cuda_scratch_op_[0], cuda_scratch_op_[1], scratch_size_);

    graph_->p_prob_pass.Forward(
        batch_size, cuda_output_prob_pass_, cuda_pol_op_[2]);

    // value head
    graph_->v_ex_conv.Forward(
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

    auto batch_prob_pass = std::vector<float>(batch_size * kOuputPassProbability);
    auto batch_prob = std::vector<float>(batch_size * kOuputProbabilitiesChannels * num_intersections);
    auto batch_ownership = std::vector<float>(batch_size * kOuputOwnershipChannels * num_intersections);
    auto batch_value_misc = std::vector<float>(batch_size * kOuputValueMisc);

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

    for (int b = 0; b < batch_size; ++b) {
        auto &output_result = batch_output_result[b];
        int pol_offset = kOuputProbabilitiesChannels * num_intersections;
        int own_offset = kOuputOwnershipChannels * num_intersections;
        for (int idx = 0; idx < num_intersections; ++idx) {
            int pol_index = b * pol_offset + 0 * num_intersections + idx;
            if (use_optimistic_policy_) {
                pol_index = b * pol_offset + 4 * num_intersections + idx;
            }
            int own_index = b * own_offset + 0 * num_intersections + idx;
            output_result.probabilities[idx] = batch_prob[pol_index];
            output_result.ownership[idx] = batch_ownership[own_index];
        }
        output_result.pass_probability = batch_prob_pass[b * kOuputPassProbability + 0];

        output_result.wdl[0] = batch_value_misc[b * kOuputValueMisc + 0];
        output_result.wdl[1] = batch_value_misc[b * kOuputValueMisc + 1];
        output_result.wdl[2] = batch_value_misc[b * kOuputValueMisc + 2];
        output_result.stm_winrate = batch_value_misc[b * kOuputValueMisc + 3];
        output_result.final_score = batch_value_misc[b * kOuputValueMisc + 8];
        output_result.q_error = batch_value_misc[b * kOuputValueMisc + 13];
        output_result.score_error = batch_value_misc[b * kOuputValueMisc + 14];

        output_result.board_size = inputs[b].board_size;
        output_result.komi = inputs[b].komi;
        output_result.fp16 = handles_.fp16;
    }

    return batch_output_result;
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

void CudaForwardPipe::PrepareWorkers() {
    worker_running_.store(true);
    if (workers_.empty()) {
        for (int gpu = 0; gpu < (int)nngraphs_.size(); ++gpu) {
            workers_.emplace_back([g=gpu, this](){ Worker(g); });
        }
    }
}

void CudaForwardPipe::Worker(int gpu) {
    const auto gpu_waittime_base = GetOption<int>("gpu_waittime");
    waittime_.store(gpu_waittime_base, std::memory_order_relaxed);

    const auto GatherBatches = [this, gpu_waittime_base](){
        const auto max_waittime = std::max(10 * gpu_waittime_base, 100);
        auto entries = std::vector<std::shared_ptr<ForwawrdEntry>>{};

        // Running the loop until there is enough entry size.
        while(true) {
            if (!worker_running_.load(std::memory_order_relaxed)) {
                return entries;
            }

            bool should_be_fast = fast_pipe_.exchange(false, std::memory_order_relaxed);
            int waittime = waittime_.load(std::memory_order_relaxed);

            if ((int)entry_queue_.size() >= max_batch_) {
                // Threre are enough batches. Finish the loop.
                waittime_.store(
                    std::min(waittime, gpu_waittime_base),
                    std::memory_order_relaxed);
                break;
            }

            // Wait for some time in order to avoid busy waiting.
            std::unique_lock<std::mutex> lock(worker_mutex_);
            bool timeout = !cv_.wait_for(lock, std::chrono::milliseconds(waittime),
                                             [this](){ return !((int)entry_queue_.size() < max_batch_); }
                                         );

            // Reset the waiting time.
            if (!entry_queue_.empty()) {
                waittime = std::min(waittime, gpu_waittime_base);

                if (timeout && should_be_fast) {
                    // We wait two times and there are always not enough batches.
                    // Simply assume threre still are not next time so set the
                    // waiting time as zero.
                    waittime = 0;
                } else if (waittime > 0) {
                    // Decrease the waiting time if it is time out.
                    waittime -= 2;
                }

                // Set the next waiting time.
                waittime_.store(std::max(waittime, 0), std::memory_order_relaxed);

                // Finish the loop.
                break;
            } else {
                if (waittime < gpu_waittime_base) {
                    waittime_.store(waittime+1, std::memory_order_relaxed);
                } else if (waittime < max_waittime) {
                    waittime_.store(waittime+10, std::memory_order_relaxed);
                }
            }
        }

        // Gather the entries.
        std::lock_guard<std::mutex> queue_lock(queue_mutex_);
        auto count = entry_queue_.size();
        if ((int)count > max_batch_) {
            count = max_batch_;
        }

        auto end = std::begin(entry_queue_);
        std::advance(end, count);
        std::move(std::begin(entry_queue_), end, std::back_inserter(entries));
        entry_queue_.erase(std::begin(entry_queue_), end);

        return entries;
    };

    while (true) {
        if (!worker_running_.load(std::memory_order_relaxed)) return;

        auto entries = GatherBatches();
        const auto batch_size = entries.size();

        if (batch_size == 0) {
            continue;
        }

        auto inputs = std::vector<InputData>(batch_size);
        for (auto b = size_t{0}; b < batch_size; ++b) {
            inputs[b] = entries[b]->input;
        }

        auto outputs = nngraphs_[gpu]->BatchForward(inputs);

        for (auto b = size_t{0}; b < batch_size; ++b) {
            entries[b]->output = outputs[b];
            while (!entries[b]->done.load(std::memory_order_relaxed)) {
                entries[b]->cv.notify_all();
            }
        }

        if ((int)batch_size <= max_batch_) {
            fast_pipe_.store(true, std::memory_order_relaxed);
        }
    }
}

void CudaForwardPipe::QuitWorkers() {
    worker_running_.store(false);
    cv_.notify_all();
    for (auto &t : workers_) {
        t.join();
    }
}

#endif
