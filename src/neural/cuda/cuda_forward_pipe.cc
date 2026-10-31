#ifdef USE_CUDA

#include "neural/cuda/cuda_forward_pipe.h"

#include <sstream>
#include <stdexcept>

#include "utils/format.h"
#include "utils/log.h"
#include "utils/option.h"

void CudaForwardPipe::Initialize(std::shared_ptr<DNNWeights> weights) {
    LOGGING << cuda::GetBackendInfo();

    dump_gpu_info_ = true;

    auto option = ForwardPipeOption::Get()
                      .SetBoardSize(GetOption<int>("default_boardsize"))
                      .SetBatchSize(GetOption<int>("batch_size"));
    Construct(option, weights);

    BatchForwardPipe::AssignWorkers(nngraphs_.size());
}

OutputResult CudaForwardPipe::Forward(const InputData& input) {
    return BatchForwardPipe::SendQueryAndWait(input);
}

std::vector<OutputResult> CudaForwardPipe::BatchForward(int gpu,
                                                        const std::vector<InputData>& inputs) {
    return nngraphs_[gpu]->BatchForward(inputs);
}

bool CudaForwardPipe::Valid() const {
    return weights_ != nullptr;
}

int CudaForwardPipe::GetNumWorkers() const {
    return static_cast<int>(nngraphs_.size());
}

void CudaForwardPipe::Construct(ForwardPipeOption option, std::shared_ptr<DNNWeights> weights) {
    // Construct the network with parameters (e.g., board_size) and weights.
    // If the current parameters are the same as the new ones, exit the function
    // immediately.
    if (weights) {
        weights_ = weights;
    }
    if (weights_ == nullptr) {
        // Note: Will use dummy backend
        return;
    }

    int board_size = option.IsValidBoardSize() ? option.board_size : board_size_;
    int batch_size = option.IsValidBatchSize() ? option.batch_size : max_batch_per_nn_;
    // Select the matched board size.
    board_size = std::max(board_size, GetOption<int>("fixed_nn_boardsize"));

    if (board_size == 0 || batch_size == 0) {
        LOGGING << "NN board size/batch size should be larger than zero.\n";
        return;
    }

    // Reject the unsupported blocks before allocating any GPU resource.
    for (auto& tower_ptr : weights_->tower) {
        if (!tower_ptr->IsResidualBlock() && !tower_ptr->IsBottleneckBlock() &&
            !tower_ptr->IsNestedBottleneckBlock() && !tower_ptr->IsMixerBlock() &&
            !tower_ptr->IsTransformerBlock()) {
            throw std::runtime_error("Unknown block type for CUDA backend.");
        }
    }

    // The actual inference batch size may be smaller than the maximum batch size
    // supported by the neural network. In this case, we only need to dynamically
    // adjust the batch size used for the current forward pipe.
    BatchForwardPipe::SetForwardingSize(batch_size);

    if (board_size_ == board_size && batch_size <= max_batch_per_nn_) {
        // The current network already supports this configuration. No reinitialization is
        // required.
        return;
    }
    Release();

    board_size_ = board_size;
    max_batch_per_nn_ = batch_size;
    BatchForwardPipe::SetBoardSize(board_size);

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
        nngraphs_.emplace_back(std::make_unique<cuda::NNGraph>());
    }

    // Construct network graph for each valid GPU.
    for (auto i = size_t{0}; i < gpus_list.size(); ++i) {
        nngraphs_[i]->ConstructGraph(
            dump_gpu_info_, gpus_list[i], max_batch_per_nn_, board_size_, weights_);
    }

    dump_gpu_info_ = false; // don't show the GPU info next time.
}

void CudaForwardPipe::Release() {
    for (auto& g : nngraphs_) {
        g->DestroyGraph();
    }
    nngraphs_.clear();
}

void CudaForwardPipe::Destroy() {
    Release();
    BatchForwardPipe::QuitWorkers();
}

#endif
