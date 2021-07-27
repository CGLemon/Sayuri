#ifdef USE_CUDA

#include "config.h"

#include "neural/cuda/cuda_forward_pipe.h"
#include "neural/cuda/cuda_common.h"
#include "neural/cuda/cuda_kernels.h"
#include "utils/log.h"

void CudaForwardPipe::Initialize(std::shared_ptr<DNNWeights> weights) {
    LOGGING << CUDA::GetDevicesInfo();
    max_batch_ = GetOption<int>("batch_size");
    Load(weights);
    PrepareWorkers();
}

OutputResult CudaForwardPipe::Forward(const InputData &inpnt) {
    OutputResult output;
    auto entry = std::make_shared<ForwawrdEntry>(inpnt, output);
    std::unique_lock<std::mutex> lock(entry->mutex);
    {
        std::unique_lock<std::mutex> queue_lock(queue_mutex_);
        entry_queue_.emplace_back(entry);
    }

    if (entry_queue_.size() >= (size_t)max_batch_) {
        cv_.notify_one();
    }
    entry->cv.wait(lock);
    entry->done.store(true);

    return output;
}

bool CudaForwardPipe::Valid() {
    return weights_->loaded;
}

void CudaForwardPipe::Load(std::shared_ptr<DNNWeights> weights) {
    weights_ = weights;
    Reload(GetOption<int>("defualt_boardsize"));
}

void CudaForwardPipe::Reload(int board_size) {
    Release();

    const auto d_cnt = CUDA::GetDeviceCount();

    if (GetOption<int>("gpu") >= 0) {
        nngraphs_.emplace_back(std::make_unique<NNGraph>());
        const auto gpu = GetOption<int>("gpu") >= d_cnt ? 0 : GetOption<int>("gpu");
        nngraphs_[0]->BuildGraph(gpu, max_batch_, board_size, weights_);
    } else {
        for (int i = 0; i < d_cnt; ++i) {
            nngraphs_.emplace_back(std::make_unique<NNGraph>());
        }
        for (int i = 0; i < d_cnt; ++i) {
            nngraphs_[i]->BuildGraph(i, max_batch_, board_size, weights_);
        }
    }
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

void CudaForwardPipe::NNGraph::BuildGraph(const int gpu,
                                          const int max_batch_size,
                                          const int board_size,
                                          std::shared_ptr<DNNWeights> weights) {
    if (graph_ != nullptr) {
        return;
    }
    graph_ = std::make_unique<Graph>();
    gpu_ = gpu;

    CUDA::SetDevice(gpu_);

    weights_ = weights;
    handel_.ApplyOnCurrentDevice();

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
        output_channels      // channels
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

    CUDA::ReportCUDAErrors(cudaMalloc(&cuda_scratch_, scratch_size_));
    CUDA::ReportCUDAErrors(cudaMalloc(&cuda_input_planes_, planes_size));

    CUDA::ReportCUDAErrors(cudaMalloc(&cuda_conv_op_[0], conv_op_size));
    CUDA::ReportCUDAErrors(cudaMalloc(&cuda_conv_op_[1], conv_op_size));
    CUDA::ReportCUDAErrors(cudaMalloc(&cuda_conv_op_[2], conv_op_size));

    CUDA::ReportCUDAErrors(cudaMalloc(&cuda_pol_op_[0], pol_op1_size));
    CUDA::ReportCUDAErrors(cudaMalloc(&cuda_pol_op_[1], pol_op2_size));

    CUDA::ReportCUDAErrors(cudaMalloc(&cuda_val_op_[0], val_op1_size));
    CUDA::ReportCUDAErrors(cudaMalloc(&cuda_val_op_[1], val_op2_size));

    CUDA::ReportCUDAErrors(cudaMalloc(&cuda_output_prob_pass_, factor));
    CUDA::ReportCUDAErrors(cudaMalloc(&cuda_output_prob_, spatia_size));
    CUDA::ReportCUDAErrors(cudaMalloc(&cuda_output_ownership_, spatia_size));
    CUDA::ReportCUDAErrors(cudaMalloc(&cuda_output_val_, val_size));
}

std::vector<OutputResult> CudaForwardPipe::NNGraph::BatchForward(const std::vector<InputData> &inpnts) {
    const auto batch_size = (int)inpnts.size();
    auto bactch_output_result = std::vector<OutputResult>(batch_size);

    assert(max_batch_ >= batch_size);

    const auto num_intersections = board_size_ * board_size_;
    auto batch_planes = std::vector<float>(batch_size * kInputChannels * num_intersections);

    for (int b = 0; b < batch_size; ++b) {
        const auto& inpnt = inpnts[b];
        for (int idx = 0; idx < kInputChannels * num_intersections; ++idx) {
            batch_planes[b * num_intersections + idx] = inpnt.planes[idx];
        }
    }

    CUDA::ReportCUDAErrors(cudaMemcpy(cuda_input_planes_,
                                      batch_planes.data(),
                                      batch_planes.size() * sizeof(float),
                                      cudaMemcpyHostToDevice));

    graph_->input_conv.Forward(batch_size,
                               cuda_input_planes_, cuda_conv_op_[0],
                               cuda_scratch_, scratch_size_, &handel_);
    graph_->input_bnorm.Forward(batch_size,
                                cuda_conv_op_[0]);

    // residual tower
    const auto residuals = weights_->residual_blocks;
    for (int i = 0; i < residuals; ++i) {
        const auto t_offset = 2 * i;
        const auto tower_ptr = weights_->tower.data() + i;

        graph_->tower_conv[t_offset+0].Forward(batch_size,
                                               cuda_conv_op_[0], cuda_conv_op_[1],
                                               cuda_scratch_, scratch_size_, &handel_);
        graph_->tower_bnorm[t_offset+0].Forward(batch_size,
                                                cuda_conv_op_[1]);

        graph_->tower_conv[t_offset+1].Forward(batch_size,
                                               cuda_conv_op_[1], cuda_conv_op_[2],
                                               cuda_scratch_, scratch_size_, &handel_);
        if (tower_ptr->apply_se) {
            graph_->tower_bnorm[t_offset+1].Forward(batch_size,
                                                    cuda_conv_op_[2]);
            graph_->tower_se[i].Forward(batch_size,
                                        cuda_conv_op_[2], cuda_conv_op_[0], &handel_);
        } else { 
            const auto tower_channels = weights_->residual_channels;
            graph_->tower_bnorm[t_offset+1].Forward(batch_size,
                                                    cuda_conv_op_[2], cuda_conv_op_[0]);
            CUDA::copy(cuda_conv_op_[0], cuda_conv_op_[2], batch_size * tower_channels * num_intersections);
        }
    }

    // policy head
    graph_->p_ex_conv.Forward(batch_size,
                               cuda_conv_op_[0], cuda_pol_op_[0],
                               cuda_scratch_, scratch_size_, &handel_);
    graph_->p_ex_bnorm.Forward(batch_size, cuda_pol_op_[0]);

    graph_->p_prob.Forward(batch_size,
                            cuda_pol_op_[0], cuda_output_prob_,
                            cuda_scratch_, scratch_size_, &handel_);

    graph_->p_pool.Forward(batch_size,
                           cuda_pol_op_[0], cuda_pol_op_[1]);
 
    graph_->p_prob_pass.Forward(batch_size,
                                cuda_pol_op_[1], cuda_output_prob_pass_, &handel_);

    // value head
    graph_->v_ex_conv.Forward(batch_size,
                              cuda_conv_op_[0], cuda_val_op_[0],
                              cuda_scratch_, scratch_size_, &handel_);
    graph_->v_ex_bnorm.Forward(batch_size, cuda_val_op_[0]);

    graph_->v_ownership.Forward(batch_size,
                                cuda_val_op_[0], cuda_output_ownership_,
                                cuda_scratch_, scratch_size_, &handel_);
    graph_->v_pool.Forward(batch_size,
                           cuda_val_op_[0], cuda_val_op_[1]);
    graph_->v_misc.Forward(batch_size,
                           cuda_val_op_[1], cuda_output_val_, &handel_);

    auto batch_prob_pass = std::vector<float>(batch_size);
    auto batch_prob = std::vector<float>(batch_size * num_intersections);
    auto batch_ownership = std::vector<float>(batch_size * num_intersections);
    auto batch_value_misc = std::vector<float>(batch_size * kOuputValueMisc);

    CUDA::ReportCUDAErrors(cudaMemcpy(batch_prob.data(), cuda_output_prob_,
                                      batch_prob.size() * sizeof(float),
                                      cudaMemcpyDeviceToHost));

    CUDA::ReportCUDAErrors(cudaMemcpy(batch_prob_pass.data(), cuda_output_prob_pass_,
                                      batch_prob_pass.size() * sizeof(float),
                                      cudaMemcpyDeviceToHost));
 
    CUDA::ReportCUDAErrors(cudaMemcpy(batch_value_misc.data(), cuda_output_val_,
                                      batch_value_misc.size() * sizeof(float),
                                      cudaMemcpyDeviceToHost));

    CUDA::ReportCUDAErrors(cudaMemcpy(batch_ownership.data(), cuda_output_ownership_,
                                      batch_ownership.size() * sizeof(float),
                                      cudaMemcpyDeviceToHost));

    for (int b = 0; b < batch_size; ++b) {
        auto &output_result = bactch_output_result[b];
        for (int idx = 0; idx < num_intersections; ++idx) {
            output_result.probabilities[idx] = batch_prob[b * num_intersections + idx];
            output_result.ownership[idx] = batch_ownership[b * num_intersections + idx];
        }
        output_result.pass_probability = batch_prob_pass[b];

        output_result.wdl[0] = batch_value_misc[b * kOuputValueMisc + 0];
        output_result.wdl[1] = batch_value_misc[b * kOuputValueMisc + 1];
        output_result.wdl[2] = batch_value_misc[b * kOuputValueMisc + 2];
        output_result.stm_winrate = batch_value_misc[b * kOuputValueMisc + 3];
        output_result.final_score = batch_value_misc[b * kOuputValueMisc + 4];

        output_result.board_size = inpnts[b].board_size;
    }

    return bactch_output_result;
}

void CudaForwardPipe::NNGraph::DestroyGraph() {
    if (graph_ == nullptr) {
        return;
    }

    CUDA::ReportCUDAErrors(cudaFree(cuda_scratch_));

    CUDA::ReportCUDAErrors(cudaFree(cuda_input_planes_));
    CUDA::ReportCUDAErrors(cudaFree(cuda_output_prob_));
    CUDA::ReportCUDAErrors(cudaFree(cuda_output_prob_pass_));
    CUDA::ReportCUDAErrors(cudaFree(cuda_output_val_));
    CUDA::ReportCUDAErrors(cudaFree(cuda_output_ownership_));

    CUDA::ReportCUDAErrors(cudaFree(cuda_conv_op_[0]));
    CUDA::ReportCUDAErrors(cudaFree(cuda_conv_op_[1]));
    CUDA::ReportCUDAErrors(cudaFree(cuda_conv_op_[2]));

    CUDA::ReportCUDAErrors(cudaFree(cuda_pol_op_[0]));
    CUDA::ReportCUDAErrors(cudaFree(cuda_pol_op_[1]));
    CUDA::ReportCUDAErrors(cudaFree(cuda_val_op_[0]));
    CUDA::ReportCUDAErrors(cudaFree(cuda_val_op_[1]));

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
    const auto gether_batches = [this](){
        const auto gpu_waittime = GetOption<int>("gpu_waittime");
        auto entries = std::vector<std::shared_ptr<ForwawrdEntry>>{};
        while(true) {
            if (!worker_running_.load()) {
                return entries;
            }

            bool narrow_pipe = narrow_pipe_.exchange(false);
            int waittime = waittime_.load();

            if ((int)entry_queue_.size() >= max_batch_) {
                break;
            }

            std::unique_lock<std::mutex> lock(mutex_);
            bool timeout = !cv_.wait_for(lock, std::chrono::milliseconds(waittime),
                                             [this](){ return !((int)entry_queue_.size() < max_batch_); }
                                         );

            if (!entry_queue_.empty()) {
                waittime = std::min(waittime, gpu_waittime);

                if (timeout && narrow_pipe) {
                    waittime = 0;
                } else if (waittime > 0) {
                    waittime -= 2;
                }

                waittime = std::max(waittime, 0);
                waittime_.store(waittime);

                break;
            } else {
                if (waittime < gpu_waittime) {
                    waittime_.store(waittime+1);
                } else if (waittime < 20 * gpu_waittime) {
                    waittime_.store(waittime+10);
                }
            }
        }

        std::unique_lock<std::mutex> queue_lock(queue_mutex_);
        auto count = entry_queue_.size();
        if (count > (size_t)max_batch_) {
            count = max_batch_;
        }

        auto end = std::begin(entry_queue_);
        std::advance(end, count);
        std::move(std::begin(entry_queue_), end, std::back_inserter(entries));
        entry_queue_.erase(std::begin(entry_queue_), end);

        return entries;
    };

    while (true) {
        if (!worker_running_) return;

        auto entries = gether_batches();
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
            while (!entries[b]->done.load()) {
                entries[b]->cv.notify_all();
            }
        }

        if (batch_size <= (size_t)max_batch_) {
            narrow_pipe_.store(false);
        }
    }
}

void CudaForwardPipe::QuitWorkers() {
    {
        std::unique_lock<std::mutex> queue_lock(queue_mutex_);
        worker_running_.store(false);
    }
    cv_.notify_all();
    for (auto &t : workers_) {
        t.join();
    }
}

#endif
