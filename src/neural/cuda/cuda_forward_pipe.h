#pragma once

#ifdef USE_CUDA

#include <atomic>
#include <condition_variable>
#include <list>
#include <memory>
#include <vector>

#include "neural/batch_forward_pipe.h"
#include "neural/cuda/cuda_common.h"
#include "neural/cuda/cuda_nn_graph.h"
#include "neural/description.h"
#include "neural/network_basic.h"
#include "utils/threadpool.h"

class CudaForwardPipe : public BatchForwardPipe {
public:
    virtual void Initialize(std::shared_ptr<DNNWeights> weights);

    virtual OutputResult Forward(const InputData& input);

    virtual bool Valid() const;

    virtual void Construct(ForwardPipeOption option, std::shared_ptr<DNNWeights> weights);

    virtual void Release();

    virtual void Destroy();

    virtual int GetNumWorkers() const;

    virtual std::vector<OutputResult> BatchForward(int gpu, const std::vector<InputData>& inputs);

private:
    std::vector<std::unique_ptr<cuda::NNGraph>> nngraphs_;

    bool dump_gpu_info_;
    int max_batch_per_nn_{0};
    int board_size_{0};
};
#endif
