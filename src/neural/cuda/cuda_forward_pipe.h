#pragma once

#ifdef USE_CUDA
#include <atomic>
#include <memory>
#include <list>
#include <array>
#include <vector>
#include <mutex>
#include <condition_variable>

#include "neural/cuda/cuda_common.h"
#include "neural/cuda/cuda_layers.h"
#include "neural/activation.h"
#include "neural/network_basic.h"
#include "neural/description.h"
#include "utils/threadpool.h"

class CudaForwardPipe : public NetworkForwardPipe {
public:
    virtual void Initialize(std::shared_ptr<DNNWeights> weights);

    virtual OutputResult Forward(const InputData &input);

    virtual bool Valid();

    virtual void Load(std::shared_ptr<DNNWeights> weights);

    virtual void Reload(int board_size);

    virtual void Release();

    virtual void Destroy();

private:

    class NNGraph {
        struct Block {
            cuda::DepthwiseConvolution dw_conv;
            cuda::Convolution pre_btl_conv;
            cuda::Convolution conv1;
            cuda::Convolution conv2;
            cuda::Convolution post_btl_conv;
            cuda::SEUnit se_module;
        };

        struct Graph {
            // intput
            cuda::Convolution input_conv;

            // block tower
            std::vector<NNGraph::Block> tower;

            // policy head
            cuda::Convolution p_ex_conv;
            cuda::GlobalPooling p_pool;
            cuda::FullyConnect p_inter;

            cuda::Convolution p_prob;
            cuda::FullyConnect p_prob_pass;

            // value head
            cuda::Convolution v_ex_conv;
            cuda::GlobalPooling v_pool;
            cuda::FullyConnect v_inter;

            cuda::Convolution v_ownership;
            cuda::FullyConnect v_misc;
        };

    public:
        NNGraph(std::mutex &mtx) : io_mutex_(mtx) {}
        ~NNGraph();
        void BuildGraph(bool dump_gpu_info,
                        const int gpu,
                        const int max_batch_size,
                        const int board_size,
                        std::shared_ptr<DNNWeights> weights);

        std::vector<OutputResult> BatchForward(const std::vector<InputData> &input);

        void DestroyGraph();

    private:
        void SetComputationMode(cuda::CudaHandles *handles);

        bool ApplyMask(const std::vector<InputData> &input);

        void FillOutputs(const std::vector<float> &batch_prob,
                         const std::vector<float> &batch_prob_pass,
                         const std::vector<float> &batch_value_misc,
                         const std::vector<float> &batch_ownership,
                         const std::vector<InputData> &batch_input,
                         std::vector<OutputResult> &batch_output_result);

        cuda::CudaHandles handles_;

        int board_size_{0};
        int max_batch_;

        std::unique_ptr<Graph> graph_{nullptr};

        void *host_input_planes_;
        void *host_output_prob_;
        void *host_output_prob_pass_;
        void *host_output_val_;
        void *host_output_ownership_;

        void *cuda_input_planes_;
        void *cuda_output_prob_;
        void *cuda_output_prob_pass_;
        void *cuda_output_val_;
        void *cuda_output_ownership_;

        std::array<void*, 2> host_mask_op_;

        std::array<void*, 2> cuda_scratch_op_;
        std::array<void*, 4> cuda_conv_op_;
        std::array<void*, 3> cuda_pol_op_;
        std::array<void*, 3> cuda_val_op_;
        std::array<void*, 2> cuda_mask_op_;

        std::mutex &io_mutex_;

        size_t scratch_size_;
        std::shared_ptr<DNNWeights> weights_{nullptr};
    };

    struct ForwawrdEntry {
	    const InputData &input;
        OutputResult &output;

        std::condition_variable cv;
        std::mutex mutex;

        ForwawrdEntry(const InputData &in,
                      OutputResult &out) :
                      input(in), output(out) {}
    };

    std::list<std::shared_ptr<ForwawrdEntry>> entry_queue_;
    std::mutex worker_mutex_;
    std::mutex queue_mutex_;
    std::mutex io_mutex_;

    std::condition_variable cv_;

    std::atomic<int> waittime_{0};
    std::atomic<bool> worker_running_;

    std::vector<std::unique_ptr<NNGraph>> nngraphs_;
    std::unique_ptr<ThreadGroup<void>> group_;

    bool dump_gpu_info_;
    int max_batch_per_nn_;
    int board_size_{0};

    void AssignWorkers();
    void Worker(int gpu);
    void QuitWorkers();
};
#endif
