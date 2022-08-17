#pragma once

#ifdef USE_CUDA
#include <atomic>
#include <memory>
#include <list>
#include <array>
#include <vector>
#include <mutex>
#include <thread>
#include <condition_variable>

#include "neural/cuda/cuda_layers.h"
#include "neural/network_basic.h"
#include "neural/description.h"

class CudaForwardPipe : public NetworkForwardPipe {
public:
    virtual void Initialize(std::shared_ptr<DNNWeights> weights);

    virtual OutputResult Forward(const InputData &inpnt);

    virtual bool Valid();

    virtual void Load(std::shared_ptr<DNNWeights> weights);

    virtual void Reload(int board_size);

    virtual void Release();

    virtual void Destroy();

private:

    class NNGraph {
        struct Graph {
            // intput
            CUDA::Convolution input_conv;
            CUDA::Batchnorm input_bnorm;

            // residual towers
            std::vector<CUDA::Convolution> tower_conv;
            std::vector<CUDA::Batchnorm> tower_bnorm;
            std::vector<CUDA::SEUnit> tower_se;

            // policy head 
            CUDA::Convolution p_ex_conv;
            CUDA::Batchnorm p_ex_bnorm;
            CUDA::GlobalPool p_pool;
            CUDA::FullyConnect p_inter;

            CUDA::Convolution p_prob;
            CUDA::FullyConnect p_prob_pass;

            // value head
            CUDA::Convolution v_ex_conv;
            CUDA::Batchnorm v_ex_bnorm;
            CUDA::GlobalPool v_pool;
            CUDA::FullyConnect v_inter;

            CUDA::Convolution v_ownership;
            CUDA::FullyConnect v_misc;
        };

    public:
        NNGraph(std::mutex &mtx) : io_mutex_(mtx) {}
        ~NNGraph();
        void BuildGraph(const int gpu, 
                        const int max_batch_size,
                        const int board_size,
                        std::shared_ptr<DNNWeights> weights);

        std::vector<OutputResult> BatchForward(const std::vector<InputData> &inpnt);

        void DestroyGraph();

    private:
        CUDA::CudaHandles handles_;

        int board_size_;
        int max_batch_;

        std::unique_ptr<Graph> graph_{nullptr};

        float *cuda_scratch_;
        float *cuda_input_planes_;
        float *cuda_output_prob_;
        float *cuda_output_prob_pass_;
        float *cuda_output_val_;
        float *cuda_output_ownership_;

        std::array<float*, 3> cuda_conv_op_;
        std::array<float*, 3> cuda_pol_op_;
        std::array<float*, 3> cuda_val_op_;

        std::mutex &io_mutex_;

        size_t scratch_size_;
        std::shared_ptr<DNNWeights> weights_{nullptr};
    };

    struct ForwawrdEntry {
        std::atomic<bool> done{false};
	    const InputData &input;
        OutputResult &output;

        std::condition_variable cv;
        std::mutex mutex;

        ForwawrdEntry(const InputData &in,
                      OutputResult &out) :
                      input(in), output(out) {}
    };

    std::shared_ptr<DNNWeights> weights_{nullptr};

    std::list<std::shared_ptr<ForwawrdEntry>> entry_queue_;
    std::mutex worker_mutex_;
    std::mutex queue_mutex_;
    std::mutex io_mutex_;

    std::condition_variable cv_;

    std::atomic<int> waittime_{0};
    std::atomic<bool> worker_running_;
    std::atomic<bool> narrow_pipe_;

    std::vector<std::unique_ptr<NNGraph>> nngraphs_;
    std::vector<std::thread> workers_;

    int max_batch_;

    void PrepareWorkers();
    void Worker(int gpu);
    void QuitWorkers();
};
#endif
