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

        CUDA::Convolution p_prob;
        CUDA::GlobalAvgPool p_pool;
        CUDA::FullyConnect p_prob_pass;

        // value head
        CUDA::Convolution v_ex_conv;
        CUDA::Batchnorm v_ex_bnorm;

        CUDA::Convolution v_ownership;
        CUDA::GlobalAvgPool v_pool;
        CUDA::FullyConnect v_misc;
    };

    class NNGraph {
    public:
        ~NNGraph();
        void BuildGraph(const int gpu, 
                        const int max_batch_size,
                        const int board_size,
                        std::shared_ptr<DNNWeights> weights);

        std::vector<OutputResult> BatchForward(const std::vector<InputData> &inpnt);

        void DestroyGraph();

    private:
        CUDA::CudaHandel handel_;

        int board_size_;
        int gpu_;
        int max_batch_;

        std::unique_ptr<Graph> graph_{nullptr};

        float *cuda_scratch_;
        float *cuda_input_planes_;
        float *cuda_output_prob_;
        float *cuda_output_prob_pass_;
        float *cuda_output_val_;
        float *cuda_output_ownership_;

        std::array<float*, 3> cuda_conv_op_;
        std::array<float*, 2> cuda_pol_op_;
        std::array<float*, 2> cuda_val_op_;

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

    std::shared_ptr<DNNWeights> weights_{nullptr};

    std::list<std::shared_ptr<ForwawrdEntry>> entry_queue_;
    std::mutex mutex_;
    std::mutex queue_mutex_;
    std::condition_variable cv_;

    std::atomic<int> waittime_{20};
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
