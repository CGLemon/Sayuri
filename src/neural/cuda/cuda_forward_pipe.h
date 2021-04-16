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

#include "config.h"
#include "neural/cuda/cuda_layers.h"
#include "neural/cuda/cuda_kernels.h"
#include "neural/cuda/cuda_common.h"

#include <memory>

#include "neural/network_basic.h"
#include "neural/description.h"


#include <atomic>
#include <memory>
#include <list>
#include <array>
#include <vector>
#include <mutex>
#include <thread>
#include <condition_variable>

class CudaForwardPipe : public NetworkForwardPipe {
public:
    virtual void initialize(std::shared_ptr<Model::NNWeights> weights);
    virtual void forward(const std::vector<float> &planes,
                         const std::vector<float> &features,
                         std::vector<float> &output_pol,
                         std::vector<float> &output_val);

    virtual void reload(std::shared_ptr<Model::NNWeights> weights);
    virtual void release();
    virtual void destroy();
    virtual bool valid();

private:
   struct Graph {
        // intput
        CUDA::Convolve input_conv;
        CUDA::Batchnorm input_bnorm;
        CUDA::InputPool input_pool;

        // residual towers
        std::vector<CUDA::Convolve> tower_conv;
        std::vector<CUDA::Batchnorm> tower_bnorm;
        std::vector<CUDA::SEUnit> tower_se;

        // policy head 
        CUDA::Convolve p_ex_conv;
        CUDA::Batchnorm p_ex_bnorm;
        CUDA::Convolve p_map;
  
        // value head
        CUDA::Convolve v_ex_conv;
        CUDA::Batchnorm v_ex_bnorm;
        CUDA::FullyConnect v_fc1;
        CUDA::FullyConnect v_fc2;
    };

    class NNGraph {
    public:
         ~NNGraph();
        void build_graph(const int gpu, std::shared_ptr<Model::NNWeights> weights);
        void batch_forward(const int batch_size,
                           std::vector<float> &planes,
                           std::vector<float> &features,
                           std::vector<float> &output_pol,
                           std::vector<float> &output_val);
        void destroy_graph();

    private:
        CUDA::CudaHandel m_handel;

        int m_gpu;
        int m_maxbatch;
        std::unique_ptr<Graph> m_graph{nullptr};
        float *cuda_scratch;
        float *cuda_input_features;
        float *cuda_input_planes;
        float *cuda_output_pol;
        float *cuda_output_val;

        std::array<float*, 3> cuda_conv_op;
        std::array<float*, 1> cuda_pol_op;
        std::array<float*, 2> cuda_val_op;

        size_t m_scratch_size;
        std::shared_ptr<Model::NNWeights> m_weights{nullptr};
    };

    struct ForwawrdEntry {
	    const std::vector<float> &in_p;
	    const std::vector<float> &in_f;
        std::vector<float> &out_pol;
        std::vector<float> &out_val;

        std::condition_variable cv;
        std::mutex mutex;
        std::atomic<bool> done{false};

        ForwawrdEntry(const std::vector<float> &planes,
                      const std::vector<float> &features,
                      std::vector<float> &output_pol,
                      std::vector<float> &output_val) :
                      in_p(planes), in_f(features), out_pol(output_pol), out_val(output_val) {}
    };

    std::list<std::shared_ptr<ForwawrdEntry>> m_forward_queue;
    std::shared_ptr<Model::NNWeights> m_weights{nullptr};
    std::mutex m_mutex;
    std::mutex m_queue_mutex;
    std::condition_variable m_cv;
    std::atomic<int> m_waittime{0};
    std::atomic<bool> m_narrow_pipe{false};
    std::atomic<bool> m_thread_running;

    std::vector<std::unique_ptr<NNGraph>> m_nngraphs;
    std::vector<std::thread> m_threads;

    void prepare_worker();
    void worker(int gpu);
    void quit_worker();
}; 
#endif
#endif
