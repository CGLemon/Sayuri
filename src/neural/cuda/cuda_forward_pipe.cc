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
#include "cuda/CUDACommon.h"

#include "config.h"
#include "Utils.h"
#include "Model.h"
#include "Board.h"

#include <iterator>
#include <chrono>

void CUDABackend::initialize(std::shared_ptr<Model::NNWeights> weights) {
    if (option<bool>("debug_verbose")) {
        CUDA::check_devices();
    }
    reload(weights);
    prepare_worker();
    DEBUG << "Create CUDA Network" << std::endl;
}

void CUDABackend::destroy() {
    release();
    quit_worker();
    DEBUG << "Release CUDA Network" << std::endl;
}

void CUDABackend::reload(std::shared_ptr<Model::NNWeights> weights) {
    release();
    m_weights = weights;
    const auto d_cnt = CUDA::get_devicecount();
    if (option<int>("gpu") >= 0) {
        m_nngraphs.emplace_back(std::make_unique<NNGraph>());
        const auto gpu = option<int>("gpu") >= d_cnt ? 0 : option<int>("gpu");
        m_nngraphs[0]->build_graph(gpu, weights);
    } else {
        for (int i = 0; i < d_cnt; ++i) {
            m_nngraphs.emplace_back(std::make_unique<NNGraph>());
        }
        for (int i = 0; i < d_cnt; ++i) {
            m_nngraphs[i]->build_graph(i, weights);
        }
    }
}

void CUDABackend::release() {
    for (auto &g : m_nngraphs) {
        g->destroy_graph();
    }
    m_nngraphs.clear();
}

bool CUDABackend::valid() {
    return m_weights->loaded;
}

void CUDABackend::forward(const std::vector<float> &planes,
                          const std::vector<float> &features,
                          std::vector<float> &output_pol,
                          std::vector<float> &output_val) {
    auto entry = std::make_shared<ForwawrdEntry>(planes,
                                                 features,
                                                 output_pol,
                                                 output_val);
    std::unique_lock<std::mutex> lock(entry->mutex);
    {
        std::unique_lock<std::mutex> queue_lock(m_queue_mutex);
        m_forward_queue.emplace_back(entry);
    }

    if (m_forward_queue.size() >= (size_t)option<int>("batchsize")) {
        m_cv.notify_one();
    }

    entry->cv.wait(lock);
    entry->done.store(true);
}


void CUDABackend::NNGraph::build_graph(const int gpu, std::shared_ptr<Model::NNWeights> weights) {
    if (m_graph != nullptr) {
        return;
    }
    m_graph = std::make_unique<Graph>();
    m_gpu = gpu;
    auto d = CUDA::get_device(m_gpu);

    cudaSetDevice(d);

    m_weights = weights;
    m_handel.apply(m_gpu);    
    m_scratch_size = 0;
    m_maxbatch = option<int>("batchsize");

    const auto output_channels = m_weights->residual_channels;

    // build graph
    // input layer
    m_graph->input_conv = CUDA::Convolve(
        m_maxbatch,          // max batch size
        3,                   // kernel size
        INPUT_CHANNELS,      // input channels
        output_channels      // output channels
    );
    m_graph->input_bnorm = CUDA::Batchnorm(
        m_maxbatch,          // max batch size
        output_channels,     // channels
        false                // relu
    );
    m_graph->input_pool = CUDA::InputPool(
        m_maxbatch,          // max batch size
        INPUT_FEATURES,      // input size
        2 * output_channels, // squeeze size
        output_channels      // channels
    );

    // residual tower
    const auto residuals = m_weights->residual_blocks;
    for (int i = 0; i < residuals; ++i) {
        m_graph->tower_conv.emplace_back(CUDA::Convolve{});
        m_graph->tower_bnorm.emplace_back(CUDA::Batchnorm{});
        m_graph->tower_conv.emplace_back(CUDA::Convolve{});
        m_graph->tower_bnorm.emplace_back(CUDA::Batchnorm{});
        m_graph->tower_se.emplace_back(CUDA::SEUnit{});
    }

    for (int i = 0; i < residuals; ++i) {
        const auto t_offset = 2 * i;
        const auto tower_channels = m_weights->residual_channels;
        const auto tower_ptr = m_weights->residual_tower.data() + i;
    
        m_graph->tower_conv[t_offset+0] = CUDA::Convolve(
            m_maxbatch,          // max batch size
            3,                   // kernel size
            tower_channels,      // input channels
            tower_channels       // output channels
        );
        m_graph->tower_bnorm[t_offset+0] = CUDA::Batchnorm(
            m_maxbatch,          // max batch size
            tower_channels       // channels
        );

        m_graph->tower_conv[t_offset+1] = CUDA::Convolve(
            m_maxbatch,          // max batch size
            3,                   // kernel size
            tower_channels,      // input channels
            tower_channels       // output channels
        );
        m_graph->tower_bnorm[t_offset+1] = CUDA::Batchnorm(
            m_maxbatch,          // max batch size
            tower_channels,      // channels
            !tower_ptr->apply_se // relu
        );

        if (tower_ptr->apply_se) {
            const size_t se_size = tower_ptr->se_size;
            m_graph->tower_se[i] = CUDA::SEUnit(
                m_maxbatch,      // max batch size
                tower_channels,  // channels
                se_size          // SE size
            );
        }
    }

    // policy head
    const auto policy_extract_channels = m_weights->policy_extract_channels;
    m_graph->p_ex_conv = CUDA::Convolve(
        m_maxbatch,             // max batch size
        3,                      // kernel size
        output_channels,        // input channels
        policy_extract_channels // output channels
    );
    m_graph->p_ex_bnorm = CUDA::Batchnorm(
        m_maxbatch,             // max batch size
        policy_extract_channels // channels
    );
    m_graph->p_map = CUDA::Convolve(
        m_maxbatch,              // max batch size
        3,                       // kernel size
        policy_extract_channels, // input channels
        POLICYMAP                // output channels
    );

    // value head
    const auto value_extract_channels = m_weights->value_extract_channels;
    m_graph->v_ex_conv = CUDA::Convolve(
        m_maxbatch,             // max batch size
        1,                      // kernel size
        output_channels,        // input channels
        value_extract_channels  // output channels
    );
    m_graph->v_ex_bnorm = CUDA::Batchnorm(
        m_maxbatch,             // max batch size
        value_extract_channels  // channels
    );
    m_graph->v_fc1 = CUDA::FullyConnect(
        m_maxbatch,                                     // max batch size
        value_extract_channels * Board::INTERSECTIONS,  // input size
        VALUELAYER,                                     // output size
        true                                            // relu
    );
    m_graph->v_fc2 = CUDA::FullyConnect(
        m_maxbatch,     // max batch size
        VALUELAYER,     // input size
        WINRATELAYER,   // output size
        false           // relu
    );

    // fill the parameters
    // input layer
    m_graph->input_conv.LoadingWeight(m_weights->input_conv.weights, m_scratch_size, &m_handel);
    m_graph->input_bnorm.LoadingWeight(m_weights->input_bn.means, m_weights->input_bn.stddevs);
    m_graph->input_pool.LoadingWeight(m_weights->input_fc1.weights, m_weights->input_fc1.biases,
                                      m_weights->input_fc2.weights, m_weights->input_fc2.biases);
    // residual tower
    for (int i = 0; i < residuals; ++i) {
        const auto t_offset = 2 * i;
        const auto tower_ptr = m_weights->residual_tower.data() + i;

        m_graph->tower_conv[t_offset+0].LoadingWeight(tower_ptr->conv_1.weights, m_scratch_size, &m_handel);
        m_graph->tower_bnorm[t_offset+0].LoadingWeight(tower_ptr->bn_1.means, tower_ptr->bn_1.stddevs);
        m_graph->tower_conv[t_offset+1].LoadingWeight(tower_ptr->conv_2.weights, m_scratch_size, &m_handel);
        m_graph->tower_bnorm[t_offset+1].LoadingWeight(tower_ptr->bn_2.means, tower_ptr->bn_2.stddevs);

        if (tower_ptr->apply_se) {
            m_graph->tower_se[i].LoadingWeight(tower_ptr->extend.weights,
                                               tower_ptr->extend.biases,
                                               tower_ptr->squeeze.weights,
                                               tower_ptr->squeeze.biases);
        }
    }

    // policy head
    m_graph->p_ex_conv.LoadingWeight(weights->p_ex_conv.weights, m_scratch_size, &m_handel);
    m_graph->p_ex_bnorm.LoadingWeight(weights->p_ex_bn.means, m_weights->p_ex_bn.stddevs);
    m_graph->p_map.LoadingWeight(weights->p_map.weights, m_weights->p_map.biases, m_scratch_size, &m_handel);

    // value head
    m_graph->v_ex_conv.LoadingWeight(weights->v_ex_conv.weights, m_scratch_size, &m_handel);
    m_graph->v_ex_bnorm.LoadingWeight(weights->v_ex_bn.means, m_weights->v_ex_bn.stddevs);
    m_graph->v_fc1.LoadingWeight(weights->v_fc1.weights, m_weights->v_fc1.biases);
    m_graph->v_fc2.LoadingWeight(weights->v_fc2.weights, m_weights->v_fc2.biases);

    const size_t factor = m_maxbatch * sizeof(float);
    const size_t planes_size = factor * INPUT_CHANNELS * Board::INTERSECTIONS;
    const size_t features_size = factor * INPUT_FEATURES;
    const size_t pol_size = factor * POLICYMAP * Board::INTERSECTIONS;
    const size_t val_size = factor * WINRATELAYER;

    const size_t conv_op_size = factor * m_weights->residual_channels * Board::INTERSECTIONS;

    const size_t pol_op1_size = factor * policy_extract_channels * Board::INTERSECTIONS;
    const size_t val_op1_size = factor * value_extract_channels * Board::INTERSECTIONS;
    const size_t val_op2_size = factor * VALUELAYER;

    CUDA::ReportCUDAErrors(cudaMalloc(&cuda_scratch, m_scratch_size));
    CUDA::ReportCUDAErrors(cudaMalloc(&cuda_input_planes, planes_size));
    CUDA::ReportCUDAErrors(cudaMalloc(&cuda_input_features, features_size));
    CUDA::ReportCUDAErrors(cudaMalloc(&cuda_output_pol, pol_size));
    CUDA::ReportCUDAErrors(cudaMalloc(&cuda_output_val, val_size));

    CUDA::ReportCUDAErrors(cudaMalloc(&cuda_conv_op[0], conv_op_size));
    CUDA::ReportCUDAErrors(cudaMalloc(&cuda_conv_op[1], conv_op_size));
    CUDA::ReportCUDAErrors(cudaMalloc(&cuda_conv_op[2], conv_op_size));

    CUDA::ReportCUDAErrors(cudaMalloc(&cuda_pol_op[0], pol_op1_size));
    CUDA::ReportCUDAErrors(cudaMalloc(&cuda_val_op[0], val_op1_size));
    CUDA::ReportCUDAErrors(cudaMalloc(&cuda_val_op[1], val_op2_size));
}


void CUDABackend::NNGraph::batch_forward(const int batch_size,
                                         std::vector<float> &planes,
                                         std::vector<float> &features,
                                         std::vector<float> &output_pol,
                                         std::vector<float> &output_val) {

    assert(m_maxbatch >= batch_size);

    const size_t factor = batch_size * sizeof(float);
    const size_t planes_s = factor * INPUT_CHANNELS * Board::INTERSECTIONS;
    const size_t features_s = factor * INPUT_FEATURES;

    CUDA::ReportCUDAErrors(cudaMemcpy(cuda_input_planes, planes.data(),
                                      planes_s, cudaMemcpyHostToDevice));
    CUDA::ReportCUDAErrors(cudaMemcpy(cuda_input_features, features.data(),
                                      features_s, cudaMemcpyHostToDevice));

    m_graph->input_conv.Forward(batch_size,
                                cuda_input_planes, cuda_conv_op[0],
                                cuda_scratch, m_scratch_size, &m_handel);
    m_graph->input_bnorm.Forward(batch_size,
                                 cuda_conv_op[0]);
    m_graph->input_pool.Forward(batch_size, cuda_input_features,
                                cuda_conv_op[0], &m_handel);

    // Residual tower
    const auto residuals = m_weights->residual_blocks;
    for (int i = 0; i < residuals; ++i) {
        // if (i == 1) break;
        const auto t_offset = 2 * i;
        const auto tower_ptr = m_weights->residual_tower.data() + i;

        m_graph->tower_conv[t_offset+0].Forward(batch_size,
                                                cuda_conv_op[0], cuda_conv_op[1],
                                                cuda_scratch, m_scratch_size, &m_handel);
        m_graph->tower_bnorm[t_offset+0].Forward(batch_size,
                                                 cuda_conv_op[1]);

        m_graph->tower_conv[t_offset+1].Forward(batch_size,
                                                cuda_conv_op[1], cuda_conv_op[2],
                                                cuda_scratch, m_scratch_size, &m_handel);
        if (tower_ptr->apply_se) {
            m_graph->tower_bnorm[t_offset+1].Forward(batch_size,
                                                     cuda_conv_op[2]);
            m_graph->tower_se[i].Forward(batch_size,
                                         cuda_conv_op[2], cuda_conv_op[0], &m_handel);
        } else { 
            const auto tower_channels = m_weights->residual_channels;
            m_graph->tower_bnorm[t_offset+1].Forward(batch_size,
                                                     cuda_conv_op[2], cuda_conv_op[0]);
            CUDA::copy(cuda_conv_op[0], cuda_conv_op[2], batch_size * tower_channels * Board::INTERSECTIONS);
        }
    }

    // policy head
    m_graph->p_ex_conv.Forward(batch_size,
                               cuda_conv_op[0], cuda_pol_op[0],
                               cuda_scratch, m_scratch_size, &m_handel);
    m_graph->p_ex_bnorm.Forward(batch_size, cuda_pol_op[0]);
    m_graph->p_map.Forward(batch_size,
                           cuda_pol_op[0], cuda_output_pol,
                           cuda_scratch, m_scratch_size, &m_handel);

    // value head
    m_graph->v_ex_conv.Forward(batch_size,
                               cuda_conv_op[0], cuda_val_op[0],
                               cuda_scratch, m_scratch_size, &m_handel);
    m_graph->v_ex_bnorm.Forward(batch_size, cuda_val_op[0]);
    m_graph->v_fc1.Forward(batch_size, cuda_val_op[0], cuda_val_op[1], &m_handel);
    m_graph->v_fc2.Forward(batch_size, cuda_val_op[1], cuda_output_val, &m_handel);

    const auto pol_size = static_cast<size_t>(factor * POLICYMAP * Board::INTERSECTIONS);
    const auto val_size = static_cast<size_t>(factor * WINRATELAYER);

    CUDA::ReportCUDAErrors(cudaMemcpy(output_pol.data(), cuda_output_pol,
                                      pol_size, cudaMemcpyDeviceToHost));
 
    CUDA::ReportCUDAErrors(cudaMemcpy(output_val.data(), cuda_output_val,
                                      val_size, cudaMemcpyDeviceToHost));
}

void CUDABackend::NNGraph::destroy_graph() {
    if (m_graph == nullptr) {
        return;
    }

    CUDA::ReportCUDAErrors(cudaFree(cuda_scratch));

    CUDA::ReportCUDAErrors(cudaFree(cuda_input_planes));
    CUDA::ReportCUDAErrors(cudaFree(cuda_output_pol));
    CUDA::ReportCUDAErrors(cudaFree(cuda_output_val));

    CUDA::ReportCUDAErrors(cudaFree(cuda_conv_op[0]));
    CUDA::ReportCUDAErrors(cudaFree(cuda_conv_op[1]));
    CUDA::ReportCUDAErrors(cudaFree(cuda_conv_op[2]));

    CUDA::ReportCUDAErrors(cudaFree(cuda_pol_op[0]));
    CUDA::ReportCUDAErrors(cudaFree(cuda_val_op[0]));
    CUDA::ReportCUDAErrors(cudaFree(cuda_val_op[1]));

    m_graph.reset();
    m_graph = nullptr;
}

CUDABackend::NNGraph::~NNGraph() {
    destroy_graph();
}

void CUDABackend::worker(int gpu) {
    const auto gether_batches = [this](){
        const size_t maxbatch = (size_t)option<int>("batchsize");
        const auto gpu_waittime = option<int>("gpu_waittime");
        std::list<std::shared_ptr<ForwawrdEntry>> inputs;
        while(true) {
            if (!m_thread_running) {
                return inputs;
            }

            int waittime = m_waittime.load();
            if (m_forward_queue.size() >= maxbatch) {
                if (waittime > gpu_waittime) {
                    m_waittime.store(gpu_waittime);
                } else if (waittime > 1) {
                    waittime--;
                    m_waittime.store(waittime);
                }
                break;
            }

            std::unique_lock<std::mutex> lock(m_mutex);
            bool timeout = !m_cv.wait_for(lock, std::chrono::milliseconds(waittime),
                                              [maxbatch, this](){ return !(m_forward_queue.size() < maxbatch); }
                                          );
            if (!m_forward_queue.empty()) {
                if (timeout && m_narrow_pipe.exchange(true) == false) {
                    if (waittime > 1) {
                        waittime--;
                        m_waittime.store(waittime);
                    }
                    break;
                }
            } else {
                if (waittime < 20 * gpu_waittime) {
                   waittime += 2;
                }
                m_waittime.store(waittime);
            }
        }

        std::unique_lock<std::mutex> queue_lock(m_queue_mutex);
        auto count = m_forward_queue.size();
        if (count > maxbatch) {
            count = maxbatch;
        }

        auto end = std::begin(m_forward_queue);
        std::advance(end, count);
        std::move(std::begin(m_forward_queue), end, std::back_inserter(inputs));
        m_forward_queue.erase(std::begin(m_forward_queue), end);

        return inputs;
    };

    while (true) {
        if (!m_thread_running) return;

        auto gather_entry = gether_batches();
        const auto batch_size = gather_entry.size();

        if (batch_size == 0) {
            continue;
        }

        const auto first = *std::begin(gather_entry);

        const auto in_p_size = first->in_p.size();
        const auto in_f_size = first->in_f.size();
        const auto out_pol_size = first->out_pol.size();
        const auto out_val_size = first->out_val.size();

        auto batch_input_planes = std::vector<float>(batch_size * in_p_size);
        auto batch_input_features = std::vector<float>(batch_size * in_f_size);
        auto batch_out_pol = std::vector<float>(batch_size * out_pol_size);
        auto batch_out_val = std::vector<float>(batch_size * out_val_size);

        auto index = size_t{0};
        for (auto &x : gather_entry) {
            std::copy(std::begin(x->in_p),
                      std::end(x->in_p),
                      std::begin(batch_input_planes) + index * in_p_size);
            std::copy(std::begin(x->in_f),
                      std::end(x->in_f),
                      std::begin(batch_input_features) + index * in_f_size);
            index++;
        }

        m_nngraphs[gpu]->batch_forward(batch_size,
                                       batch_input_planes,
                                       batch_input_features,
                                       batch_out_pol,
                                       batch_out_val);

        index = 0;
        for (auto &x : gather_entry) {
            std::copy(std::begin(batch_out_pol) + index * out_pol_size,
                      std::begin(batch_out_pol) + (index+1) * out_pol_size,
                      std::begin(x->out_pol));
            std::copy(std::begin(batch_out_val) + index * out_val_size,
                      std::begin(batch_out_val) + (index+1) * out_val_size,
                      std::begin(x->out_val));
            while (!x->done.load()) {
                x->cv.notify_all();
            }
            index++;
        }

        if (batch_size <= (size_t)option<int>("batchsize")) {
            m_narrow_pipe.store(false);
        }
    }
}

void CUDABackend::prepare_worker() {
    m_thread_running = true;
    if (m_threads.size() == 0) {
        for (int gpu = 0; gpu < (int)m_nngraphs.size(); ++gpu) {
            m_threads.emplace_back([g=gpu, this](){ worker(g); });
        }
    }
}

void CUDABackend::quit_worker() {
    {
        std::unique_lock<std::mutex> queue_lock(m_queue_mutex);
        m_thread_running = false;
    }
    m_cv.notify_all();
    for (auto &t : m_threads) {
        t.join();
    }
}
#endif           
