#pragma once

#ifdef USE_CUDA

#include <array>
#include <atomic>
#include <condition_variable>
#include <list>
#include <memory>
#include <vector>

#include "neural/activation.h"
#include "neural/batch_forward_pipe.h"
#include "neural/cuda/cuda_common.h"
#include "neural/cuda/cuda_layers.h"
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
    class NNGraph {
        struct Block {
            // TODO: Use list to store all conv.
            cuda::DepthwiseConvolution dw_conv;
            cuda::Convolution pre_btl_conv;
            cuda::Convolution conv1;
            cuda::Convolution conv2;
            cuda::Convolution conv3;
            cuda::Convolution conv4;
            cuda::Convolution post_btl_conv;
            cuda::SEUnit se_module;
        };

        struct Graph {
            // intput
            cuda::Convolution input_conv;

            // block tower
            std::vector<NNGraph::Block> tower;

            // policy head
            cuda::Convolution p_hd_conv;
            cuda::DepthwiseConvolution p_dw_conv;
            cuda::Convolution p_pt_conv;
            cuda::GlobalPooling p_pool;
            cuda::FullyConnect p_inter;

            cuda::Convolution p_prob;
            cuda::FullyConnect p_prob_pass;

            // value head
            cuda::Convolution v_hd_conv;
            cuda::GlobalPooling v_pool;
            cuda::FullyConnect v_inter;

            cuda::Convolution v_ownership;
            cuda::FullyConnect v_misc;
        };

    public:
        NNGraph() = default;
        ~NNGraph();
        void ConstructGraph(bool dump_gpu_info,
                            const int gpu,
                            const int max_batch_size,
                            const int board_size,
                            std::shared_ptr<DNNWeights> weights);

        std::vector<OutputResult> BatchForward(const std::vector<InputData>& input);

        void DestroyGraph();

    private:
        void SetComputationMode(cuda::CudaHandles* handles);

        bool ApplyMask(const std::vector<InputData>& input);

        void FillOutputs(const std::vector<float>& batch_prob,
                         const std::vector<float>& batch_prob_pass,
                         const std::vector<float>& batch_value_misc,
                         const std::vector<float>& batch_ownership,
                         const std::vector<InputData>& batch_input,
                         std::vector<OutputResult>& batch_output_result);

        cuda::CudaHandles handles_;

        int board_size_{0};
        int max_batch_;

        std::unique_ptr<Graph> graph_{nullptr};

        void* host_input_planes_;
        void* host_output_prob_;
        void* host_output_prob_pass_;
        void* host_output_val_;
        void* host_output_ownership_;

        void* cuda_input_planes_;
        void* cuda_output_prob_;
        void* cuda_output_prob_pass_;
        void* cuda_output_val_;
        void* cuda_output_ownership_;

        std::array<void*, 2> host_mask_op_;

        std::array<void*, 2> cuda_scratch_op_;
        std::array<void*, 4> cuda_conv_op_;
        std::array<void*, 3> cuda_pol_op_;
        std::array<void*, 3> cuda_val_op_;
        std::array<void*, 2> cuda_mask_op_;

        size_t scratch_size_;
        std::shared_ptr<DNNWeights> weights_{nullptr};
    };
    std::vector<std::unique_ptr<NNGraph>> nngraphs_;

    bool dump_gpu_info_;
    int max_batch_per_nn_{0};
    int board_size_{0};
};
#endif
