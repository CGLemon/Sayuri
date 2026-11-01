#pragma once

#ifdef USE_CUDA

#include <array>
#include <memory>
#include <vector>

#include "neural/activation.h"
#include "neural/cuda/cuda_common.h"
#include "neural/cuda/cuda_layers.h"
#include "neural/description.h"
#include "neural/network_basic.h"

namespace cuda {

class NNGraph {
    // The layers of a block on the GPU. Each block type declares only its own
    // layers, and the SE module is the only exception kept on the base.
    struct BlockLayers {
        virtual ~BlockLayers() = default;

        SEUnit se_module;
    };

    struct ResidualLayers : BlockLayers {
        Convolution conv1;
        Convolution conv2;
    };

    struct BottleneckLayers : BlockLayers {
        Convolution pre_btl_conv;
        Convolution conv1;
        Convolution conv2;
        Convolution post_btl_conv;
    };

    struct NestedBottleneckLayers : BlockLayers {
        Convolution pre_btl_conv;
        Convolution conv1;
        Convolution conv2;
        Convolution conv3;
        Convolution conv4;
        Convolution post_btl_conv;
    };

    struct MixerLayers : BlockLayers {
        DepthwiseConvolution dw_conv;
        Convolution conv1;
        Convolution conv2;
    };

    struct TransformerLayers : BlockLayers {
        RMSNorm attn_norm;
        Convolution q_proj;
        Convolution k_proj;
        Convolution v_proj;
        MultiHeadAttention attention;
        Convolution out_proj;
        RMSNorm ffn_norm;
        Convolution ffn_fc1;
        Convolution ffn_gate;
        Convolution ffn_fc2;
    };

    struct Graph {
        // intput
        Convolution input_conv;

        // block tower
        std::vector<std::unique_ptr<BlockLayers>> tower;

        // policy head
        Convolution p_hd_conv;
        DepthwiseConvolution p_dw_conv;
        Convolution p_pt_conv;
        GlobalPooling p_pool;
        FullyConnect p_inter;

        Convolution p_prob;
        FullyConnect p_prob_pass;

        // value head
        Convolution v_hd_conv;
        GlobalPooling v_pool;
        FullyConnect v_inter;

        Convolution v_ownership;
        FullyConnect v_misc;
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
    struct PeakSizes {
        int channels{0};
        int attn_channels{0};
        int attn_heads{0};
    };

    void SetComputationMode(CudaHandles* handles);

    // Build the layers of each block type and load their weights.
    std::unique_ptr<ResidualLayers> ConstructResidualBlock(ResidualBlock* block, PeakSizes& peaks);

    std::unique_ptr<BottleneckLayers> ConstructBottleneckBlock(BottleneckBlock* block,
                                                               PeakSizes& peaks);

    std::unique_ptr<NestedBottleneckLayers>
    ConstructNestedBottleneckBlock(NestedBottleneckBlock* block, PeakSizes& peaks);

    std::unique_ptr<MixerLayers> ConstructMixerBlock(MixerBlock* block, PeakSizes& peaks);

    std::unique_ptr<TransformerLayers> ConstructTransformerBlock(TransformerBlock* block,
                                                                 PeakSizes& peaks);

    void ConstructSEUnit(BlockLayers& layers, BlockBasic* tower_ptr);

    void ResidualBlockForward(const int batch_size,
                              ResidualLayers& layers,
                              ResidualBlock* block,
                              void* mask);

    void BottleneckBlockForward(const int batch_size,
                                BottleneckLayers& layers,
                                BottleneckBlock* block,
                                void* mask);

    void NestedBottleneckBlockForward(const int batch_size,
                                      NestedBottleneckLayers& layers,
                                      NestedBottleneckBlock* block,
                                      void* mask);

    void
    MixerBlockForward(const int batch_size, MixerLayers& layers, MixerBlock* block, void* mask);

    void TransformerBlockForward(const int batch_size,
                                 TransformerLayers& layers,
                                 TransformerBlock* block,
                                 void* mask);

    bool ApplyMask(const std::vector<InputData>& input);

    void FillOutputs(const std::vector<float>& batch_prob,
                     const std::vector<float>& batch_prob_pass,
                     const std::vector<float>& batch_value_misc,
                     const std::vector<float>& batch_ownership,
                     const std::vector<InputData>& batch_input,
                     std::vector<OutputResult>& batch_output_result);

    CudaHandles handles_;

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
    std::array<void*, 2> cuda_attn_op_;

    size_t scratch_size_;
    std::shared_ptr<DNNWeights> weights_{nullptr};
};

} // namespace cuda

#endif
