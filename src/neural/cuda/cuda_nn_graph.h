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
    struct Block {
        std::vector<Convolution> convs;
        std::vector<DepthwiseConvolution> dw_convs;
        SEUnit se_module;

        // transformer block
        RMSNorm attn_norm;
        MultiHeadAttention attention;
        RMSNorm ffn_norm;
    };

    struct Graph {
        // intput
        Convolution input_conv;

        // block tower
        std::vector<NNGraph::Block> tower;

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
    void ConstructResidualBlock(Block& block, BlockBasic* tower_ptr, PeakSizes& peaks);

    void ConstructBottleneckBlock(Block& block, BlockBasic* tower_ptr, PeakSizes& peaks);

    void ConstructNestedBottleneckBlock(Block& block, BlockBasic* tower_ptr, PeakSizes& peaks);

    void ConstructMixerBlock(Block& block, BlockBasic* tower_ptr, PeakSizes& peaks);

    void ConstructTransformerBlock(Block& block, BlockBasic* tower_ptr, PeakSizes& peaks);

    void ConstructSEUnit(Block& block, BlockBasic* tower_ptr);

    void
    ResidualBlockForward(const int batch_size, Block& block, BlockBasic* tower_ptr, void* mask);

    void
    BottleneckBlockForward(const int batch_size, Block& block, BlockBasic* tower_ptr, void* mask);

    void NestedBottleneckBlockForward(const int batch_size,
                                      Block& block,
                                      BlockBasic* tower_ptr,
                                      void* mask);

    void MixerBlockForward(const int batch_size, Block& block, BlockBasic* tower_ptr, void* mask);

    void
    TransformerBlockForward(const int batch_size, Block& block, BlockBasic* tower_ptr, void* mask);

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
