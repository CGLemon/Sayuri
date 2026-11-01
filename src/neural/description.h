#pragma once

#include <cmath>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "neural/activation.h"

class LinearLayer {
public:
    void Set(int inputs, int outputs);

    void LoadWeights(std::vector<float>& load_weights);
    void LoadBiases(std::vector<float>& load_weights);

    int GetInputs() const;
    int GetOutputs() const;

    std::vector<float>& GetWeights();
    std::vector<float>& GetBiases();

private:
    std::vector<float> weights_;
    std::vector<float> biases_;

    int inputs_{0};
    int outputs_{0};
};

class BatchNormLayer {
public:
    void Set(int channels);

    void LoadMeans(std::vector<float>& load_weights);
    void LoadStddevs(std::vector<float>& load_weights, bool is_v1);

    std::vector<float>& GetMeans();
    std::vector<float>& GetStddevs();

    int GetChannels() const;

private:
    template <typename container> void ProcessVariance(container& weights) {
        static constexpr float epsilon = 1e-5f;
        for (auto&& w : weights) {
            w = 1.0f / std::sqrt(w + epsilon);
        }
    }
    template <typename container> void ProcessStddev(container& weights) {
        for (auto&& w : weights) {
            w = 1.0f / w;
        }
    }

    std::vector<float> means_;
    std::vector<float> stddevs_;

    int channels_{0};
};

class RMSNormLayer {
public:
    void Set(int channels);

    void LoadGammas(std::vector<float>& load_weights);

    int GetChannels() const;

    std::vector<float>& GetGammas();

private:
    std::vector<float> gammas_;

    int channels_{0};
};

class RoPELayer {
public:
    void Set(int head_dim, int num_heads);

    void LoadFrequencies(std::vector<float>& load_weights);

    int GetHeadDim() const;
    int GetNumHeads() const;

    std::vector<float>& GetFrequencies();

private:
    std::vector<float> frequencies_;

    int head_dim_{0};
    int num_heads_{0};
};

class ConvLayer {
public:
    void Set(int inputs, int outputs, int filter);

    void LoadWeights(std::vector<float>& load_weights);
    void LoadBiases(std::vector<float>& load_weights);
    void TransformF();

    int GetInputs() const;
    int GetOutputs() const;
    int GetFilter() const;

    std::vector<float>& GetWeights();
    std::vector<float>& GetTransformF();
    std::vector<float>& GetBiases();

private:
    std::vector<float> weights_;
    std::vector<float> transformed_f_;
    std::vector<float> biases_;

    int inputs_{0};
    int outputs_{0};
    int filter_{0};
};

class BlockBasic {
public:
    enum class Type : int {
        kUnknown,
        kResidualBlock,
        kBottleneckBlock,
        kNestedBottleneckBlock,
        kMixerBlock,
        kTransformerBlock
    };

    virtual ~BlockBasic() = default;

    Type GetType() const {
        return type_;
    }

    template <typename BlockType> BlockType* Cast() {
        static_assert(std::is_base_of<BlockBasic, BlockType>::value,
                      "the cast target should be a block type");
        if (type_ != BlockType::kBlockType) {
            return nullptr;
        }
        return static_cast<BlockType*>(this);
    }

    template <typename BlockType> const BlockType* Cast() const {
        static_assert(std::is_base_of<BlockBasic, BlockType>::value,
                      "the cast target should be a block type");
        if (type_ != BlockType::kBlockType) {
            return nullptr;
        }
        return static_cast<const BlockType*>(this);
    }

    bool IsResidualBlock() const {
        return type_ == Type::kResidualBlock;
    }
    bool IsBottleneckBlock() const {
        return type_ == Type::kBottleneckBlock;
    }
    bool IsNestedBottleneckBlock() const {
        return type_ == Type::kNestedBottleneckBlock;
    }
    bool IsMixerBlock() const {
        return type_ == Type::kMixerBlock;
    }
    bool IsTransformerBlock() const {
        return type_ == Type::kTransformerBlock;
    }

    LinearLayer squeeze;
    LinearLayer excite;
    int se_size{0};
    bool apply_se{false};

protected:
    explicit BlockBasic(Type type) : type_(type) {}

private:
    Type type_{Type::kUnknown};
};

class ResidualBlock : public BlockBasic {
public:
    static constexpr Type kBlockType = Type::kResidualBlock;

    ResidualBlock() : BlockBasic(kBlockType) {}

    ConvLayer conv1;
    BatchNormLayer bn1;
    ConvLayer conv2;
    BatchNormLayer bn2;
};

class BottleneckBlock : public BlockBasic {
public:
    static constexpr Type kBlockType = Type::kBottleneckBlock;

    BottleneckBlock() : BlockBasic(kBlockType) {}

    ConvLayer pre_btl_conv;
    BatchNormLayer pre_btl_bn;
    ConvLayer conv1;
    BatchNormLayer bn1;
    ConvLayer conv2;
    BatchNormLayer bn2;
    ConvLayer post_btl_conv;
    BatchNormLayer post_btl_bn;
    int bottleneck_channels{0};
};

class NestedBottleneckBlock : public BlockBasic {
public:
    static constexpr Type kBlockType = Type::kNestedBottleneckBlock;

    NestedBottleneckBlock() : BlockBasic(kBlockType) {}

    ConvLayer pre_btl_conv;
    BatchNormLayer pre_btl_bn;
    ConvLayer conv1;
    BatchNormLayer bn1;
    ConvLayer conv2;
    BatchNormLayer bn2;
    ConvLayer conv3;
    BatchNormLayer bn3;
    ConvLayer conv4;
    BatchNormLayer bn4;
    ConvLayer post_btl_conv;
    BatchNormLayer post_btl_bn;
    int bottleneck_channels{0};
};

class MixerBlock : public BlockBasic {
public:
    static constexpr Type kBlockType = Type::kMixerBlock;

    MixerBlock() : BlockBasic(kBlockType) {}

    // The depthwise convolution (token mixing).
    ConvLayer dw_conv;
    BatchNormLayer dw_bn;

    // The pointwise feedforward (channel mixing).
    ConvLayer conv1;
    BatchNormLayer bn1;
    ConvLayer conv2;
    BatchNormLayer bn2;
    int feedforward_channels{0};
};

class TransformerBlock : public BlockBasic {
public:
    static constexpr Type kBlockType = Type::kTransformerBlock;

    TransformerBlock() : BlockBasic(kBlockType) {}

    // The self-attention sublayer.
    RMSNormLayer attn_norm;
    LinearLayer q_proj;
    LinearLayer k_proj;
    LinearLayer v_proj;
    LinearLayer out_proj;
    RoPELayer rope;
    int num_heads{0};
    int head_dim{0};

    // The gated feedforward sublayer.
    RMSNormLayer ffn_norm;
    LinearLayer ffn_fc1;
    LinearLayer ffn_gate;
    LinearLayer ffn_fc2;
    int feedforward_channels{0};
};

enum class PolicyHeadType { kNormal, kRepLK };

class DNNWeights {
public:
    std::string name;
    std::string path;
    int version{-1};

    bool loaded{false};
    bool winograd{false};

    int input_channels{0};

    int residual_blocks{0};
    int residual_channels{0};

    PolicyHeadType policy_head_type{PolicyHeadType::kNormal};
    int policy_head_channels{0};
    int probabilities_channels{0};
    int pass_probability_outputs{0};

    int value_head_channels{0};
    int ownership_channels{0};
    int value_misc_outputs{0};

    Activation default_act;

    // input layer
    ConvLayer input_conv;
    BatchNormLayer input_bn;

    // block tower
    std::vector<std::unique_ptr<BlockBasic>> tower;

    // policy head
    ConvLayer p_hd_conv;
    BatchNormLayer p_hd_bn;
    LinearLayer p_inter_fc;

    ConvLayer p_dw_conv;
    BatchNormLayer p_dw_bn;
    ConvLayer p_pt_conv;
    BatchNormLayer p_pt_bn;

    ConvLayer prob_conv;
    LinearLayer pass_fc;

    // value head
    ConvLayer v_hd_conv;
    BatchNormLayer v_hd_bn;
    LinearLayer v_inter_fc;

    ConvLayer v_ownership;
    LinearLayer v_misc;
};
