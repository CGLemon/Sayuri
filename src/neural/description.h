#pragma once

#include <cmath>
#include <vector>
#include <memory>

class LinearLayer {
public:
    void Set(int inputs, int outputs);

    void LoadWeights(std::vector<float> &load_weights);
    void LoadBiases(std::vector<float> &load_weights);

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

    void LoadMeans(std::vector<float> &load_weights);
    void LoadStddevs(std::vector<float> &load_weights, bool is_v1);

    std::vector<float>& GetMeans();
    std::vector<float>& GetStddevs();

    int GetChannels() const;

private:
    template <typename container>
    void ProcessVariance(container &weights) {
        static constexpr float epsilon = 1e-5f;
        for (auto &&w : weights) {
            w = 1.0f / std::sqrt(w + epsilon);
        }
    }
    template <typename container>
    void ProcessStddev(container &weights) {
        for (auto &&w : weights) {
            w = 1.0f / w;
        }
    }

    std::vector<float> means_;
    std::vector<float> stddevs_;

    int channels_{0};
};

class ConvLayer {
public:
    void Set(int inputs, int outputs, int filter);

    void LoadWeights(std::vector<float> &load_weights);
    void LoadBiases(std::vector<float> &load_weights);

    int GetInputs() const;
    int GetOutputs() const;
    int GetFilter() const;

    std::vector<float>& GetWeights();
    std::vector<float>& GetBiases();

private:
    std::vector<float> weights_;
    std::vector<float> biases_;

    int inputs_{0};
    int outputs_{0};
    int filter_{0};
};

class BlockBasic {
public:
    enum Type {
        kUnknown,
        kResidualBlock,
        kBottleneckBlock,
        kMixerBlock
    };
    BlockBasic() = default;

    bool IsResidualBlock() { return type == Type::kResidualBlock; }
    bool IsBottleneckBlock() { return type == Type::kBottleneckBlock; }
    bool IsMixerBlock() { return type == Type::kMixerBlock; }

    Type type{kUnknown};

    ConvLayer conv1;
    BatchNormLayer bn1;
    ConvLayer conv2;
    BatchNormLayer bn2;

    ConvLayer pre_btl_conv;
    BatchNormLayer pre_btl_bn;
    ConvLayer post_btl_conv;
    BatchNormLayer post_btl_bn;
    int bottleneck_channels{0};

    ConvLayer dw_conv;
    BatchNormLayer dw_bn;
    int feedforward_channels{0};

    LinearLayer squeeze;
    LinearLayer excite;
    int se_size{0};
    bool apply_se{false};
};

class ResidualBlock : public BlockBasic {
public:
    ResidualBlock() { type = Type::kResidualBlock; }
};

class BottleneckBlock : public BlockBasic {
public:
    BottleneckBlock() { type = Type::kBottleneckBlock; }
};

class MixerBlock : public BlockBasic {
public:
    MixerBlock() { type = Type::kMixerBlock; }
};

class DNNWeights {
public:
    bool loaded{false};
    bool winograd{false};
    bool winograd_initialized{false};

    int input_channels{0};

    int residual_blocks{0};
    int residual_channels{0};

    int policy_extract_channels{0};
    int value_extract_channels{0};

    // input layer
    ConvLayer input_conv;
    BatchNormLayer input_bn;

    // block tower
    std::vector<std::unique_ptr<BlockBasic>> tower;

    // policy head
    ConvLayer p_ex_conv;
    BatchNormLayer p_ex_bn;
    LinearLayer p_inter_fc;

    ConvLayer prob_conv;
    LinearLayer pass_fc;

    // value head
    ConvLayer v_ex_conv;
    BatchNormLayer v_ex_bn;
    LinearLayer v_inter_fc;

    ConvLayer v_ownership;
    LinearLayer v_misc;
};
