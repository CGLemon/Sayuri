#pragma once

#include <cmath>
#include <vector>

class LinearLayer {
public:
    void Set(int inputs, int outputs);

    void LoadWeights(std::vector<float> &load_weights);
    void LoadBiases(std::vector<float> &load_weights);

    int GetInputs() const;
    int GetOutputs() const;

    const std::vector<float>& GetWeights() const;
    const std::vector<float>& GetBiases() const;

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
    void LoadStddevs(std::vector<float> &load_weights);

    const std::vector<float>& GetMeans() const;
    const std::vector<float>& GetStddevs() const;

    int GetChannels() const;

private:
    template <typename container>
    void ProcessVariant(container &weights) {
        static constexpr float epsilon = 1e-5f;
        for (auto &&w : weights) {
            w = 1.0f / std::sqrt(w + epsilon);
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

    const std::vector<float>& GetWeights() const;
    const std::vector<float>& GetBiases() const;

private:
    std::vector<float> weights_;
    std::vector<float> biases_;

    int inputs_{0};
    int outputs_{0};
    int filter_{0};
};

class ResidualBlock {
public:
    ConvLayer conv1;
    BatchNormLayer bn1;

    ConvLayer conv2;
    BatchNormLayer bn2;

    LinearLayer extend;
    LinearLayer squeeze;

    int se_size{0};
    int apply_se{false};
};


class DNNWeights {
public:
    bool loaded{false};
    bool winograd{false};

    int input_channels{0};

    int residual_blocks{0};
    int residual_channels{0};

    int policy_extract_channels{0};
    int value_extract_channels{0};

    // input layer
    ConvLayer input_conv;
    BatchNormLayer input_bn;

    // residual tower
    std::vector<ResidualBlock> tower;

    // policy head
    ConvLayer p_ex_conv;
    BatchNormLayer p_ex_bn;

    ConvLayer prob_conv;
    LinearLayer pass_fc;

    // value head
    ConvLayer v_ex_conv;
    BatchNormLayer v_ex_bn;

    ConvLayer ownership;
    LinearLayer val_fc;
};
