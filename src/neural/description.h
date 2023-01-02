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

class ResidualBlock {
public:
    ConvLayer conv1;
    BatchNormLayer bn1;

    ConvLayer conv2;
    BatchNormLayer bn2;

    LinearLayer squeeze;
    LinearLayer excite;

    int se_size{0};
    int apply_se{false};
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

    // residual tower
    std::vector<ResidualBlock> tower;

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
