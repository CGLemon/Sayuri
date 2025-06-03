#include <stdexcept>
#include "neural/description.h"
#include "utils/format.h"

void LinearLayer::Set(int inputs, int outputs) {
    inputs_ = inputs;
    outputs_ = outputs;
}

void LinearLayer::LoadWeights(std::vector<float> &load_weights) {
    const int weights_size = load_weights.size();
    const int expected_size = GetInputs() * GetOutputs();
    if (weights_size != expected_size) {
        throw std::runtime_error{
            Format("the weights size of linear layer is not acceptable, expect [%d, %d] but we get %d",
                       GetInputs(), GetOutputs(), weights_size)};
    }
    weights_ = load_weights;
}

void LinearLayer::LoadBiases(std::vector<float> &load_weights) {
    const int weights_size = load_weights.size();
    const int expected_size = GetOutputs();
    if (weights_size != expected_size) {
        throw std::runtime_error{
            Format("the biases size of linear layer is not acceptable, expect %d but we get %d",
                       expected_size, weights_size)};
    }
    biases_ = load_weights;
}

int LinearLayer::GetInputs() const {
    return inputs_;
}

int LinearLayer::GetOutputs() const {
    return outputs_;
}

std::vector<float>& LinearLayer::GetWeights() {
    return weights_;
}

std::vector<float>& LinearLayer::GetBiases() {
    return biases_;
}

void BatchNormLayer::Set(int channels) {
    channels_ = channels;
}

void BatchNormLayer::LoadMeans(std::vector<float> &load_weights){
    const int weights_size = load_weights.size();
    const int expected_size = GetChannels();
    if (weights_size != expected_size) {
        throw std::runtime_error{
            Format("the means size of batch normalization layer is not acceptable, expect %d but we get %d",
                       expected_size, weights_size)};
    }
    means_ = load_weights;
}

void BatchNormLayer::LoadStddevs(std::vector<float> &load_weights, bool is_v1){
    const int weights_size = load_weights.size();
    const int expected_size = GetChannels();
    if (weights_size != expected_size) {
        throw std::runtime_error{
            Format("the stddev size of batch normalization layer is not acceptable, expect %d but we get %d",
                       expected_size, weights_size)};
    }
    if (is_v1) {
        ProcessVariance(load_weights); // variance -> 1/stddev
    } else {
        ProcessStddev(load_weights); // stddev -> 1/stddev
    }
    stddevs_ = load_weights;
}

std::vector<float>& BatchNormLayer::GetMeans() {
    return means_;
}

std::vector<float>& BatchNormLayer::GetStddevs() {
    return stddevs_;
}

int BatchNormLayer::GetChannels() const {
    return channels_;
}

void ConvLayer::Set(int inputs, int outputs, int filter) {
    inputs_ = inputs;
    outputs_ = outputs;
    filter_ = filter;
}

void ConvLayer::LoadWeights(std::vector<float> &load_weights) {
    const int weights_size = load_weights.size();
    const int expected_size = GetInputs() * GetOutputs() * GetFilter() * GetFilter();
    if (weights_size != expected_size) {
        throw std::runtime_error{
            Format("the weights size of convolutional layer is not acceptable, expect [%d, %d, %d, %d] but we get %d",
                       GetInputs(), GetOutputs(), GetFilter(), GetFilter(), weights_size)};
    }
    weights_ = load_weights;
}

void ConvLayer::LoadBiases(std::vector<float> &load_weights) {
    const int weights_size = load_weights.size();
    const int expected_size = GetOutputs();
    if (weights_size != expected_size) {
        throw std::runtime_error{
            Format("the biases size of convolutional layer is not acceptable, expect %d but we get %d",
                       expected_size, weights_size)};
    }
    biases_ = load_weights;
}

int ConvLayer::GetInputs() const {
    return inputs_;
}

int ConvLayer::GetOutputs() const {
    return outputs_;
}

int ConvLayer::GetFilter() const {
    return filter_;
}

std::vector<float>& ConvLayer::GetWeights() {
    return weights_;
}

std::vector<float>& ConvLayer::GetBiases() {
    return biases_;
}
