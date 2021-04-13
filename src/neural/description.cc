#include "neural/description.h"

void LinearLayer::Set(int inputs, int outputs) {
    inputs_ = inputs;
    outputs_ = outputs;
}

void LinearLayer::LoadWeights(std::vector<float> &load_weights) {
    if ((int)load_weights.size() != GetInputs() * GetOutputs()) {
        throw "The linear layer is not acceptable.";
    }
    weights_ = std::move(load_weights);
}

void LinearLayer::LoadBiases(std::vector<float> &load_weights) {
    if ((int)load_weights.size() != GetOutputs()) {
        throw "The linear layer is not acceptable.";
    }
    biases_ = std::move(load_weights);
}

int LinearLayer::GetInputs() const {
    return inputs_;
}

int LinearLayer::GetOutputs() const {
    return outputs_;
}

const std::vector<float>& LinearLayer::GetWeights() const {
    return weights_;
}

const std::vector<float>& LinearLayer::GetBiases() const {
    return biases_;
}

void BatchNormLayer::Set(int channels) {
    channels_ = channels;
}

void BatchNormLayer::LoadMeans(std::vector<float> &load_weights){
    if ((int)load_weights.size() != GetChannels()) {
        throw "The Batch Normal layer is not acceptable.";
    }
    means_ = std::move(load_weights);
}

void BatchNormLayer::LoadStddevs(std::vector<float> &load_weights){
    if ((int)load_weights.size() != GetChannels()) {
        throw "The linear layer is not acceptable.";
    }
    ProcessVariant(load_weights);
    stddevs_ = std::move(load_weights);
}

const std::vector<float>& BatchNormLayer::GetMeans() const {
    return means_;
}

const std::vector<float>& BatchNormLayer::GetStddevs() const {
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
    if ((int)load_weights.size() != GetInputs() * GetOutputs() * GetFilter() * GetFilter()) {
        throw "The one of Convolutional Layers weights size is not acceptable.";
    }
    weights_ = std::move(load_weights);
}

void ConvLayer::LoadBiases(std::vector<float> &load_weights) {
    if ((int)load_weights.size() != GetOutputs()) {
        throw "The linear layer is not acceptable.";
    }
    biases_ = std::move(load_weights);
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

const std::vector<float>& ConvLayer::GetWeights() const {
    return weights_;
}

const std::vector<float>& ConvLayer::GetBiases() const {
    return biases_;
}
