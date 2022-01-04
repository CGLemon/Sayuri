#include "neural/description.h"

void LinearLayer::Set(int inputs, int outputs) {
    inputs_ = inputs;
    outputs_ = outputs;
}

void LinearLayer::LoadWeights(std::vector<float> &load_weights) {
    if ((int)load_weights.size() != GetInputs() * GetOutputs()) {
        throw "The weights size of linear layer is not acceptable";
    }
    weights_ = std::move(load_weights);
}

void LinearLayer::LoadBiases(std::vector<float> &load_weights) {
    if ((int)load_weights.size() != GetOutputs()) {
        throw "The biases size of linear layer are is acceptable";
    }
    biases_ = std::move(load_weights);
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
    if ((int)load_weights.size() != GetChannels()) {
        throw "The means size of batch normallayer are not acceptable";
    }
    means_ = std::move(load_weights);
}

void BatchNormLayer::LoadStddevs(std::vector<float> &load_weights){
    if ((int)load_weights.size() != GetChannels()) {
        throw "The StdDevs size of batch normallayer are not acceptable";
    }
    ProcessVariant(load_weights);
    stddevs_ = std::move(load_weights);
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
    if ((int)load_weights.size() != GetInputs() * GetOutputs() * GetFilter() * GetFilter()) {
        throw "The weights size of convolutional layer is not acceptable";
    }
    weights_ = std::move(load_weights);
}

void ConvLayer::LoadBiases(std::vector<float> &load_weights) {
    if ((int)load_weights.size() != GetOutputs()) {
        throw "The biases size of convolutional layer is not acceptable";
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

std::vector<float>& ConvLayer::GetWeights() {
    return weights_;
}

std::vector<float>& ConvLayer::GetBiases() {
    return biases_;
}
