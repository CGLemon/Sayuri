#pragma once

#include "neural/description.h"

#include <string>
#include <memory>
#include <fstream>
#include <unordered_map>

class DNNLoder {
public:
    static DNNLoder& Get();

    void FormFile(std::shared_ptr<DNNWeights> weights, std::string filename) const;

private:
    using LayerShape = std::vector<int>;
    using NetStruct = std::vector<LayerShape>;
    using NetInfo = std::unordered_map<std::string, std::string>;

    void Parse(std::shared_ptr<DNNWeights> weights, std::istream &buffer) const;
    void ParseInfo(NetInfo &netinfo, std::istream &buffer) const;
    void ParseStruct(NetStruct &netstruct, std::istream &buffer) const;
    void FillWeights(NetInfo &netinfo,
                     NetStruct &netstruct,
                     std::shared_ptr<DNNWeights> weights,
                     std::istream &buffer) const;

    void ProcessWeights(std::shared_ptr<DNNWeights> &weights, bool winograd) const;
    void GetWeightsFromBuffer(std::vector<float> &weights, std::istream &buffer) const;


    void FillFullyconnectLayer(LinearLayer &layer,
                               std::istream &buffer,
                               const int in_size,
                               const int out_size) const;

    void FillBatchnormLayer(BatchNormLayer &layer,
                            std::istream &buffer,
                            const int channels) const;

    void FillConvolutionLayer(ConvLayer &layer,
                              std::istream &buffer,
                              const int in_channels,
                              const int out_channels,
                              const int kernel_size) const;

};
