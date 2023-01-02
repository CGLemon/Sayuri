#pragma once

#include "neural/description.h"

#include <string>
#include <memory>
#include <fstream>
#include <unordered_map>

class DNNLoder {
public:
    static DNNLoder& Get();

    void FromFile(std::shared_ptr<DNNWeights> weights, std::string filename);

private:
    using LayerShape = std::vector<int>;
    using NetStruct = std::vector<LayerShape>;
    using NetInfo = std::unordered_map<std::string, std::string>;

    void Parse(std::shared_ptr<DNNWeights> weights, std::istream &buffer);
    void ParseInfo(NetInfo &netinfo, std::istream &buffer) const;
    void ParseStruct(NetStruct &netstruct, std::istream &buffer) const;
    void CkeckMisc(NetInfo &netinfo);
    void DumpInfo(std::shared_ptr<DNNWeights> weights) const;

    void FillWeights(NetInfo &netinfo,
                         NetStruct &netstruct,
                         std::shared_ptr<DNNWeights> weights,
                         std::istream &buffer) const;

    void ProcessWeights(std::shared_ptr<DNNWeights> weights) const;
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

    bool use_binary_;
    int version_;
};
