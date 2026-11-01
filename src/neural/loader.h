#pragma once

#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>

#include "neural/description.h"
#include "utils/splitter.h"

class DNNLoader {
public:
    static DNNLoader& Get();

    void FromFile(std::shared_ptr<DNNWeights> weights, std::string filename);

private:
    using LayerShape = std::vector<int>;
    using NetStack = std::vector<std::string>;
    using NetStruct = std::vector<LayerShape>;
    using NetInfo = std::unordered_map<std::string, std::string>;

    void Parse(std::istream& buffer);
    void ParseInfo(NetInfo& netinfo, std::istream& buffer) const;
    void ParseStack(NetStack& netstack, std::istream& buffer) const;
    void ParseStruct(NetStruct& netstruct, std::istream& buffer) const;
    void CheckMisc(NetInfo& netinfo, NetStack& netstack, NetStruct& netstruct);
    void DumpInfo() const;

    void FillWeights(NetInfo& netinfo,
                     NetStack& netstack,
                     NetStruct& netstruct,
                     std::istream& buffer) const;
    int FillBlock(int offset, Splitter block_spt, NetStruct& netstruct, std::istream& buffer) const;
    int FillResidualBlock(int offset,
                          ResidualBlock* block,
                          NetStruct& netstruct,
                          std::istream& buffer) const;
    int FillBottleneckBlock(int offset,
                            BottleneckBlock* block,
                            NetStruct& netstruct,
                            std::istream& buffer) const;
    int FillNestedBottleneckBlock(int offset,
                                  NestedBottleneckBlock* block,
                                  NetStruct& netstruct,
                                  std::istream& buffer) const;
    int
    FillMixerBlock(int offset, MixerBlock* block, NetStruct& netstruct, std::istream& buffer) const;
    int FillTransformerBlock(int offset,
                             TransformerBlock* block,
                             NetStruct& netstruct,
                             std::istream& buffer) const;
    void ProcessWeights() const;
    void GetWeightsFromBuffer(std::vector<float>& weights, std::istream& buffer) const;

    void FillFullyconnectLayer(LinearLayer& layer,
                               std::istream& buffer,
                               const int in_size,
                               const int out_size) const;

    void FillBatchnormLayer(BatchNormLayer& layer, std::istream& buffer, const int channels) const;

    void FillRmsnormLayer(RMSNormLayer& layer, std::istream& buffer, const int channels) const;

    void FillRopeLayer(RoPELayer& layer,
                       std::istream& buffer,
                       const int head_dim,
                       const int num_heads) const;

    void FillConvolutionLayer(ConvLayer& layer,
                              std::istream& buffer,
                              const int in_channels,
                              const int out_channels,
                              const int kernel_size) const;
    DNNWeights* weights_;
    bool use_binary_;
};
