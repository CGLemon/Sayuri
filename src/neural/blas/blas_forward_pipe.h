#pragma once

#include <memory>

#include "neural/activation.h"
#include "neural/network_basic.h"
#include "neural/description.h"

class BlasForwardPipe : public NetworkForwardPipe {
public:
    virtual void Initialize(std::shared_ptr<DNNWeights> weights);

    virtual OutputResult Forward(const InputData &inpnt);

    virtual void Construct(ForwardPipeParameters param,
                           std::shared_ptr<DNNWeights> weights);

    virtual void Release();

    virtual void Destroy();

    virtual bool Valid() const;

private:
    void InitWinograd();

    void Convolution3Forward(const int board_size,
                             const int input_channels,
                             const int output_channels,
                             const std::vector<float> &input,
                             const std::vector<float> &weights,
                             std::vector<float> &workspace0,
                             std::vector<float> &workspace1,
                             std::vector<float> &output);

    void ResidualBlockForward(const int board_size,
                              BlockBasic * tower_ptr,
                              std::vector<float> &residual,
                              std::vector<float> &conv_in,
                              std::vector<float> &conv_out,
                              std::vector<float> &workspace0,
                              std::vector<float> &workspace1);

    void BottleneckBlockForward(const int board_size,
                                BlockBasic * tower_ptr,
                                std::vector<float> &residual,
                                std::vector<float> &conv_in,
                                std::vector<float> &conv_out,
                                std::vector<float> &workspace0,
                                std::vector<float> &workspace1);

    void NestedBottleneckBlockForward(const int board_size,
                                      BlockBasic * tower_ptr,
                                      std::vector<float> &residual0,
                                      std::vector<float> &residual1,
                                      std::vector<float> &conv_in,
                                      std::vector<float> &conv_out,
                                      std::vector<float> &workspace0,
                                      std::vector<float> &workspace1);

    void MixerBlockForward(const int board_size,
                           BlockBasic * tower_ptr,
                           std::vector<float> &residual,
                           std::vector<float> &conv_in,
                           std::vector<float> &conv_out);

    void FillOutputs(const std::vector<float> &output_prob,
                     const std::vector<float> &output_pass,
                     const std::vector<float> &output_misc,
                     const std::vector<float> &output_ownership,
                     const InputData &inpnts,
                     OutputResult &output);
};
