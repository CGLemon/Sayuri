#pragma once

#include <memory>

#include "neural/activation.h"
#include "neural/network_basic.h"
#include "neural/description.h"

class BlasForwardPipe : public NetworkForwardPipe {
public:
    virtual void Initialize(std::shared_ptr<DNNWeights> weights);

    virtual OutputResult Forward(const InputData &inpnt);

    virtual bool Valid();

    virtual void Load(std::shared_ptr<DNNWeights> weights);

    virtual void Reload(int);

    virtual void Release();

    virtual void Destroy();

private:
    void InitWinograd();

    void ResidualBlockForward(int board_size,
                              BlockBasic * tower_ptr,
                              bool use_winograd,
                              std::vector<float> &residual,
                              std::vector<float> &conv_in,
                              std::vector<float> &conv_out,
                              std::vector<float> &workspace0,
                              std::vector<float> &workspace1);

    void BottleneckBlockForward(int board_size,
                                BlockBasic * tower_ptr,
                                bool use_winograd,
                                std::vector<float> &residual,
                                std::vector<float> &conv_in,
                                std::vector<float> &conv_out,
                                std::vector<float> &workspace0,
                                std::vector<float> &workspace1);

    void MixerBlockForward(int board_size,
                           BlockBasic * tower_ptr,
                           std::vector<float> &residual,
                           std::vector<float> &conv_in,
                           std::vector<float> &conv_out);
};
