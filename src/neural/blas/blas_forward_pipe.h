#pragma once

#include <memory>

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

    bool use_optimistic_policy_;

    std::shared_ptr<DNNWeights> weights_{nullptr};
};
