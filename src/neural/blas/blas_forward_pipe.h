#pragma once

#include <memory>

#include "neural/network_basic.h"
#include "neural/description.h"

class BlasForwardPipe : public NetworkForwardPipe {
public:
   virtual void Initialize(std::shared_ptr<DNNWeights> weights);

   virtual OutputResult Forward(const InputData &inpnt);

   virtual bool Valid();

private:
    std::shared_ptr<DNNWeights> weights_{nullptr};

};
