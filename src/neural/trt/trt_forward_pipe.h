#ifdef USE_TENSORRT

#include <atomic>
#include <memory>
#include <list>
#include <array>
#include <vector>
#include <mutex>
#include <condition_variable>

#include "neural/activation.h"
#include "neural/network_basic.h"
#include "neural/description.h"
#include "utils/threadpool.h"

class TrtForwardPipe : public NetworkForwardPipe {
    virtual void Initialize(std::shared_ptr<DNNWeights> weights);

    virtual OutputResult Forward(const InputData &input);

    virtual bool Valid() const;

    virtual void Construct(ForwardPipeOption option,
                           std::shared_ptr<DNNWeights> weights);

    virtual void Release();

    virtual void Destroy();
};

#endif
