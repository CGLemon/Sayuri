#ifdef USE_TENSORRT

#include "neural/trt/trt_forward_pipe.h"

void TrtForwardPipe::Initialize(std::shared_ptr<DNNWeights> weights) {

}

OutputResult TrtForwardPipe::Forward(const InputData &input) {

}

bool TrtForwardPipe::Valid() const {

}

void TrtForwardPipe::Construct(ForwardPipeOption option,
                                std::shared_ptr<DNNWeights> weights) {
}

void TrtForwardPipe::Release() {

}

void TrtForwardPipe::Destroy() {

}

#endif
