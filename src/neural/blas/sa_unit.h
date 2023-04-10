#pragma once

#include <vector>
#include <cstddef>

class ChannelPooling {
public:
    ChannelPooling() = delete;
    static void Forward(const size_t board_size,
                        const size_t channels,
                        const std::vector<float> &input,
                        std::vector<float> &output);

private:
    static constexpr size_t kMaxBSize = 19;
    static constexpr size_t kMinBSize = 9;
    static constexpr float kAvgBSize = (float)(kMaxBSize + kMinBSize) / 2.f;
};

class SAUnit {
public:
    SAUnit() = delete;
    static void Forward(const size_t board_size,
                        const size_t channels,
                        std::vector<float> &input,
                        const std::vector<float> &residual,
                        const std::vector<float> &weights,
                        const std::vector<float> &biases,
                        std::vector<float> &workspace,
                        bool ReLU);
private:
    static void SAProcess(const size_t board_size,
                          const size_t channels,
                          std::vector<float> &input,
                          const std::vector<float> &residual,
                          const std::vector<float> &scale,
                          bool ReLU);

};
