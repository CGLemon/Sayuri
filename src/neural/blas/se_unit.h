#pragma once

#include <vector>

class GlobalAvgPool {
public:
    GlobalAvgPool() = delete;
    static void Forward(const size_t board_size,
                        const size_t channels,
                        const std::vector<float> &input,
                        std::vector<float> &output);
};

class SEUnit {
public:
    SEUnit() = delete;
    static void Forward(const size_t board_size,
                        const size_t channels,
                        const size_t se_size,
                        std::vector<float> &input,
                        const std::vector<float> &residual,
                        const std::vector<float> &weights_w1,
                        const std::vector<float> &weights_b1,
                        const std::vector<float> &weights_w2,
                        const std::vector<float> &weights_b2);

private:
    static void SEProcess(const size_t board_size,
                          const size_t channels,
                          std::vector<float> &input,
                          const std::vector<float> &residual,
                          const std::vector<float> &scale);

};
