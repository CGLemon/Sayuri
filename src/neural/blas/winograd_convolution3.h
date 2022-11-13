#pragma once

#include <vector>

class WinogradConvolution3 {
public:
    static void Forward(const size_t board_size,
                            const size_t output_channels,
                            const std::vector<float>& input,
                            const std::vector<float>& U,
                            std::vector<float>& V,
                            std::vector<float>& M,
                            std::vector<float>& output);

    static size_t GetWorkspaceSize(const size_t board_size, const size_t channels);

private:
    static void TransformIn(const int board_size,
                                const std::vector<float>& in,
                                std::vector<float>& V, int C);

    static void Sgemm(const int board_size,
                          const std::vector<float>& U,
                          const std::vector<float>& V,
                          std::vector<float>& M, int C, int K);

    static void TransformOut(const int board_size,
                                 const std::vector<float>& M,
                                 std::vector<float>& Y, int K);
};

