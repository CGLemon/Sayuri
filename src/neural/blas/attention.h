#pragma once

#include <cstddef>
#include <vector>

// Root-mean-square normalization along the channel dimension. Each intersection
// of the NCHW tensor is normalized independently.
class RMSNorm {
public:
    RMSNorm() = delete;
    static void Forward(const size_t board_size,
                        const size_t channels,
                        const std::vector<float>& input,
                        const std::vector<float>& gammas,
                        std::vector<float>& output);
};

// Learnable 2D rotary position embedding. There is one (omega_x, omega_y)
// frequency pair per head, per dimension pair. It rotates the adjacent element
// pairs of the queries and the keys in-place.
class RotaryEmbedding {
public:
    RotaryEmbedding() = delete;
    static void Forward(const size_t board_size,
                        const size_t num_heads,
                        const size_t head_dim,
                        const std::vector<float>& frequencies,
                        std::vector<float>& queries,
                        std::vector<float>& keys);
};

// Multi-head scaled dot-product attention. Every intersection is a valid token
// so there is no attention mask here. The 'scores' is the [S, S] workspace.
class MultiHeadAttention {
public:
    MultiHeadAttention() = delete;
    static void Forward(const size_t board_size,
                        const size_t num_heads,
                        const size_t head_dim,
                        const std::vector<float>& queries,
                        const std::vector<float>& keys,
                        const std::vector<float>& values,
                        std::vector<float>& scores,
                        std::vector<float>& output);

private:
    static void Softmax(const size_t rows, const size_t cols, float* data);
};
