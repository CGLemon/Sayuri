#include "neural/blas/attention.h"

#include <algorithm>
#include <cmath>

#include "neural/blas/blas.h"

void RMSNorm::Forward(const size_t board_size,
                      const size_t channels,
                      const std::vector<float>& input,
                      const std::vector<float>& gammas,
                      std::vector<float>& output) {
    static constexpr float kEpsilon = 1e-6f;
    const auto spatial_size = board_size * board_size;

    // Gather the mean square value of every intersection first. It keeps the
    // reduction contiguous in the NCHW layout.
    auto scales = std::vector<float>(spatial_size, 0.0f);

    for (auto c = size_t{0}; c < channels; ++c) {
        const float* input_ptr = input.data() + c * spatial_size;
        for (auto i = size_t{0}; i < spatial_size; ++i) {
            scales[i] += input_ptr[i] * input_ptr[i];
        }
    }
    for (auto i = size_t{0}; i < spatial_size; ++i) {
        scales[i] = 1.0f / std::sqrt(scales[i] / (float)channels + kEpsilon);
    }
    for (auto c = size_t{0}; c < channels; ++c) {
        const float* input_ptr = input.data() + c * spatial_size;
        float* output_ptr = output.data() + c * spatial_size;
        const auto gamma = gammas[c];
        for (auto i = size_t{0}; i < spatial_size; ++i) {
            output_ptr[i] = gamma * scales[i] * input_ptr[i];
        }
    }
}

void RotaryEmbedding::Forward(const size_t board_size,
                              const size_t num_heads,
                              const size_t head_dim,
                              const std::vector<float>& frequencies,
                              std::vector<float>& queries,
                              std::vector<float>& keys) {
    const auto width = board_size;
    const auto spatial_size = board_size * board_size;
    const auto num_pairs = head_dim / 2;

    const auto lambda_rotate =
        [](float* first, float* second, const float cos_val, const float sin_val) {
            const auto x0 = *first;
            const auto x1 = *second;
            *first = x0 * cos_val - x1 * sin_val;
            *second = x0 * sin_val + x1 * cos_val;
        };

    for (auto h = size_t{0}; h < num_heads; ++h) {
        for (auto p = size_t{0}; p < num_pairs; ++p) {
            const auto freq_x = frequencies[2 * (h * num_pairs + p) + 0];
            const auto freq_y = frequencies[2 * (h * num_pairs + p) + 1];

            // The rotated pair is the adjacent element pair of the head.
            const auto offset = (h * head_dim + 2 * p) * spatial_size;
            float* queries_ptr = queries.data() + offset;
            float* keys_ptr = keys.data() + offset;

            for (auto i = size_t{0}; i < spatial_size; ++i) {
                const auto pos_x = (float)(i % width);
                const auto pos_y = (float)(i / width);
                const auto angle = pos_x * freq_x + pos_y * freq_y;
                const auto cos_val = std::cos(angle);
                const auto sin_val = std::sin(angle);

                lambda_rotate(queries_ptr + i, queries_ptr + i + spatial_size, cos_val, sin_val);
                lambda_rotate(keys_ptr + i, keys_ptr + i + spatial_size, cos_val, sin_val);
            }
        }
    }
}

void MultiHeadAttention::Forward(const size_t board_size,
                                 const size_t num_heads,
                                 const size_t head_dim,
                                 const std::vector<float>& queries,
                                 const std::vector<float>& keys,
                                 const std::vector<float>& values,
                                 std::vector<float>& scores,
                                 std::vector<float>& output) {
    const auto seq_size = board_size * board_size;
    const auto head_size = head_dim * seq_size;
    const auto scale = 1.0f / std::sqrt((float)head_dim);

    for (auto h = size_t{0}; h < num_heads; ++h) {
        const float* queries_ptr = queries.data() + h * head_size;
        const float* keys_ptr = keys.data() + h * head_size;
        const float* values_ptr = values.data() + h * head_size;
        float* output_ptr = output.data() + h * head_size;

        // scores[i, j] = scale * dot(queries[:, i], keys[:, j])
        Blas::AttentionSgemmTN((int)seq_size,
                               (int)seq_size,
                               (int)head_dim,
                               scale,
                               queries_ptr,
                               keys_ptr,
                               scores.data());
        Softmax(seq_size, seq_size, scores.data());

        // output[:, i] = sum_j scores[i, j] * values[:, j]
        Blas::AttentionSgemmNT((int)head_dim,
                               (int)seq_size,
                               (int)seq_size,
                               1.0f,
                               values_ptr,
                               scores.data(),
                               output_ptr);
    }
}

void MultiHeadAttention::Softmax(const size_t rows, const size_t cols, float* data) {
    for (auto i = size_t{0}; i < rows; ++i) {
        float* row_ptr = data + i * cols;
        const auto max_val = *std::max_element(row_ptr, row_ptr + cols);

        auto sum = 0.0f;
        for (auto j = size_t{0}; j < cols; ++j) {
            row_ptr[j] = std::exp(row_ptr[j] - max_val);
            sum += row_ptr[j];
        }

        const auto inv_sum = 1.0f / sum;
        for (auto j = size_t{0}; j < cols; ++j) {
            row_ptr[j] *= inv_sum;
        }
    }
}
