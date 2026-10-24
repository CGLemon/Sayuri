#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include "neural/blas/attention.h"
#include "neural/blas/blas.h"
#include "tests/unitest.h"

#ifdef USE_CUDA
#include "neural/cuda/cuda_common.h"
#include "neural/cuda/cuda_layers.h"
#endif

namespace {

constexpr float kTolerance = 1e-4f;

std::vector<float> RandomTensor(const size_t size, const int seed) {
    auto rng = std::mt19937(seed);
    auto dist = std::uniform_real_distribution<float>(-1.5f, 1.5f);

    auto tensor = std::vector<float>(size);
    for (auto& val : tensor) {
        val = dist(rng);
    }
    return tensor;
}

// The naive reference of the multi-head scaled dot-product attention.
std::vector<float> ReferenceAttention(const size_t board_size,
                                      const size_t num_heads,
                                      const size_t head_dim,
                                      const std::vector<float>& queries,
                                      const std::vector<float>& keys,
                                      const std::vector<float>& values) {
    const auto seq_size = board_size * board_size;
    const auto scale = 1.f / std::sqrt((float)head_dim);
    auto output = std::vector<float>(num_heads * head_dim * seq_size);

    for (auto h = size_t{0}; h < num_heads; ++h) {
        for (auto i = size_t{0}; i < seq_size; ++i) {
            auto weights = std::vector<float>(seq_size);
            auto max_val = -5000.f;

            for (auto j = size_t{0}; j < seq_size; ++j) {
                auto dot = 0.f;
                for (auto d = size_t{0}; d < head_dim; ++d) {
                    const auto c = (h * head_dim + d) * seq_size;
                    dot += queries[c + i] * keys[c + j];
                }
                weights[j] = scale * dot;
                max_val = std::max(max_val, weights[j]);
            }

            auto sum = 0.f;
            for (auto j = size_t{0}; j < seq_size; ++j) {
                weights[j] = std::exp(weights[j] - max_val);
                sum += weights[j];
            }

            for (auto d = size_t{0}; d < head_dim; ++d) {
                const auto c = (h * head_dim + d) * seq_size;
                auto acc = 0.f;
                for (auto j = size_t{0}; j < seq_size; ++j) {
                    acc += weights[j] * values[c + j];
                }
                output[c + i] = acc / sum;
            }
        }
    }
    return output;
}

#ifdef USE_CUDA
constexpr float kFp16Tolerance = 2e-2f;

// Copy the board at the top-left corner of a [channels, graph_size, graph_size]
// tensor which starts from 'offset'.
std::vector<float> CropBoard(const std::vector<float>& tensor,
                             const size_t offset,
                             const int channels,
                             const int graph_size,
                             const int board_size) {
    auto crop = std::vector<float>(channels * board_size * board_size);
    for (int c = 0; c < channels; ++c) {
        for (int y = 0; y < board_size; ++y) {
            for (int x = 0; x < board_size; ++x) {
                crop[(c * board_size + y) * board_size + x] =
                    tensor[offset + (c * graph_size + y) * graph_size + x];
            }
        }
    }
    return crop;
}

std::vector<float> CopyToHost(cuda::CudaHandles* handles, void* cuda_op, const size_t size) {
    auto output = std::vector<float>(size);
    cuda::WaitToFinish(handles->stream);
    cuda::CopyToHostOp(handles->fp16, output, &cuda_op);
    return output;
}

// Run the test with FP32, then with FP16 if the device supports it. Skip it
// if there is no GPU.
template <typename TestFunc> void RunEachPrecision(TestFunc test) {
    auto device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        return;
    }
    cuda::SetDevice(0);

    auto handles = cuda::CudaHandles{};
    handles.ApplyOnCurrentDevice();
    const auto support_fp16 = handles.fp16;

    handles.fp16 = false;
    test(&handles, kTolerance);
    if (support_fp16) {
        handles.fp16 = true;
        test(&handles, kFp16Tolerance);
    }
    handles.Release();
}

// The 'board_sizes' are the board sizes of each batch, placed at the top-left
// corner of the graph.
void CheckCudaAttention(cuda::CudaHandles* handles,
                        const float tolerance,
                        const int graph_size,
                        const std::vector<int>& board_sizes,
                        const bool use_mask) {
    const int batch = board_sizes.size();
    const int num_heads = 2;
    const int head_dim = 4;
    const int attn_channels = num_heads * head_dim;
    const int spatial = graph_size * graph_size;
    const auto tensor_size = (size_t)batch * attn_channels * spatial;

    const auto queries = RandomTensor(tensor_size, 111);
    const auto keys = RandomTensor(tensor_size, 222);
    const auto values = RandomTensor(tensor_size, 333);
    const auto frequencies = RandomTensor(num_heads * (head_dim / 2) * 2, 444);

    auto mask = std::vector<float>(batch * spatial);
    for (int b = 0; b < batch; ++b) {
        for (int idx = 0; idx < spatial; ++idx) {
            const auto inside =
                idx % graph_size < board_sizes[b] && idx / graph_size < board_sizes[b];
            mask[b * spatial + idx] = inside ? 1.f : 0.f;
        }
    }

    void* cuda_queries;
    void* cuda_keys;
    void* cuda_values;
    void* cuda_scores;
    void* cuda_mask = nullptr;
    cuda::MallocAndCopy(handles->fp16, &cuda_queries, queries);
    cuda::MallocAndCopy(handles->fp16, &cuda_keys, keys);
    cuda::MallocAndCopy(handles->fp16, &cuda_values, values);
    cuda::MallocCudaOp(handles->fp16, &cuda_scores, (size_t)batch * num_heads * spatial * spatial);
    if (use_mask) {
        cuda::MallocAndCopy(handles->fp16, &cuda_mask, mask);
    }

    auto attention = cuda::MultiHeadAttention(handles, batch, graph_size, num_heads, head_dim);
    attention.LoadWeights(frequencies);
    attention.Forward(batch, cuda_queries, cuda_keys, cuda_values, cuda_scores, cuda_mask);
    const auto output = CopyToHost(handles, cuda_queries, tensor_size);

    for (int b = 0; b < batch; ++b) {
        const int board_size = board_sizes[b];
        const auto offset = (size_t)b * attn_channels * spatial;

        auto q = CropBoard(queries, offset, attn_channels, graph_size, board_size);
        auto k = CropBoard(keys, offset, attn_channels, graph_size, board_size);
        const auto v = CropBoard(values, offset, attn_channels, graph_size, board_size);

        RotaryEmbedding::Forward(board_size, num_heads, head_dim, frequencies, q, k);
        const auto expected = ReferenceAttention(board_size, num_heads, head_dim, q, k, v);

        const auto result = CropBoard(output, offset, attn_channels, graph_size, board_size);
        for (auto i = size_t{0}; i < result.size(); ++i) {
            EXPECT_NEAR(result[i], expected[i], tolerance);
        }
    }

    cuda::ReportCUDAErrors(cudaFree(cuda_queries));
    cuda::ReportCUDAErrors(cudaFree(cuda_keys));
    cuda::ReportCUDAErrors(cudaFree(cuda_values));
    cuda::ReportCUDAErrors(cudaFree(cuda_scores));
    cuda::ReportCUDAErrors(cudaFree(cuda_mask));
}
#endif

} // namespace

UNITEST(Attention, SgemmTransposedFirstMatchesNaive) {
    const int M = 5;
    const int N = 7;
    const int K = 3;
    const auto alpha = 0.5f;

    // The A is a [K, M] row-major matrix and the B is a [K, N] one.
    const auto A = RandomTensor(K * M, 101);
    const auto B = RandomTensor(K * N, 202);
    auto C = std::vector<float>(M * N, 5000.f);

    Blas::AttentionSgemmTN(M, N, K, alpha, A.data(), B.data(), C.data());

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            auto expected = 0.f;
            for (int k = 0; k < K; ++k) {
                expected += A[k * M + m] * B[k * N + n];
            }
            EXPECT_NEAR(C[m * N + n], alpha * expected, kTolerance);
        }
    }
}

UNITEST(Attention, SgemmTransposedSecondMatchesNaive) {
    const int M = 4;
    const int N = 6;
    const int K = 3;
    const auto alpha = 1.f;

    // The A is a [M, K] row-major matrix and the B is a [N, K] one.
    const auto A = RandomTensor(M * K, 303);
    const auto B = RandomTensor(N * K, 404);
    auto C = std::vector<float>(M * N, 5000.f);

    Blas::AttentionSgemmNT(M, N, K, alpha, A.data(), B.data(), C.data());

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            auto expected = 0.f;
            for (int k = 0; k < K; ++k) {
                expected += A[m * K + k] * B[n * K + k];
            }
            EXPECT_NEAR(C[m * N + n], alpha * expected, kTolerance);
        }
    }
}

UNITEST(Attention, RMSNormMatchesKnownValues) {
    const size_t board_size = 2;
    const size_t channels = 2;

    // The expected values are computed by the RMSNorm of train/torch.
    const auto input = std::vector<float>{3.f, 1.f, -2.f, 0.f, 4.f, 1.f, 0.f, 0.f};
    const auto gammas = std::vector<float>{2.f, 0.5f};
    const auto expected =
        std::vector<float>{1.697056f, 1.999999f, -2.828426f, 0.f, 0.565685f, 0.5f, 0.f, 0.f};

    auto output = std::vector<float>(channels * board_size * board_size);
    RMSNorm::Forward(board_size, channels, input, gammas, output);

    for (auto i = size_t{0}; i < output.size(); ++i) {
        EXPECT_NEAR(output[i], expected[i], kTolerance);
    }
}

UNITEST(Attention, RMSNormHasUnitRootMeanSquare) {
    const size_t board_size = 3;
    const size_t channels = 6;
    const auto spatial_size = board_size * board_size;

    const auto input = RandomTensor(channels * spatial_size, 505);
    const auto gammas = std::vector<float>(channels, 1.f);

    auto output = std::vector<float>(channels * spatial_size);
    RMSNorm::Forward(board_size, channels, input, gammas, output);

    for (auto i = size_t{0}; i < spatial_size; ++i) {
        auto sq_sum = 0.f;
        for (auto c = size_t{0}; c < channels; ++c) {
            sq_sum += output[c * spatial_size + i] * output[c * spatial_size + i];
        }
        EXPECT_NEAR(std::sqrt(sq_sum / (float)channels), 1.f, kTolerance);
    }
}

UNITEST(Attention, RotaryEmbeddingKeepsNormAndOrigin) {
    const size_t board_size = 4;
    const size_t num_heads = 2;
    const size_t head_dim = 4;
    const auto spatial_size = board_size * board_size;

    const auto queries = RandomTensor(num_heads * head_dim * spatial_size, 707);
    const auto keys = RandomTensor(num_heads * head_dim * spatial_size, 808);
    const auto frequencies = RandomTensor(num_heads * (head_dim / 2) * 2, 909);

    auto rotated_q = queries;
    auto rotated_k = keys;
    RotaryEmbedding::Forward(board_size, num_heads, head_dim, frequencies, rotated_q, rotated_k);

    for (auto h = size_t{0}; h < num_heads; ++h) {
        for (auto p = size_t{0}; p < head_dim / 2; ++p) {
            const auto offset = (h * head_dim + 2 * p) * spatial_size;
            for (auto i = size_t{0}; i < spatial_size; ++i) {
                const auto x0 = queries[offset + i];
                const auto x1 = queries[offset + spatial_size + i];
                const auto y0 = rotated_q[offset + i];
                const auto y1 = rotated_q[offset + spatial_size + i];

                // The rotation preserves the length of the element pair.
                EXPECT_NEAR(y0 * y0 + y1 * y1, x0 * x0 + x1 * x1, kTolerance);

                // The first intersection is the origin, so it is not rotated.
                if (i == 0) {
                    EXPECT_NEAR(y0, x0, kTolerance);
                    EXPECT_NEAR(y1, x1, kTolerance);
                }
            }
        }
    }
}

UNITEST(Attention, RotaryEmbeddingMatchesNaive) {
    const size_t board_size = 3;
    const size_t num_heads = 1;
    const size_t head_dim = 4;
    const auto spatial_size = board_size * board_size;

    const auto queries = RandomTensor(num_heads * head_dim * spatial_size, 111);
    const auto frequencies = RandomTensor(num_heads * (head_dim / 2) * 2, 222);

    auto rotated_q = queries;
    auto rotated_k = queries;
    RotaryEmbedding::Forward(board_size, num_heads, head_dim, frequencies, rotated_q, rotated_k);

    for (auto p = size_t{0}; p < head_dim / 2; ++p) {
        const auto freq_x = frequencies[2 * p + 0];
        const auto freq_y = frequencies[2 * p + 1];
        const auto offset = (2 * p) * spatial_size;

        for (auto i = size_t{0}; i < spatial_size; ++i) {
            const auto angle = (float)(i % board_size) * freq_x + (float)(i / board_size) * freq_y;
            const auto x0 = queries[offset + i];
            const auto x1 = queries[offset + spatial_size + i];

            EXPECT_NEAR(
                rotated_q[offset + i], x0 * std::cos(angle) - x1 * std::sin(angle), kTolerance);
            EXPECT_NEAR(rotated_q[offset + spatial_size + i],
                        x0 * std::sin(angle) + x1 * std::cos(angle),
                        kTolerance);
        }
    }
}

UNITEST(Attention, MultiHeadAttentionMatchesNaive) {
    const size_t board_size = 4;
    const size_t num_heads = 3;
    const size_t head_dim = 4;
    const auto seq_size = board_size * board_size;
    const auto total_size = num_heads * head_dim * seq_size;

    const auto queries = RandomTensor(total_size, 333);
    const auto keys = RandomTensor(total_size, 444);
    const auto values = RandomTensor(total_size, 555);
    const auto expected =
        ReferenceAttention(board_size, num_heads, head_dim, queries, keys, values);

    auto scores = std::vector<float>(seq_size * seq_size);
    auto output = std::vector<float>(total_size);
    MultiHeadAttention::Forward(
        board_size, num_heads, head_dim, queries, keys, values, scores, output);

    for (auto i = size_t{0}; i < output.size(); ++i) {
        EXPECT_NEAR(output[i], expected[i], kTolerance);
    }
}

UNITEST(Attention, MultiHeadAttentionIsConvexCombination) {
    const size_t board_size = 3;
    const size_t num_heads = 2;
    const size_t head_dim = 4;
    const auto seq_size = board_size * board_size;
    const auto total_size = num_heads * head_dim * seq_size;

    const auto queries = RandomTensor(total_size, 666);
    const auto keys = RandomTensor(total_size, 777);

    // Every token shares the same value, so the attention output must be that
    // value no matter what the attention weights are.
    auto values = std::vector<float>(total_size);
    for (auto c = size_t{0}; c < num_heads * head_dim; ++c) {
        for (auto i = size_t{0}; i < seq_size; ++i) {
            values[c * seq_size + i] = 0.25f * (float)c;
        }
    }

    auto scores = std::vector<float>(seq_size * seq_size);
    auto output = std::vector<float>(total_size);
    MultiHeadAttention::Forward(
        board_size, num_heads, head_dim, queries, keys, values, scores, output);

    for (auto c = size_t{0}; c < num_heads * head_dim; ++c) {
        for (auto i = size_t{0}; i < seq_size; ++i) {
            EXPECT_NEAR(output[c * seq_size + i], 0.25f * (float)c, kTolerance);
        }
    }
}

#ifdef USE_CUDA
UNITEST(CudaAttention, RMSNormMatchesBlas) {
    RunEachPrecision([](cuda::CudaHandles* handles, const float tolerance) {
        const int batch = 2;
        const int board_size = 3;
        const int channels = 6;
        const auto plane_size = (size_t)channels * board_size * board_size;

        const auto input = RandomTensor(batch * plane_size, 505);
        const auto gammas = RandomTensor(channels, 606);

        void* cuda_input;
        void* cuda_output;
        cuda::MallocAndCopy(handles->fp16, &cuda_input, input);
        cuda::MallocCudaOp(handles->fp16, &cuda_output, input.size());

        auto norm = cuda::RMSNorm(handles, batch, board_size, channels);
        norm.LoadWeights(gammas);
        norm.Forward(batch, cuda_output, cuda_input);
        const auto output = CopyToHost(handles, cuda_output, input.size());

        for (int b = 0; b < batch; ++b) {
            const auto begin = std::begin(input) + b * plane_size;
            const auto batch_input = std::vector<float>(begin, begin + plane_size);
            auto expected = std::vector<float>(plane_size);
            RMSNorm::Forward(board_size, channels, batch_input, gammas, expected);

            for (auto i = size_t{0}; i < plane_size; ++i) {
                EXPECT_NEAR(output[b * plane_size + i], expected[i], tolerance);
            }
        }

        cuda::ReportCUDAErrors(cudaFree(cuda_input));
        cuda::ReportCUDAErrors(cudaFree(cuda_output));
    });
}

UNITEST(CudaAttention, MultiHeadAttentionMatchesReference) {
    RunEachPrecision([](cuda::CudaHandles* handles, const float tolerance) {
        CheckCudaAttention(handles, tolerance, 4, {4, 4}, false);
    });
}

UNITEST(CudaAttention, MaskedAttentionMatchesSmallerBoard) {
    RunEachPrecision([](cuda::CudaHandles* handles, const float tolerance) {
        CheckCudaAttention(handles, tolerance, 4, {3, 4}, true);
    });
}
#endif
