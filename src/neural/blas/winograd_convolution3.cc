#include "neural/blas/winograd_convolution3.h"
#include "neural/blas/blas.h"
#include "neural/winograd_helper.h"

void ClearVector2D(std::vector<std::vector<float>> &vec2d) {
    for (auto &v: vec2d) {
        std::fill(std::begin(v), std::end(v), 0.f);
    }
}

void WinogradConvolution3::TransformIn(const int board_size,
                                       const std::vector<float>& in,
                                       std::vector<float>& V, const int C) {
    const int W = board_size;
    const int H = board_size;
    const int WTILES = GetWinogradWTiles(board_size);
    const int P = GetWinogradP(board_size);
    constexpr auto SQ2 = kSqrt2;

    const auto Wpad = 2 + kWinogradM * WTILES;
    constexpr auto buffersize = 32;

    // std::array<std::array<float, Wpad>, Wpad> in_pad{{{0.0f}}};
    std::vector<std::vector<float>> in_pad;
    in_pad.resize(Wpad);
    for (auto &v: in_pad) {
        v.resize(Wpad);
    }

    std::array<float, buffersize * kWinogradAlpha * kWinogradAlpha> buffer;
    auto buffer_offset = 0;
    auto buffer_entries = 0;

    // multiple vector [i0..i5] by Bt and produce [o0..o5]
    // const auto Bt = std::array<float, kWinogradTile>{
    //     1.0f,  0.0f,       -5.0f / 2.0f,  0.0f,        1.0f, 0.0f,
    //     0.0f, -SQ2,        -2.0f,         SQ2 / 2.0f,  1.0f, 0.0f,
    //     0.0f,  SQ2,        -2.0f,        -SQ2 / 2.0f,  1.0f, 0.0f,
    //     0.0f, -SQ2 / 2.0f, -1.0f / 2.0f,  SQ2,         1.0f, 0.0f,
    //     0.0f,  SQ2 / 2.0f, -1.0f / 2.0f, -SQ2,         1.0f, 0.0f,
    //     0.0f,  1.0f,        0.0f,        -5.0f / 2.0f, 0.0f, 1.0f};
    const auto multiply_bt = [](float& o0, float& o1, float& o2,
                                float& o3, float& o4, float& o5,
                                const float i0, const float i1, const float i2,
                                const float i3, const float i4, const float i5) {
        auto i3m1 = i1 * -SQ2 + i3 * (SQ2 / 2.0f);
        auto i4m2 = i2 * -2.0f + i4 * 1.0f;

        o0 = i0 + i2 * (-5.0f / 2.0f) + i4;
        o1 = i3m1 + i4m2;
        o2 = -i3m1 + i4m2;

        auto i3m1_2 = i3 * (SQ2) + i1 * (-SQ2 / 2.0f);
        auto i4m2_2 = i2 * (-1.0f / 2.0f) + i4;

        o3 = i3m1_2 + i4m2_2;
        o4 = -i3m1_2 + i4m2_2;

        o5 = i1 + i3 * (-5.0f / 2.0f) + i5;
    };

    for (int ch = 0; ch < C; ch++) {
        ClearVector2D(in_pad);

        for (int yin = 0; yin < H; yin++) {
            for (int xin = 0; xin < W; xin++) {
                in_pad[yin + 1][xin + 1] = in[ch * (W * H) + yin * W + xin];
            }
        }
        for (int block_y = 0; block_y < WTILES; block_y++) {
            // Tiles overlap by 2
            const auto yin = kWinogradM * block_y;
            for (int block_x = 0; block_x < WTILES; block_x++) {
                const auto xin = kWinogradM * block_x;
#define DECL_T1(XX)                                                            \
    float T1_##XX##_0, T1_##XX##_1, T1_##XX##_2, T1_##XX##_3, T1_##XX##_4,     \
        T1_##XX##_5;
                DECL_T1(0)
                DECL_T1(1)
                DECL_T1(2)
                DECL_T1(3)
                DECL_T1(4)
                DECL_T1(5)

                // Calculates transpose(B).x.B
#define MULTIPLY_BT(XX)                                                        \
    multiply_bt(T1_0_##XX, T1_1_##XX, T1_2_##XX, T1_3_##XX, T1_4_##XX,         \
                T1_5_##XX,                                                     \
                in_pad[yin + 0][xin + XX],                                     \
                in_pad[yin + 1][xin + XX],                                     \
                in_pad[yin + 2][xin + XX],                                     \
                in_pad[yin + 3][xin + XX],                                     \
                in_pad[yin + 4][xin + XX],                                     \
                in_pad[yin + 5][xin + XX]);
                MULTIPLY_BT(0)
                MULTIPLY_BT(1)
                MULTIPLY_BT(2)
                MULTIPLY_BT(3)
                MULTIPLY_BT(4)
                MULTIPLY_BT(5)

#define MULTIPLY_B(XX)                                                         \
    multiply_bt(                                                               \
        buffer[buffersize * (XX * kWinogradAlpha + 0) + buffer_entries],       \
        buffer[buffersize * (XX * kWinogradAlpha + 1) + buffer_entries],       \
        buffer[buffersize * (XX * kWinogradAlpha + 2) + buffer_entries],       \
        buffer[buffersize * (XX * kWinogradAlpha + 3) + buffer_entries],       \
        buffer[buffersize * (XX * kWinogradAlpha + 4) + buffer_entries],       \
        buffer[buffersize * (XX * kWinogradAlpha + 5) + buffer_entries],       \
        T1_##XX##_0, T1_##XX##_1, T1_##XX##_2, T1_##XX##_3, T1_##XX##_4,       \
        T1_##XX##_5);
                MULTIPLY_B(0)
                MULTIPLY_B(1)
                MULTIPLY_B(2)
                MULTIPLY_B(3)
                MULTIPLY_B(4)
                MULTIPLY_B(5)

                if (buffer_entries == 0) {
                    buffer_offset = ch * P + block_y * WTILES + block_x;
                }
                buffer_entries++;

                if (buffer_entries >= buffersize
                    || (ch == C - 1 && block_x == WTILES - 1
                        && block_y == WTILES - 1)) {

                    for (int i = 0; i < kWinogradAlpha * kWinogradAlpha; i++) {
                        for (int entry = 0; entry < buffer_entries; entry++) {
                            V[i * C * P + buffer_offset + entry] =
                                buffer[i * buffersize + entry];
                        }
                    }
                    buffer_entries = 0;
                }
            }
        }
    }
}

void WinogradConvolution3::Sgemm(const int board_size,
                                 const std::vector<float>& U,
                                 const std::vector<float>& V,
                                 std::vector<float>& M,
                                 const int C, const int K) {
    //    [C, K, P] are [input_channels, output_channels, Ptiles]
    // U dimensions are [36,  input_channels, output_channels].
    // V dimensions are [36,  input_channels, p_tiles].
    // M dimensions are [36, output_channels, p_tiles].

    const int P = GetWinogradP(board_size);
    for (int b = 0; b < kWinogradTile; b++) {
        const int offset_u = b * K * C;
        const int offset_v = b * C * P;
        const int offset_m = b * K * P;
        Blas::WinogradSgemm(offset_u, offset_v, offset_m,
                                K, P, C,
                                1.0f,
                                U.data(), K,
                                V.data(), P,
                                0.0f,
                                M.data(), P);
    }
}

void WinogradConvolution3::TransformOut(const int board_size,
                                        const std::vector<float>& M,
                                        std::vector<float>& Y, const int K) {
    const int W = board_size;
    const int H = board_size;
    const int WTILES = GetWinogradWTiles(board_size);
    const int P = GetWinogradP(board_size);

    constexpr auto SQ2 = kSqrt2;

    // multiple vector [i0..i5] by At and produce [o0..o3]
    // const auto At = std::array<float, kWinogradAlpha * kWinogradM>{
    //     1.0f, 1.0f,        1.0f,        1.0f,        1.0f,       0.0f,
    //     0.0f, SQ2 / 2.0f, -SQ2 / 2.0f,  SQ2,        -SQ2,        0.0f,
    //     0.0f, 1.0f / 2.0f, 1.0f / 2.0f, 2.0f,        2.0f,       0.0f,
    //     0.0f, SQ2 / 4.0f, -SQ2 / 4.0f,  2.0f * SQ2, -2.0f * SQ2, 1.0f};
    const auto multiply_at = [](float& o0, float& o1, float& o2, float& o3,
                                const float i0, const float i1,
                                const float i2, const float i3,
                                const float i4, const float i5) {
        auto t1p2 = (i1 + i2) * (1.0f / 2.0f);
        auto t1m2 = (i1 - i2) * (SQ2 / 4.0f);
        auto t3p4 = i3 + i4;
        auto t3m4 = (i3 - i4) * (SQ2);

        o0 = i0 + t1p2 + t1p2 + t3p4;
        o1 = t1m2 + t1m2 + t3m4;
        o2 = t1p2 + t3p4 + t3p4;
        o3 = t1m2 + t3m4 + t3m4 + i5;
    };

    for (int k = 0; k < K; k++) {
        for (int block_x = 0; block_x < WTILES; block_x++) {
            const auto x = kWinogradM * block_x;
            for (int block_y = 0; block_y < WTILES; block_y++) {
                const auto y = kWinogradM * block_y;

                const auto b = block_y * WTILES + block_x;
                using WinogradTile =
                    std::array<std::array<float, kWinogradAlpha>,
                               kWinogradAlpha>;
                WinogradTile temp_m;
                for (int xi = 0; xi < kWinogradAlpha; xi++) {
                    for (int nu = 0; nu < kWinogradAlpha; nu++) {
                        temp_m[xi][nu] =
                            M[(xi * kWinogradAlpha + nu) * K * P + k * P + b];
                    }
                }
                std::array<std::array<float, kWinogradAlpha>, kWinogradM> temp;
                std::array<std::array<float, kWinogradM>, kWinogradM> o;

                // Calculates transpose(A).temp_m.A
                for (int j = 0; j < kWinogradAlpha; j++) {
                    multiply_at(temp[0][j], temp[1][j], temp[2][j], temp[3][j],
                                temp_m[0][j], temp_m[1][j], temp_m[2][j],
                                temp_m[3][j], temp_m[4][j], temp_m[5][j]);
                }

                for (int i = 0; i < kWinogradM; i++) {
                    multiply_at(o[i][0], o[i][1], o[i][2], o[i][3],
                                temp[i][0], temp[i][1], temp[i][2],
                                temp[i][3], temp[i][4], temp[i][5]);
                }

                const auto y_ind = k * H * W + y * W + x;
                for (int i = 0; i < kWinogradM; i++) {
                    for (int j = 0; j < kWinogradM; j++) {
                        if (y + i < H && x + j < W) {
                            Y[y_ind + i * W + j] = o[i][j];
                        }
                    }
                }
            }
        }
    }
}

void WinogradConvolution3::Forward(const size_t board_size,
                                   const size_t input_channels,
                                   const size_t output_channels,
                                   const std::vector<float>& input,
                                   const std::vector<float>& U,
                                   std::vector<float>& V,
                                   std::vector<float>& M,
                                   std::vector<float>& output) {
    TransformIn(board_size, input, V, input_channels);
    Sgemm(board_size, U, V, M, input_channels, output_channels);
    TransformOut(board_size, M, output, output_channels);
}


size_t WinogradConvolution3::GetWorkspaceSize(const size_t board_size, const size_t channels) {
    return kWinogradTile * channels * GetWinogradP(board_size);
}
