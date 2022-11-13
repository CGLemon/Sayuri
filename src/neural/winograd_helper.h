#pragma once

#include <vector>
#include <array>

/*
constexpr auto WINOGRAD_M = 4;
constexpr auto WINOGRAD_ALPHA = WINOGRAD_M + 3 - 1;
constexpr auto WINOGRAD_WTILES =
    BOARD_SIZE / WINOGRAD_M + (BOARD_SIZE % WINOGRAD_M != 0);
constexpr auto WINOGRAD_TILE = WINOGRAD_ALPHA * WINOGRAD_ALPHA;
constexpr auto WINOGRAD_P = WINOGRAD_WTILES * WINOGRAD_WTILES;
constexpr auto SQ2 = 1.4142135623730951f; // Square root of 2
*/

// Winograd filter transformation changes 3x3 filters to M + 3 - 1
static constexpr int kWinogradM = 4;
static constexpr int kWinogradAlpha = kWinogradM + 3 - 1;
static constexpr int kWinogradTile = kWinogradAlpha * kWinogradAlpha;
static constexpr double kSqrt2 = 1.4142135623730951f; // Square root of 2

int GetWinogradWTiles(const int board_size);

int GetWinogradP(const int board_size);

std::vector<float> WinogradTransformF(const std::vector<float>& f,
                                          const int outputs,
                                          const int channels);
