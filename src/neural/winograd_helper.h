#pragma once

#include <vector>
#include <array>

// Winograd filter transformation changes 3x3 filters to M + 3 - 1
static constexpr int kWinogradM = 4;
static constexpr int kWinogradAlpha = kWinogradM + 3 - 1;
static constexpr int kWinogradTile = kWinogradAlpha * kWinogradAlpha;
static constexpr double kSqrt2 = 1.4142135623730951f; // Square root of 2

int GetWinogradWTiles(const int board_size);

int GetWinogradP(const int board_size);

std::vector<float> WinogradTransformF(const std::vector<float>& f,
                                          const int out_channels,
                                          const int in_channels);
