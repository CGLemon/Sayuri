#pragma once

#include <array>
#include <vector>

void PtcoordsInit();
void PatternHashInit();
int CharToColor(char s);

static constexpr int kMaxPatternDist = 10;
static constexpr int kMaxPatternArea = kMaxPatternDist * kMaxPatternDist;

struct PtCoord { int x, y; };
extern std::array<PtCoord, kMaxPatternArea> kPointCoords;
extern std::array<int, kMaxPatternDist + 2> kPointIndex;
extern std::uint64_t PatternHash[8][4][kMaxPatternArea];
