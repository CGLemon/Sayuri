#pragma once

#include <iostream>

// Assume the machine is little-endian.
float ParseBinFloat32(std::istream &in, bool big_endian);

bool MatchFloat32(float f, std::uint32_t n);

bool IsLittleEndian();
