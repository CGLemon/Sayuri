#pragma once

#include <cstdint>
#include <iostream>

// Assume the defaul stream is little-endian.
float ParseBinFloat32(std::istream &in, bool big_endian=false);

bool MatchFloat32(float f, std::uint32_t n);

bool IsLittleEndian();
