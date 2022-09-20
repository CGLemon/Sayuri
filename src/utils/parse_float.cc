#include "utils/parse_float.h"

#include <cstdint>
#include <cstring>

// Assume the machine is little-endian.
float ParseBinFloat32(std::istream &in, bool big_endian) {
    static_assert(sizeof(char) * 4 == sizeof(float), "");

    char c[4];
    for (int i = 0; i < 4; ++i) {
        in.get(c[i]);
    }

    if (big_endian) {
        char buf[4];
        for (int i = 0; i < 4; ++i) {
            buf[i] = c[3-i];
        }
        memcpy(c, buf, sizeof(char) * 4);
    }

    float dest;
    memcpy(&dest, c, sizeof(float));

    return dest;
}

bool MatchFloat32(float f, std::uint32_t n) {
    static_assert(sizeof(float) == sizeof(std::uint32_t), "");

    std::uint32_t fval;
    memcpy(&fval, &f, sizeof(float));

    return fval == n;
}

bool IsLittleEndian() {
    unsigned int a = 0x12345678;
    unsigned char *c = (unsigned char*)(&a);
    if (*c == 0x78) {
       return true;
    }
    return false;
}
