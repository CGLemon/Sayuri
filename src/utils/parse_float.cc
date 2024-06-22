#include "utils/parse_float.h"

#include <cstring>

float ParseBinFloat32(std::istream &in, bool big_endian) {
    static_assert(sizeof(char) * 4 ==
                      sizeof(float),
                      "The float must be 4 bytes.");
    char c[4];
    for (int i = 0; i < 4; ++i) {
        in.get(c[i]);
    }

    if (big_endian) {
        char buf[4];
        for (int i = 0; i < 4; ++i) {
            buf[i] = c[3-i];
        }
        std::memcpy(c, buf, sizeof(char) * 4);
    }

    float dest;
    std::memcpy(&dest, c, sizeof(float));

    return dest;
}

bool MatchFloat32(float f, std::uint32_t n) {
    static_assert(sizeof(float) ==
                      sizeof(std::uint32_t),
                      "The float must be 4 bytes.");

    std::uint32_t fval;
    std::memcpy(&fval, &f, sizeof(float));

    return fval == n;
}

bool IsLittleEndian() {
    static_assert(sizeof(unsigned int) ==
                      sizeof(std::uint32_t),
                      "The int must be 4 bytes.");

    unsigned int a = 0x12345678;
    unsigned char *c = (unsigned char*)(&a);
    if (*c == 0x78) {
       return true;
    }
    return false;
}
