#pragma once

#include <cstdint>
#include <limits>
#include <cstring>

static_assert(std::numeric_limits<float>::is_iec559, "Machine must support for IEEE-754.");

enum class FpConvertStatus {
    kNormal     = 1,
    kZero       = 1 << 1,
    kDenormal   = 1 << 2,
    kNotANumber = 1 << 3,
    kInfinite   = 1 << 4,
    kUnderflow  = 1 << 5,
    kOverflow   = 1 << 6
};

// IEEE 754 float16
// bit  1~10: mantissa
// bit 11~15: exponent
// bit    16: sign
typedef std::uint16_t half_float_t;

// IEEE 754 float32
// bit  1~23: mantissa
// bit 24~31: exponent
// bit    32: sign
typedef std::uint32_t single_float_t;


template<std::float_round_style R>
single_float_t Rounded(single_float_t value,
                           unsigned int g, unsigned int s) {
    // The 'g' is ground bit.
    // The 's' is sticky bit.
    return (R==std::round_to_nearest) ? (value+(g&(s|value))) :
               (R==std::round_toward_infinity) ? (value+(~(value>>15)&(g|s))) :
               (R==std::round_toward_neg_infinity) ? (value+((value>>15)&(g|s))) :
               value;
}

inline FpConvertStatus FloatToHalf(half_float_t &fp16, float fp32) {
    single_float_t buf;
    std::memcpy(&buf, &fp32, sizeof(float));

    unsigned int sign = (buf>>16) & 0x8000;
    buf &= 0x7FFFFFFF; // Remove the sign field.

    if (buf > 0x7F800000) {
        fp16 = sign | 0x7C00 | (0x200|((buf>>13)&0x3FF));
        return FpConvertStatus::kNotANumber;
    }
    if (buf == 0x7F800000) {
        fp16 = sign | 0x7C00;
        return FpConvertStatus::kInfinite;
    }
    if (buf >= 0x47800000) {
        // Overflow case, we always set the half as greatest
        // number.
        fp16 = sign | 0x7800 | 0x03ff;
        return FpConvertStatus::kOverflow;
    }
    if (buf >= 0x38800000) {
        fp16 = Rounded<std::round_to_nearest>(
                   sign|(((buf>>23)-112)<<10)|((buf>>13)&0x3FF),
                   (buf>>12)&1,
                   (buf&0xFFF)!=0);
        return FpConvertStatus::kNormal;
    }
    if (buf >= 0x33000000) {
        int i = 125 - (buf>>23);
        buf = (buf&0x7FFFFF) | 0x800000;
        fp16 = Rounded<std::round_to_nearest>(
                   sign|(buf>>(i+1)),
                   (buf>>i)&1,
                   (buf&((static_cast<std::uint32_t>(1)<<i)-1))!=0);
        return FpConvertStatus::kDenormal;
    }

    // Underflow or zero case, we always set the half as zero.
    fp16 = sign;
    if(buf != 0) {
        return FpConvertStatus::kUnderflow;
    }
    return FpConvertStatus::kZero;
}

inline FpConvertStatus HalfToFloat(float &fp32, half_float_t fp16) {
    single_float_t buf = static_cast<single_float_t>(fp16&0x8000) << 16;

    int abs = fp16 & 0x7FFF;
    if (abs) {
        buf |= 0x38000000 << static_cast<unsigned>(abs>=0x7C00);
        for(; abs<0x400; abs<<=1,buf-=0x800000) ;
        buf += (static_cast<single_float_t>(abs) << 13);
    }

    std::memcpy(&fp32, &buf, sizeof(float));
    return FpConvertStatus::kNormal;
}

inline half_float_t GetFp16(float fp32) {
    half_float_t fp16;
    FloatToHalf(fp16, fp32);
    return fp16;
}
