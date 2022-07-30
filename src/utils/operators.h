#pragma once

#define ENABLE_BITWISE_OPERATORS_ON(T)                               \
constexpr T operator | (T d1, T d2) { return T((int)d1 | (int)d2); } \
constexpr T operator & (T d1, T d2) { return T((int)d1 & (int)d2); } \
constexpr T operator ^ (T d1, T d2) { return T((int)d1 ^ (int)d2); } \
constexpr T operator << (T d1, int d2) { return T((int)d1 << d2); }  \
constexpr T operator >> (T d1, int d2) { return T((int)d1 >> d2); }
