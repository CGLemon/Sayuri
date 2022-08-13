#pragma once

// Enable the operators in the global scope.
#define ENABLE_BITWISE_OPERATORS_ON(T)                               \
constexpr T operator | (T d1, T d2) { return T((int)d1 | (int)d2); } \
constexpr T operator & (T d1, T d2) { return T((int)d1 & (int)d2); } \
constexpr T operator ^ (T d1, T d2) { return T((int)d1 ^ (int)d2); } \
constexpr T operator << (T d1, int d2) { return T((int)d1 << d2); }  \
constexpr T operator >> (T d1, int d2) { return T((int)d1 >> d2); }

// Enable the operators in the class scope.
#define ENABLE_FRIEND_BITWISE_OPERATORS_ON(T)                               \
constexpr T friend operator | (T d1, T d2) { return T((int)d1 | (int)d2); } \
constexpr T friend operator & (T d1, T d2) { return T((int)d1 & (int)d2); } \
constexpr T friend operator ^ (T d1, T d2) { return T((int)d1 ^ (int)d2); } \
constexpr T friend operator << (T d1, int d2) { return T((int)d1 << d2); }  \
constexpr T friend operator >> (T d1, int d2) { return T((int)d1 >> d2); }
