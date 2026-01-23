#include "utils/random.h"
#include <chrono>

namespace random_utils {

constexpr std::uint64_t SplitMix64(std::uint64_t z) {
    // Please see the details from:
    // https://github.com/lemire/testingRNG/blob/master/source/splitmix64.h

    z += 0x9e3779b97f4a7c15;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

constexpr std::uint64_t Rotl(const std::uint64_t x, const int k) {
    return (x << k) | (x >> (64 - k));
}

std::uint64_t ResolveSeed(std::uint64_t seed) {
    if (seed == kThreadSeed) {
        // Get the seed from thread ID.
        auto thread_id = std::hash<std::thread::id>()(std::this_thread::get_id());
        return static_cast<std::uint64_t>(thread_id);
    }
    if (seed == kTimeSeed) {
        // Get the seed from system time.
        auto get_time = std::chrono::system_clock::now().time_since_epoch().count();
        return static_cast<std::uint64_t>(get_time);
    }
    return seed;
}

} // namespace random_utils


#define RANDOM_INIT__(TYPE__, CNT__)                   \
template<>                                             \
void Random<TYPE__>::InitSeed(std::uint64_t seed) {    \
    seed = random_utils::ResolveSeed(seed);            \
    static_assert(kMaxSeedSize >= CNT__,               \
        "The number of seeds is not enough?\n");       \
    for (auto i = size_t{0}; i < kMaxSeedSize; ++i) {  \
        seed = random_utils::SplitMix64(seed);         \
        seeds_[i] = seed;                              \
    }                                                  \
}

template<RandomMethod R>
thread_local std::uint64_t
    Random<R>::seeds_[Random<R>::kMaxSeedSize];

RANDOM_INIT__(RandomMethod::kSplitMix64, 1);

RANDOM_INIT__(RandomMethod::kXoroShiro128Plus, 2);

template<>
std::uint64_t Random<RandomMethod::kSplitMix64>::Generate() {
    // Please see the details from:
    // https://github.com/lemire/testingRNG/blob/master/source/splitmix64.h

    constexpr auto kSeedIndex = kMaxSeedSize - 1;

    seeds_[kSeedIndex] += 0x9e3779b97f4a7c15;
    auto z = seeds_[kSeedIndex];
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

template<>
std::uint64_t Random<RandomMethod::kXoroShiro128Plus>::Generate() {
    // Please see the details from:
    // https://github.com/lemire/testingRNG/blob/master/source/xoroshiro128plus.h

    const std::uint64_t s0 = seeds_[0];
    std::uint64_t s1 = seeds_[1];
    const std::uint64_t result = s0 + s1;

    s1 ^= s0;
    seeds_[0] = random_utils::Rotl(s0, 55) ^ s1 ^ (s1 << 14);
    seeds_[1] = random_utils::Rotl(s1, 36);

    return result;
}
