#include "utils/random.h"
#include <chrono>

namespace random_utils {

static inline std::uint64_t SplitMix64(std::uint64_t z) {
    /*
     * The detail of parameteres are from
     * https://github.com/lemire/testingRNG/blob/master/source/splitmix64.h
     */

    z += 0x9e3779b97f4a7c15;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

static inline std::uint64_t rotl(const std::uint64_t x, const int k) {
    return (x << k) | (x >> (64 - k));
}

static inline std::uint64_t GetSeed(std::uint64_t seed) {
    if (seed == kThreadSeed) {
        // Get the seed from thread id.
        auto thread_id = std::hash<std::thread::id>()(std::this_thread::get_id());
        seed = static_cast<std::uint64_t>(thread_id);
    } else if (seed == kTimeSeed) {
        // Get the seed from system time.
        auto get_time = std::chrono::system_clock::now().time_since_epoch().count();
        seed = static_cast<std::uint64_t>(get_time);
    }
    return seed;
}
} // namespace random_utils


#define RANDOM_INIT__(TYPE__, CNT__)                \
template<>                                          \
void Random<TYPE__>::InitSeed(std::uint64_t seed) { \
    seed = random_utils::GetSeed(seed);             \
    static_assert(kSeedSize >= CNT__,               \
        "The number of seeds is not enough?\n");    \
    for (auto i = size_t{0}; i < kSeedSize; ++i) {  \
        seed = random_utils::SplitMix64(seed);      \
        seeds_[i] = seed;                           \
    }                                               \
}


template<RandomType T>
thread_local std::uint64_t 
    Random<T>::seeds_[Random<T>::kSeedSize];

RANDOM_INIT__(RandomType::kSplitMix64, 1);

RANDOM_INIT__(RandomType::kXoroShiro128Plus, 2);

template<>
std::uint64_t Random<RandomType::kSplitMix64>::Generate() {
    /*
     * The detail of parameteres are from
     * https://github.com/lemire/testingRNG/blob/master/source/splitmix64.h
     */
    constexpr auto kSeedIndex = kSeedSize - 1;

    seeds_[kSeedIndex] += 0x9e3779b97f4a7c15;
    auto z = seeds_[kSeedIndex];
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

template<>
std::uint64_t Random<RandomType::kXoroShiro128Plus>::Generate() {
    /*
     * The detail of parameteres are from
     * https://github.com/lemire/testingRNG/blob/master/source/xoroshiro128plus.h
     */

    const std::uint64_t s0 = seeds_[0];
    std::uint64_t s1 = seeds_[1];
    const std::uint64_t result = s0 + s1;

    s1 ^= s0;
    seeds_[0] = random_utils::rotl(s0, 55) ^ s1 ^ (s1 << 14);
    seeds_[1] = random_utils::rotl(s1, 36);

    return result;
}
