#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <random>
#include <thread>

// Special sentinel values for automatic seeding. "kThreadSeed" is default.
static constexpr std::uint64_t kThreadSeed = -1;
static constexpr std::uint64_t kTimeSeed = -2;

// Select the different random generator that you want.
enum RandomMethod {
    kSplitMix64,
    kXoroShiro128Plus
};

template<RandomMethod=kXoroShiro128Plus>
class Random {
public:
    Random() = delete;

    Random(std::uint64_t seed);

    static Random& Get(const std::uint64_t seed = kThreadSeed);

    // Generate a 64-bit uniformly distributed random number.
    // The returned value is assumed to cover the full uint64_t space.
    std::uint64_t Generate();

    // Generate a random integer in [0, Range).
    std::uint32_t RandFix(std::uint32_t range);

    // Returns true with probability.
    bool Roulette(double prob);

    // The interface for STL-compatible random generators.
    using result_type = std::uint64_t;

    // The interface for STL-compatible random generators.
    constexpr static result_type min() {
        return std::numeric_limits<result_type>::min();
    }

    // The interface for STL-compatible random generators.
    constexpr static result_type max() {
        return std::numeric_limits<result_type>::max();
    }

    result_type operator()() { return Generate(); }

private:
    static constexpr size_t kMaxSeedSize = 2;

    static thread_local std::uint64_t seeds_[kMaxSeedSize];

    void InitSeed(std::uint64_t);
};

template<RandomMethod R>
Random<R>::Random(std::uint64_t seed) {
    InitSeed(seed);
}

template<RandomMethod R>
Random<R>& Random<R>::Get(const std::uint64_t seed) {
    static thread_local Random s_rng{seed};
    return s_rng;
}

template<RandomMethod R>
std::uint32_t Random<R>::RandFix(std::uint32_t range) {
    // Please see the details from:
    // https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
    return ((Generate() >> 32) * static_cast<std::uint64_t>(range)) >> 32;
}

template<RandomMethod R>
bool Random<R>::Roulette(double prob) {
    constexpr long double k2Pow64 = 18446744073709551616.0L; // 2^64

    prob = std::max(0.0, prob);
    prob = std::min(prob, 1.0);
    if (prob <= 0.0) {
        return false;
    }
    if (prob >= 1.0) {
        return true;
    }

    // The thres should be in [1, 2^64-1].
    const std::uint64_t thres = (std::uint64_t)(prob * k2Pow64);
    return Generate() <= thres;
}
