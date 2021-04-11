#pragma once

#include <cstdint>
#include <limits>
#include <random>
#include <thread>

// Select the different seed from different ways.
// "kThreadSeed" is default.
static constexpr std::uint64_t kThreadSeed = 0;
static constexpr std::uint64_t kTimeSeed = 1;


// Select the different random generator that you want.
enum RandomType {
    kSplitMix64,
    kXoroShiro128Plus
};

template<RandomType RandomType>
class Random {
public:
    Random() = delete;

    Random(std::uint64_t seed);

    static Random &Get(const std::uint64_t seed = kThreadSeed);

    // Generate the random number
    std::uint64_t Generate();

    template<int Range>
    std::uint32_t RandFix() {
        static_assert(0 < Range && Range < std::numeric_limits<std::uint32_t>::max(),
                          "randfix out of range?\n");
        return static_cast<std::uint32_t>(Generate()) % Range;
    }

    template<int Precision>
    bool Roulette(float threshold) {
        const int res = RandFix<Precision>();
        const int thres = (float)Precision * threshold;
        if (thres < res) {
            return true;
        }
        return false;
    }

    // The interface for STL.
    using result_type = std::uint64_t;

    constexpr static result_type min() {
        return std::numeric_limits<result_type>::min();
    }

    constexpr static result_type max() {
        return std::numeric_limits<result_type>::max();
    }

    result_type operator()() { return Generate(); }

private:
    static constexpr size_t kSeedSize = 2;

    static thread_local std::uint64_t seeds_[kSeedSize];

    void InitSeed(std::uint64_t);
};

template<RandomType T>
Random<T>::Random(std::uint64_t seed) {
    InitSeed(seed);
}

template<RandomType T>
Random<T> &Random<T>::Get(const std::uint64_t seed) {
    static thread_local Random s_rng{seed};
    return s_rng;
}
