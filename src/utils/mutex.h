#pragma once

#include "utils/cppattributes.h"

#include <mutex>
#include <atomic>
#include <cassert>
#include <thread>

#if !defined(__arm__) && !defined(__aarch64__) && !defined(_M_ARM) && \
    !defined(_M_ARM64)
#include <emmintrin.h>
#endif

class SMP {
public:
    // Get number of cpu cores.
    static size_t GetNumberOfCores() {
        return std::thread::hardware_concurrency();
    }
};

// A very simple mutex lock.
class CAPABILITY("mutex") Mutex {
public:
    // std::lock_guard<Mutex> wrapper.
    class SCOPED_CAPABILITY Lock {
    public:
        Lock(Mutex& m) ACQUIRE(m) : lock_(m) {}
        ~Lock() RELEASE() {}

    private:
        std::lock_guard<Mutex> lock_;
    };

    void lock() ACQUIRE() {
        // Test and Test-and-Set reduces memory contention
        // However, just trying to Test-and-Set first improves performance in almost
        // all cases
        while (exclusive_.exchange(true, std::memory_order_acquire)) {
            while (exclusive_.load(std::memory_order_relaxed));
        }
    }

    void unlock() RELEASE() {
        auto lock_held = exclusive_.exchange(false, std::memory_order_release);

        // If this fails it means we are unlocking an unlocked lock
    #ifdef NDEBUG
        (void)lock_held;
    #else
        assert(lock_held);
    #endif
    }

    Mutex() = default;
    ~Mutex() = default;

private:
    std::atomic<bool> exclusive_{false};
};

// A very simple spin lock.
class CAPABILITY("mutex") SpinMutex {
public:
    // std::lock_guard<SpinMutex> wrapper.
    class SCOPED_CAPABILITY Lock {
    public:
        Lock(SpinMutex& m) ACQUIRE(m) : lock_(m) {}
        ~Lock() RELEASE() {}

    private:
        std::lock_guard<SpinMutex> lock_;
    };

    void lock() ACQUIRE() {
        int spins = 0;
        while (true) {
            auto old_val = 0;
            if (owner_.compare_exchange_weak(old_val, 1, std::memory_order_acq_rel)) {
                break;
            }

            // Help avoid complete resource starvation by yielding occasionally if
            // needed.
            if (++spins % 512 == 0) {
                std::this_thread::yield();
            } else {
                SpinLoopPause();
            }
        }
    }
    void unlock() RELEASE()  {
        owner_.store(0, std::memory_order_release);
    }

    SpinMutex() = default;
    ~SpinMutex() = default;

private:
    std::atomic<int> owner_{0};

    inline void SpinLoopPause() {
#if !defined(__arm__) && !defined(__aarch64__) && !defined(_M_ARM) && \
    !defined(_M_ARM64)
        _mm_pause();
#endif
    }
};
