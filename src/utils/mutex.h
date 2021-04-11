#pragma once

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

private:
    std::atomic<bool> lock_;

    // Get the lock reference.
    inline std::atomic<bool>& Get() { return lock_; }

    // Set lock state.
    void Set(bool state) {
        lock_.store(state, std::memory_order_relaxed);
    }
    friend class Mutex;
    friend class SpinMutex;
};

class Mutex {
public:
    class Lock {
    public:
        Lock(Mutex& m) : lock_(m) {}
        ~Lock() {}

    private:
        std::lock_guard<Mutex> lock_;
    };

    void lock() {
        // Test and Test-and-Set reduces memory contention
        // However, just trying to Test-and-Set first improves performance in almost
        // all cases
        while (Get().exchange(true, std::memory_order_acquire)) {
            while (Get().load(std::memory_order_relaxed));
        }
    }

    void unlock() {
        auto lock_held = Get().exchange(false, std::memory_order_release);

        // If this fails it means we are unlocking an unlocked lock
    #ifdef NDEBUG
        (void)lock_held;
    #else
        assert(lock_held);
    #endif
    }

    Mutex() { exclusive_lock_.Set(false); }
    ~Mutex() = default;

private:
    inline std::atomic<bool>& Get() { return exclusive_lock_.Get(); }

    SMP exclusive_lock_;
};

// A very simple spin lock.
class SpinMutex {
public:
  // std::unique_lock<SpinMutex> wrapper.
    class Lock {
    public:
        Lock(SpinMutex& m) : lock_(m) {}
        ~Lock() {}

    private:
        std::unique_lock<SpinMutex> lock_;
    };

    void lock() {
        int spins = 0;
        while (true) {
            auto val = false;
            if (Get().compare_exchange_weak(val, true, std::memory_order_acq_rel)) {
                break;
            }
            ++spins;
            // Help avoid complete resource starvation by yielding occasionally if
            // needed.
            if (spins % 512 == 0) {
                std::this_thread::yield();
            } else {
                SpinloopPause();
            }
        }
    }
    void unlock() {
        Get().store(false, std::memory_order_release);
    }

    SpinMutex() { exclusive_lock_.Set(false); }
    ~SpinMutex() = default;

private:
    inline std::atomic<bool>& Get() { return exclusive_lock_.Get(); }

    SMP exclusive_lock_;


    inline void SpinloopPause() {
#if !defined(__arm__) && !defined(__aarch64__) && !defined(_M_ARM) && \
    !defined(_M_ARM64)
        _mm_pause();
#endif
    }
};
