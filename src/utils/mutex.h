#ifndef UTILS_MUTEX_H_INCLUDE
#define UTILS_MUTEX_H_INCLUDE

#include <mutex>
#include <atomic>
#include <cassert>
#include <thread>

class SMP {
public:
    // Get number of cpu cores.
    static size_t GetNumberOfCores() {
        return std::thread::hardware_concurrency();
    }

private:
    std::atomic<bool> lock_;
    std::atomic<bool>& Get() { return lock_; }

    // Set lock state.
    void Set(bool state) {
        lock_.store(state);
    }
    friend class Mutex;
};

class Mutex {
public:
    class Lock {
    public:
        Lock(Mutex& m) : exclusive_(m.Get()) {
            lock();
        }
        ~Lock() {
            // If we don't claim to hold the lock,
            // don't bother trying to unlock in the destructor.
            if (owns_lock_) {
                unlock();
            }
        }

    private:
        void lock() {
            assert(!owns_lock_);
            // Test and Test-and-Set reduces memory contention
            // However, just trying to Test-and-Set first improves performance in almost
            // all cases
            while (exclusive_.exchange(true, std::memory_order_acquire)) {
                while (exclusive_.load(std::memory_order_relaxed));
            }
            owns_lock_ = true;
        }

        void unlock() {
            assert(owns_lock_);
            auto lock_held = exclusive_.exchange(false, std::memory_order_release);

            // If this fails it means we are unlocking an unlocked lock
        #ifdef NDEBUG
            (void)lock_held;
        #else
            assert(lock_held);
        #endif
            owns_lock_ = false;
        }

        bool owns_lock_{false};
        std::atomic<bool> & exclusive_;
    }; 

    Mutex() { exclusive_lock_.Set(false); }
    ~Mutex() = default;

    std::atomic<bool>& Get() { return exclusive_lock_.Get(); }
private:
    SMP exclusive_lock_;
};


#endif

