#pragma once

#include <atomic>

template <typename T>
void AtomicFetchAdd(std::atomic<T> &f, T d,
                    std::memory_order order = std::memory_order_relaxed) {
    T old = f.load(std::memory_order_relaxed);
    while (!f.compare_exchange_weak(old, old + d, order)) {}
}
