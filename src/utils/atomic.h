#pragma once

#include <atomic>

template <typename T> 
void AtomicFetchAdd(std::atomic<T> &f, T d) {
    T old = f.load();
    while (!f.compare_exchange_weak(old, old + d)) {}
}
