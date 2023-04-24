#pragma once

#include "utils/mutex.h"
#include "utils/cppattributes.h"

#include <memory>
#include <algorithm>
#include <vector>
#include <deque>
#include <unordered_map>

// Generic FIFO cache. Thread-safe.
template<typename V>
class FifoCache {
public:
    FifoCache() {
        capacity_ = 0;
        allocated_ = 0;
    }

    FifoCache(size_t capacity) {
        capacity_ = capacity;
        allocated_ = 0;
    }

    FifoCache(FifoCache&& cache) {
        capacity_ = cache.capacity_;
        allocated_ = cache.allocated_;
    }

    // Set the capacity.
    void SetCapacity(size_t size);

    // Insert the new item to the cache.
    void Insert(std::uint64_t key, V value);

    // Lookup the item and pin the item. The pined item is
    // not allowed to release.
    V* LookupAndPin(std::uint64_t key);

    // Only lookup the item.
    V* LookupItem(std::uint64_t key);

    // Unpin the item to the cache.
    void Unpin(std::uint64_t key);

    void Clear();

    size_t GetEntrySize() const;

private:
    struct Entry {
        std::unique_ptr<V> value;
        int pines;
    };

    static constexpr size_t kEntrySize = sizeof(Entry) + sizeof(V);

    void Evict() REQUIRES(mutex_);

    SpinLock mutex_;

    std::unordered_map<uint64_t, Entry> lookup_ GUARDED_BY(mutex_);
    std::deque<uint64_t> order_ GUARDED_BY(mutex_);
    std::deque<uint64_t> evicted_ GUARDED_BY(mutex_);

    size_t allocated_ GUARDED_BY(mutex_);
    size_t capacity_ GUARDED_BY(mutex_);
};

template<typename V>
void FifoCache<V>::SetCapacity(size_t size) {
    SpinLock::Lock lock(mutex_);

    capacity_ = size;
    while (allocated_ > capacity_) {
        Evict();
    }
}

template<typename V>
void FifoCache<V>::Insert(std::uint64_t key, V value) {
    SpinLock::Lock lock(mutex_);

    auto it = lookup_.find(key);
    if (it != std::end(lookup_)) {
        // Had existed.
        return;
    }
    auto entry = Entry{};
    entry.value = std::make_unique<V>(value);
    entry.pines = 0;

    lookup_.insert({key, std::move(entry)});
    order_.emplace_back(key);
    allocated_++;

    while (allocated_ > capacity_) {
        Evict();
    }
}

template<typename V>
V* FifoCache<V>::LookupAndPin(std::uint64_t key) {
    SpinLock::Lock lock(mutex_);

    auto it = lookup_.find(key);
    if (it == std::end(lookup_)) {
        // Not found.
        return nullptr;
    }
    auto &entry = it->second;
    entry.pines++;

    return entry.value.get();
}

template<typename V>
V* FifoCache<V>::LookupItem(std::uint64_t key) {
    SpinLock::Lock lock(mutex_);

    auto it = lookup_.find(key);
    if (it == std::end(lookup_)) {
        // Not found.
        return nullptr;
    }
    auto &entry = it->second;

    return entry.value.get();
}

template<typename V>
void FifoCache<V>::Unpin(std::uint64_t key) {
    SpinLock::Lock lock(mutex_);

    auto it = lookup_.find(key);
    if (it != std::end(lookup_)) {
        auto &entry = it->second;
        entry.pines--;
        assert(entry.pines >= 0);

        auto evicted = std::find(std::begin(evicted_), std::end(evicted_), key);
        if (entry.pines == 0 &&
                std::end(evicted_) != evicted) {
            evicted_.erase(evicted);
            lookup_.erase(it);
        }

        return;
    }
}

template<typename V>
void FifoCache<V>::Evict() {
    if (allocated_ == 0) return;

    auto key = order_.front();
    auto it = lookup_.find(key);

    auto &entry = it->second;
    if (entry.pines > 0) {
        evicted_.emplace_back(key);
    } else {
        lookup_.erase(key);
    }
    order_.pop_front();
    allocated_--;
}

template<typename V>
void FifoCache<V>::Clear() {
    while (allocated_ > 0) {
        Evict();
    }
}

template<typename V>
size_t FifoCache<V>::GetEntrySize() const {
    return kEntrySize;
}

template<typename V>
bool LookupCache(FifoCache<V> &cache, std::uint64_t key, V& val) {
    auto result = cache.LookupItem(key);
    if (result) {
        val = *result;
        return true;
    }
    return false;
}
