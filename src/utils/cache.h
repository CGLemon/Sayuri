#pragma once

#include "utils/mutex.h"
#include "cppattributes.h"

#include <memory>
#include <algorithm>
#include <vector>
#include <deque>
#include <unordered_map>

// Generic LRU cache. Thread-safe.
template<typename V>
class LruCache {
public:
    LruCache() {
        capacity_ = 0;
        allocated_ = 0;
    }

    LruCache(size_t capacity) {
        capacity_ = capacity;
        allocated_ = 0;
    }

    LruCache(LruCache&& cache) {
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

private:
    void Evict() REQUIRES(mutex_);

    SpinMutex mutex_;

    struct Entry {
        std::unique_ptr<V> value;
        int pines;
    };
    std::unordered_map<uint64_t, Entry> lookup_ GUARDED_BY(mutex_);
    std::deque<uint64_t> order_ GUARDED_BY(mutex_);
    std::deque<uint64_t> evicted_ GUARDED_BY(mutex_);

    size_t allocated_ GUARDED_BY(mutex_);
    size_t capacity_ GUARDED_BY(mutex_);
};

template<typename V>
void LruCache<V>::SetCapacity(size_t size) {
    SpinMutex::Lock lock(mutex_);

    capacity_ = size;
    while (allocated_ > capacity_) {
        Evict();
    }
}

template<typename V>
void LruCache<V>::Insert(std::uint64_t key, V value) {
    SpinMutex::Lock lock(mutex_);

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
V* LruCache<V>::LookupAndPin(std::uint64_t key) {
    SpinMutex::Lock lock(mutex_);

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
V* LruCache<V>::LookupItem(std::uint64_t key) {
    SpinMutex::Lock lock(mutex_);

    auto it = lookup_.find(key);
    if (it == std::end(lookup_)) {
        // Not found.
        return nullptr;
    }
    auto &entry = it->second;

    return entry.value.get();
}

template<typename V>
void LruCache<V>::Unpin(std::uint64_t key) {
    SpinMutex::Lock lock(mutex_);

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
void LruCache<V>::Evict() {
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
void LruCache<V>::Clear() {
    while (allocated_ > 0) {
        Evict();
    }
}

template<typename V>
bool LookupCache(LruCache<V> &cache, std::uint64_t key, V& val) {
    auto result = cache.LookupItem(key);
    if (result) {
        val = *result;
        return true;
    }
    return false;
}
