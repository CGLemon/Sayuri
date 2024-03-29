#pragma once

#include "utils/mutex.h"

#include <cstdint>
#include <memory>
#include <algorithm>
#include <vector>

template<typename V>
class HashKeyCache {
public:
    HashKeyCache() {
        capacity_ = 0;
        generation_ = 0;
    }

    HashKeyCache(size_t capacity) {
        SetCapacity(capacity);
        generation_ = 0;
    }

    HashKeyCache(HashKeyCache&& cache) {
        SetCapacity(cache.capacity_);
        Clear();
        generation_ = cache.generation_;
    }

    // Set the capacity.
    void SetCapacity(size_t size);

    // Insert the new item to the cache.
    void Insert(std::uint64_t key, const V& value);

    // Lookup the item.
    bool LookupItem(std::uint64_t key, V& val);

    // Clear the hash table.
    void Clear();

    size_t GetEntrySize() const;

private:
    struct Entry {
        Entry() : generation{0} {}
        std::uint64_t key;
        std::uint64_t generation;
        std::unique_ptr<V> pointer;
    };

    SpinLock mutex_;

    static constexpr size_t kClusterSize = 8;
    static constexpr size_t kEntrySize = sizeof(Entry) + sizeof(V);

    std::vector<Entry> table_ GUARDED_BY(mutex_);

    size_t capacity_ GUARDED_BY(mutex_);
    size_t blocks_ GUARDED_BY(mutex_);
    std::uint64_t generation_ GUARDED_BY(mutex_);
};

template<typename V>
void HashKeyCache<V>::SetCapacity(size_t size) {
    if (size % kClusterSize) {
        auto t = size / kClusterSize;
        size = (t+1) * kClusterSize;
    }

    SpinLock::Lock lock(mutex_);

    blocks_ = size / kClusterSize;
    capacity_ = size;
    table_.resize(size);
    table_.shrink_to_fit();
}

template<typename V>
void HashKeyCache<V>::Insert(std::uint64_t key, const V& value) {
    const auto idx = (key % blocks_) * kClusterSize;
    Entry *entry = table_.data() + idx;

    SpinLock::Lock lock(mutex_);

    size_t min_i = 0;
    size_t min_g = entry->generation;
    for (size_t offset = 1; offset < kClusterSize; ++offset) {
        Entry *e = entry + offset;
        if (min_g > e->generation) {
            min_g = e->generation;
            min_i = offset;
        }
    }

    ++generation_;

    Entry *new_entry = entry + min_i;
    new_entry->key = key;
    new_entry->generation = generation_;
    new_entry->pointer = std::make_unique<V>(value);
}

template<typename V>
bool HashKeyCache<V>::LookupItem(std::uint64_t key, V& val) {
    const auto idx = (key % blocks_) * kClusterSize;
    Entry *entry = table_.data() + idx;

    SpinLock::Lock lock(mutex_);

    for (size_t offset = 0; offset < kClusterSize; ++offset) {
        Entry *e = entry + offset;
        if (e->key == key && e->generation != 0) {
            val = *(e->pointer.get());
            return true;
        }
    }

    return false;
}

template<typename V>
void HashKeyCache<V>::Clear() {
    SpinLock::Lock lock(mutex_);

    generation_ = 0;
    std::for_each(std::begin(table_), std::end(table_),
                     [](auto &e) {
                         e.generation = 0;
                         e.pointer.reset(nullptr);
                     }
                 );
}

template<typename V>
size_t HashKeyCache<V>::GetEntrySize() const {
    return kEntrySize;
}
