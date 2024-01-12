#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>
#include <atomic>

#include "game/types.h"

class Transposition {
public:
    Transposition();
    Transposition(size_t capacity);

    void SetCapacity(size_t size);

    float Lookup(std::uint64_t hash, float q, int visits, int color);

    float Update(std::uint64_t hash, float eval, float q, int visits);

    void Clear();

    size_t GetEntrySize() const;

    size_t GetCapacity() const;

    void UpdateRootVisits(int v);

private:
    int GetMinVisits() const;

    struct Entry {
        Entry() : hash(0ULL), q(0.0), visits(0) {}
        std::uint64_t hash;
        float q;
        int visits;
    };

    Entry * GetEntry(std::uint64_t hash);

    std::vector<Entry> table_;
    size_t capacity_;
    std::atomic<int> root_visits_{0};
};
