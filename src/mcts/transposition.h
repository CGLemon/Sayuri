#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>

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

private:
    static constexpr int kMinVisits = 1;

    struct Entry {
        Entry() : hash(0ULL), q(0.0), visits(0) {}
        std::uint64_t hash;
        float q;
        int visits;
    };

    Entry * GetEntry(std::uint64_t hash);

    std::vector<Entry> table_;
    size_t capacity_;
};
