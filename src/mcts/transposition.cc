#include "mcts/transposition.h"

#include <algorithm>

Transposition::Transposition() {
    SetCapacity(0);
}

Transposition::Transposition(size_t capacity) {
    SetCapacity(capacity);
}

void Transposition::SetCapacity(size_t size) {
    capacity_ = size;
    table_.resize(capacity_);
}

float Transposition::Mix(std::uint64_t hash, float q, int visits, int color) {
    if (capacity_ == 0) {
        return q;
    }

    // catch the entry
    Entry e = *GetEntry(hash);
    if (e.hash != hash) {
        return q;
    }
    if (color != kBlack) {
        e.q = 1.f - e.q;
    }

    double update_q = e.q;
    double update_visits = e.visits;
    double mix_q = q + update_visits *
        (update_q - q) / (visits + update_visits);

    return mix_q;
}

void Transposition::Update(std::uint64_t hash, float q, int visits) {
    if (capacity_ == 0) {
        return;
    }

    Entry * e = GetEntry(hash);
    if (visits < kMinVisits) {
        return;
    }
    if (e->hash == hash && e->visits > visits) {
        return;
    }
    e->hash = hash;
    e->q = q;
    e->visits = visits;
}

void Transposition::Clear() {
    std::for_each(std::begin(table_), std::end(table_),
                     [](auto &e) {
                         e = Entry{};
                     }
                 );
}

Transposition::Entry * Transposition::GetEntry(std::uint64_t hash) {
    const auto idx = hash % capacity_;
    Entry * e = table_.data() + idx;
    return e;
}

size_t Transposition::GetEntrySize() const {
    return sizeof(Entry);
}

size_t Transposition::GetCapacity() const {
    return capacity_;
}
