#include "mcts/transposition.h"

#include <algorithm>
#include <cmath>

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

float Transposition::Lookup(std::uint64_t hash, float q, int visits, int color) {
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

    double lookup_q = e.q;
    double lookup_visits = e.visits;
    double mix_q = q + lookup_visits *
        (lookup_q - q) / (visits + lookup_visits);

    return mix_q;
}

float Transposition::Update(std::uint64_t hash, float eval, float q, int visits) {
    // assume the color is black.

    if (capacity_ == 0) {
        return eval;
    }
    if (visits < GetMinVisits()) {
        return eval;
    }

    Entry * e = GetEntry(hash);
    if (e->hash == hash && e->visits > visits) {
        double diff = std::sqrt((double)(e->visits) - visits);
        double update_q = e->q;
        double mix_eval = ((double)eval * visits + update_q * diff)/(visits + diff);
        return mix_eval;
    }
    e->hash = hash;
    e->q = q;
    e->visits = visits;
    return eval;
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

void Transposition::UpdateRootVisits(int v) {
    root_visits_.store(v, std::memory_order_relaxed);
}

int Transposition::GetMinVisits() const {
    // TODO: We should tune these value.
    int v = root_visits_.load(std::memory_order_relaxed);
    int c = static_cast<int>(capacity_);
    if (v >= c) {
        return v;
    }
    int half_c = 0.5 * c; // half space is buffer
    int remaining = std::max(1, half_c - v);
    return 1 + std::min(half_c / remaining, 100);
}
