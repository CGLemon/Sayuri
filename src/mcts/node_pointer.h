#pragma once

#include <atomic>
#include <vector>
#include <cstdint>
#include <thread>
#include <cstring>

#include "mcts/parameters.h"

#define POINTER_MASK (3ULL)

static constexpr std::uint64_t kUninflated = 2ULL;
static constexpr std::uint64_t kInflating  = 1ULL;
static constexpr std::uint64_t kPointer    = 0ULL;

static_assert(sizeof(float) == sizeof(std::uint32_t), "");

template<typename NodeType>
class NodePointer {
public:
    NodePointer() = default;

    explicit NodePointer(std::int16_t vertex, float policy);
    explicit NodePointer(NodePointer &&n);
    NodePointer& operator=(NodePointer&&);

    // Construct with left value. Forbid it because
    // we may release same memory again.
    NodePointer(const NodePointer &) = delete;

    ~NodePointer() = default;

    bool IsPointer() const;
    bool IsInflating() const;
    bool IsUninflated() const;

    NodeType * GetPointer() const;

    bool Inflate(Parameters *param);
    bool Release();

    int GetVertex() const;
    float GetPolicy() const;
    int GetVisits() const;

private:
    std::atomic<std::uint64_t> pointer_{kUninflated};

    NodeType * ReadPointer(uint64_t v) const;
    int ReadVertex(std::uint64_t v) const;
    float ReadPolicy(std::uint64_t v) const;

    bool IsPointer(std::uint64_t v) const;
    bool IsInflating(std::uint64_t v) const;
    bool IsUninflated(std::uint64_t v) const;
};


template<typename NodeType>
inline NodePointer<NodeType>::NodePointer(std::int16_t vertex, float policy) {
    std::uint64_t buf = 0ULL;

    std::memcpy((std::uint32_t *)(&buf) + 1, &policy, sizeof(float));
    std::memcpy((std::int16_t *)(&buf) + 1, &vertex, sizeof(std::int16_t));

    buf |= kUninflated;
    pointer_.store(buf, std::memory_order_relaxed);
}

template<typename NodeType>
inline NodePointer<NodeType>::NodePointer(NodePointer &&n) {
    // Construct with right value. It's pointer is
    // uninflated.
    auto v = n.pointer_.load(std::memory_order_relaxed);
    pointer_.store(v, std::memory_order_relaxed);

    // The original pointer should be raw.
    assert((v & POINTER_MASK) == kUninflated);
}

template<typename NodeType>
inline NodePointer<NodeType>& NodePointer<NodeType>::operator=(NodePointer&& n) {
    // Should we release the original pointer? I guess
    // it is not necessary. The 'std::remove_if' will use
    // the 'operator='. All pointers Should not be released
    // in this process.
    auto v = n.pointer_.load(std::memory_order_relaxed);
    pointer_.store(v, std::memory_order_relaxed);

    return *this;
}

template<typename NodeType>
inline bool NodePointer<NodeType>::IsPointer(std::uint64_t v) const {
    return (v & POINTER_MASK) == kPointer;
}

template<typename NodeType>
inline bool NodePointer<NodeType>::IsInflating(std::uint64_t v) const {
    return (v & POINTER_MASK) == kInflating;
}

template<typename NodeType>
inline bool NodePointer<NodeType>::IsUninflated(std::uint64_t v) const {
    return (v & POINTER_MASK) == kUninflated;
}

template<typename NodeType>
inline bool NodePointer<NodeType>::IsPointer() const {
    return IsPointer(pointer_.load(std::memory_order_relaxed));
}

template<typename NodeType>
inline bool NodePointer<NodeType>::IsInflating() const {
    return IsInflating(pointer_.load(std::memory_order_relaxed));
}

template<typename NodeType>
inline bool NodePointer<NodeType>::IsUninflated() const {
    return IsUninflated(pointer_.load(std::memory_order_relaxed));
}

template<typename NodeType>
inline NodeType *NodePointer<NodeType>::ReadPointer(uint64_t v) const {
    assert(IsPointer(v));
    return reinterpret_cast<NodeType *>(v & ~(POINTER_MASK));
}

template<typename NodeType>
inline NodeType *NodePointer<NodeType>::GetPointer() const {
    auto v = pointer_.load(std::memory_order_relaxed);
    if (IsPointer(v))
        return ReadPointer(v);
    return nullptr;
}

template<typename NodeType>
inline bool NodePointer<NodeType>::Inflate(Parameters *param) {

inflate_loop: // Try to allocate new memory for the pointer.

    auto v = pointer_.load(std::memory_order_relaxed);
    if (IsPointer(v)) {
        // Another thread had already inflated the pointer yet.
        return false;
    }

    // Erase the pointer type.
    v &= (~POINTER_MASK);

    // Try to fetch the owner.
    auto uninflated = v | kUninflated;
    if (!pointer_.compare_exchange_strong(uninflated, kInflating)) {
        // Fail to get the owner. Try to do it next time.
        goto inflate_loop;
    }

    // Fetch the data.
    const std::int16_t vertex = ReadVertex(v);
    const float policy = ReadPolicy(v);

    // Success to get the owner. Now allocate new memory.
    auto new_pointer =
             reinterpret_cast<std::uint64_t>(
                 new NodeType(param, vertex, policy)) | kPointer;
    auto old_pointer = pointer_.exchange(new_pointer);
#ifdef NDEBUG
    (void) old_pointer;
#endif
    assert(IsInflating(old_pointer));
    return true;

}

template<typename NodeType>
inline bool NodePointer<NodeType>::Release() {
    // Besure that only one thread can release the memory.
    auto v = pointer_.load(std::memory_order_relaxed);

    if (IsPointer(v)) {
        delete ReadPointer(v);
        auto pointer = pointer_.exchange(kUninflated);
#ifdef NDEBUG
        (void) pointer;
#endif
        assert(pointer == v);
        return true;
    }
    return false;
}

template<typename NodeType>
inline int NodePointer<NodeType>::ReadVertex(std::uint64_t v) const {
    std::int16_t res;
    std::memcpy(&res, (std::int16_t *)(&v) + 1, sizeof(std::int16_t));

    return res;
}

template<typename NodeType>
inline float NodePointer<NodeType>::ReadPolicy(std::uint64_t v) const {
    float res;
    std::memcpy(&res, (std::uint32_t *)(&v) + 1, sizeof(float));

    return res;
}

template<typename NodeType>
inline int NodePointer<NodeType>::GetVertex() const {
    auto v = pointer_.load(std::memory_order_relaxed);

    while (IsInflating(v)) {
        v = pointer_.load(std::memory_order_relaxed);
    }
    if (IsPointer(v)) {
        return ReadPointer(v)->GetVertex();
    }
    return ReadVertex(v);
}

template<typename NodeType>
inline float NodePointer<NodeType>::GetPolicy() const {
    auto v = pointer_.load(std::memory_order_relaxed);

    while (IsInflating(v)) {
        v = pointer_.load(std::memory_order_relaxed);
    }
    if (IsPointer(v)) {
        return ReadPointer(v)->GetPolicy();
    }
    return ReadPolicy(v);
}

template<typename NodeType>
inline int NodePointer<NodeType>::GetVisits() const {
    auto v = pointer_.load(std::memory_order_relaxed);

    while (IsInflating(v)) {
        v = pointer_.load(std::memory_order_relaxed);
    }
    if (IsPointer(v)) {
        return ReadPointer(v)->GetVisits();
    }
    return 0;
}
