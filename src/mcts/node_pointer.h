#pragma once

#include <atomic>
#include <vector>
#include <cstdint>
#include <thread>

#define POINTER_MASK (3ULL)

static constexpr std::uint64_t kUninflated = 2ULL;
static constexpr std::uint64_t kInflating  = 1ULL;
static constexpr std::uint64_t kPointer    = 0ULL;

template<typename NodeType, typename DataType>
class NodePointer {
public:
    NodePointer() = default;
    NodePointer(DataType data);
    NodePointer(NodePointer &&n);
    NodePointer(const NodePointer &) = delete;
    NodePointer& operator=(NodePointer&&);

    ~NodePointer() = default;

    bool IsPointer() const;
    bool IsInflating() const;
    bool IsUninflated() const;

    NodeType *ReadPointer(uint64_t v) const;
    NodeType *Get() const;

    bool Inflate();
    bool Release();

    DataType* Data();

private:
    DataType data_;
    std::atomic<std::uint64_t> pointer_{kUninflated};

    bool IsPointer(std::uint64_t v) const;
    bool IsInflating(std::uint64_t v) const;
    bool IsUninflated(std::uint64_t v) const;
};

template<typename NodeType, typename DataType>
inline NodePointer<NodeType, DataType>::NodePointer(DataType data) {
    data_ = data;
}

template<typename NodeType, typename DataType>
inline NodePointer<NodeType, DataType>::NodePointer(NodePointer &&n) {
    data_ = n.data_;
    pointer_.store(n.pointer_.load(std::memory_order_relaxed), std::memory_order_relaxed);
}

template<typename NodeType, typename DataType>
inline NodePointer<NodeType, DataType>& NodePointer<NodeType, DataType>::operator=(NodePointer&& n) {
    data_ = n.data_;
    pointer_.store(n.pointer_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    return *this;
}

template<typename NodeType, typename DataType>
inline bool NodePointer<NodeType, DataType>::IsPointer(std::uint64_t v) const {
    return (v & POINTER_MASK) == kPointer;
}

template<typename NodeType, typename DataType>
inline bool NodePointer<NodeType, DataType>::IsInflating(std::uint64_t v) const {
    return (v & POINTER_MASK) == kInflating;
}

template<typename NodeType, typename DataType>
inline bool NodePointer<NodeType, DataType>::IsUninflated(std::uint64_t v) const {
    return (v & POINTER_MASK) == kUninflated;
}

template<typename NodeType, typename DataType>
inline bool NodePointer<NodeType, DataType>::IsPointer() const {
    return IsPointer(pointer_.load(std::memory_order_relaxed));
}

template<typename NodeType, typename DataType>
inline bool NodePointer<NodeType, DataType>::IsInflating() const {
    return IsInflating(pointer_.load(std::memory_order_relaxed));
}

template<typename NodeType, typename DataType>
inline bool NodePointer<NodeType, DataType>::IsUninflated() const {
    return IsUninflated(pointer_.load(std::memory_order_relaxed));
}

template<typename NodeType, typename DataType>
inline NodeType *NodePointer<NodeType, DataType>::ReadPointer(uint64_t v) const {
    assert(IsPointer(v));
    return reinterpret_cast<NodeType *>(v & ~(POINTER_MASK));
}

template<typename NodeType, typename DataType>
inline NodeType *NodePointer<NodeType, DataType>::Get() const {
    auto v = pointer_.load();
    if (IsPointer(v))
        return ReadPointer(v);
    return nullptr;
}

template<typename NodeType, typename DataType>
inline bool NodePointer<NodeType, DataType>::Inflate() {

inflate_loop: // Try to allocate new memory for this node.

    if (IsPointer(pointer_.load())) {
        // Another thread already inflated the pointer.
        return false;
    }

    auto uninflated = kUninflated;
    if (!pointer_.compare_exchange_strong(uninflated, kInflating)) {
        // Fail to get the owner. Try it next time.
        goto inflate_loop;
    }
    auto new_pointer =
             reinterpret_cast<std::uint64_t>(new NodeType(data_)) | kPointer;
    auto old_pointer = pointer_.exchange(new_pointer);
#ifdef NDEBUG
    (void) old_pointer;
#endif
    assert(IsInflating(old_pointer));
    return true;

}

template<typename NodeType, typename DataType>
inline bool NodePointer<NodeType, DataType>::Release() {
    // Only support for one thread to release the node. 
    auto v = pointer_.load();
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

template<typename NodeType, typename DataType>
inline DataType *NodePointer<NodeType, DataType>::Data() {
    return &data_;
}
