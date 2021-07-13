#pragma once

#include <memory>
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
    NodePointer(std::shared_ptr<DataType> data);
    NodePointer(NodePointer &&n);
    NodePointer(const NodePointer &) = delete;
    NodePointer& operator=(const NodePointer&);

    ~NodePointer();

    bool IsPointer() const;
    bool IsInflating() const;
    bool IsUninflated() const;

    NodeType *ReadPointer(uint64_t v) const;
    NodeType *Get() const;

    bool Inflate();
    bool Release();

    std::shared_ptr<DataType> Data() const;

private:
    bool AcquireInflating();

    std::shared_ptr<DataType> data_{nullptr};
    std::atomic<std::uint64_t> pointer_{kUninflated};

    bool IsPointer(std::uint64_t v) const;
    bool IsInflating(std::uint64_t v) const;
    bool IsUninflated(std::uint64_t v) const;
};

template<typename NodeType, typename DataType>
inline NodePointer<NodeType, DataType>::NodePointer(std::shared_ptr<DataType> data) {
    data_ = data;
}

template<typename NodeType, typename DataType>
inline NodePointer<NodeType, DataType>::NodePointer(NodePointer &&n) {
    data_ = n.data_;
}


template<typename NodeType, typename DataType>
inline NodePointer<NodeType, DataType>::~NodePointer() {
    Release();
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
    return IsPointer(pointer_.load());
}

template<typename NodeType, typename DataType>
inline bool NodePointer<NodeType, DataType>::IsInflating() const {
    return IsInflating(pointer_.load());
}

template<typename NodeType, typename DataType>
inline bool NodePointer<NodeType, DataType>::IsUninflated() const {
    return IsUninflated(pointer_.load());
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
inline bool NodePointer<NodeType, DataType>::AcquireInflating() {
    auto uninflated = kUninflated;
    return pointer_.compare_exchange_strong(uninflated, kInflating);
}

template<typename NodeType, typename DataType>
inline bool NodePointer<NodeType, DataType>::Inflate() {
    while (true) {
        if (IsPointer(pointer_.load())) {
            return false;
        }
        if (!AcquireInflating()) {
            continue;
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
}

template<typename NodeType, typename DataType>
inline bool NodePointer<NodeType, DataType>::Release() {
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
inline std::shared_ptr<DataType> NodePointer<NodeType, DataType>::Data() const {
    return data_;
}
