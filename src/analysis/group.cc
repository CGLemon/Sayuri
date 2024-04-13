#include "analysis/group.h"

Group::Item &Group::GetItem(std::string id, Network &network) {
    std::lock_guard<std::mutex> lk(mtx_);

    if (pool_.find(id) == std::end(pool_)) {
        pool_[id] = std::make_unique<Item>();
        pool_[id]->Assgin(id, network);
    }
    Item &item = *pool_[id];
    item.pinned.store(true, std::memory_order_relaxed);
    return item;
}

void Group::UnpinItem(std::string id) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (pool_.find(id) == std::end(pool_)) {
        return;
    }
    pool_[id]->pinned.store(false, std::memory_order_relaxed);
}

bool Group::RemoveItem(std::string id) {
    std::lock_guard<std::mutex> lk(mtx_);

    if (pool_.find(id) == std::end(pool_)) {
        return false;
    }
    Item &item = *pool_[id];
    if (item.pinned.load(std::memory_order_relaxed)) {
        return false;
    }

    pool_.erase(id);
    return true;
}

bool Group::IsExistence(std::string id) {
    std::lock_guard<std::mutex> lk(mtx_);
    return pool_.find(id) != std::end(pool_);
}
