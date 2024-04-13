#pragma once

#include "game/game_state.h"
#include "game/types.h"
#include "mcts/search.h"
#include "neural/network.h"
#include "config.h"

#include <memory>
#include <mutex>
#include <unordered_map>
#include <string>

class Group {
public:
    struct Item {
        Item() = default;

        void Assgin(std::string id, Network &network) {
            this->id = id;
            state.Reset(GetOption<int>("defualt_boardsize"),
                            GetOption<float>("defualt_komi"),
                            GetOption<int>("scoring_rule"));
            search = std::make_unique<Search>(state, network);
            pinned.store(false, std::memory_order_relaxed);
        }

        std::atomic<bool> pinned;
        std::string id;
        GameState state;
        std::unique_ptr<Search> search; 
    };

    Item &GetItem(std::string id, Network &network);
    void UnpinItem(std::string id);
    bool RemoveItem(std::string id);
    bool IsExistence(std::string id);

private:
    std::unordered_map<std::string, std::unique_ptr<Item>> pool_;
    std::mutex mtx_;
};
