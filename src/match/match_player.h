#pragma once

#include <vector>
#include <string>
#include <memory>

class PlayersPool {
public:
    struct Player {
        int elo;
        std::string name;
    };

    PlayersPool() = default;

    void Initialize();
    bool RandomMatch(PlayersPool::Player **a, PlayersPool::Player **b);

private:
    std::vector<std::unique_ptr<Player>> players_;
};
