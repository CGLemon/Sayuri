#pragma once

#include "game/game_state.h"
#include "game/types.h"
#include "mcts/search.h"
#include "utils/parser.h"

#include <vector>
#include <memory>
#include <mutex>

class Engine {
public:
    void Initialize();

    void SaveSgf(std::string filename, int g);
    void SaveTrainingData(std::string filename, int g);
    void PrepareGame(int g);
    void Selfplay(int g);

    int GetParallelGames() const;

private:
    void SetNormalGame(int g);
    void SetHandicapGame(int g);

    void SetFairKomi(int g);

    void Handel(int g);

    std::mutex io_mtx_;

    int parallel_games_;

    std::unique_ptr<Network> network_{nullptr};
    std::vector<std::unique_ptr<Search>> search_pool_;
    std::vector<GameState> game_pool_;
};
