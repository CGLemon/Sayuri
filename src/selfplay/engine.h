#pragma once

#include "game/game_state.h"
#include "game/types.h"
#include "mcts/search.h"
#include "utils/parser.h"

#include <vector>
#include <memory>

class Engine {
public:
    void Initialize();

    void SaveSgf(std::string filename, int g);
    void GatherTrainingData(std::vector<Training> &chunk, int g);
    void PrepareGame(int g);
    void Selfplay(int g);

    int GetParallelGames() const;

private:
    struct BoardQuery {
        int board_size;
        float komi;
        float prob;
    };

    void ParseQueries();
    void SetNormalGame(int g);
    void SetHandicapGame(int g);

    void SetFairKomi(int g);

    void Handel(int g);

    int parallel_games_;

    std::vector<BoardQuery> board_queries_;

    std::unique_ptr<Network> network_{nullptr};
    std::vector<std::unique_ptr<Search>> search_pool_;
    std::vector<GameState> game_pool_;
};
