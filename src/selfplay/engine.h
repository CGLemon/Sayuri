#pragma once

#include "game/game_state.h"
#include "game/types.h"
#include "mcts/search.h"

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
    size_t GetNetReportQueries();

private:
    struct BoardQuery {
        int board_size;
        float komi;
        float prob;
    };
    struct HandicapQuery {
        int board_size;
        int handicaps;
        float probabilities;
    };

    std::string SelectWeights() const;

    void ParseQueries();
    void SetNormalGame(int g);
    void SetHandicapGame(int g, int handicaps);
    void SetRandomOpeningGame(int g);

    void SetUnfairKomi(int g);
    void SetFairKomi(int g);
    int GetHandicaps(int g);

    void Handel(int g);

    float komi_stddev_;
    float komi_big_stddev_;
    float komi_big_stddev_prob_;
    float handicap_fair_komi_prob_;
    float random_opening_prob_;
    float random_moves_factor_;
    float random_opening_temp_;
    int default_playouts_;
    int parallel_games_;

    std::vector<BoardQuery> board_queries_;
    std::vector<HandicapQuery> handicap_queries_;
    std::vector<ScoringRuleType> scoring_set_;

    std::unique_ptr<Network> network_{nullptr};
    std::vector<std::unique_ptr<Search>> search_pool_;
    std::vector<GameState> game_pool_;

    size_t last_net_accm_queries_;
};
