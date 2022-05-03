#pragma once

#include "mcts/time_control.h"
#include "mcts/parameters.h"
#include "mcts/node.h"

#include "game/game_state.h"
#include "data/training.h"
#include "utils/threadpool.h"
#include "utils/time.h"

#include <thread>
#include <memory>
#include <atomic>

struct SearchResult {
public:
    SearchResult() = default;
    bool IsValid() const { return nn_evals_ != nullptr; }
    NodeEvals *GetEvals() const { return nn_evals_.get(); }

    void FromNetEvals(NodeEvals nn_evals) { 
        nn_evals_ = std::make_unique<NodeEvals>(nn_evals);
    }

    void FromGameover(GameState &state) {
        if (nn_evals_ == nullptr) {
            nn_evals_ = std::make_unique<NodeEvals>();
        }

        assert(state.GetPasses() >= 2);

        auto black_score = 0;
        auto ownership = state.GetOwnership();

        for (int idx = 0; idx < (int)ownership.size(); ++idx) {
            auto owner = ownership[idx];
            if (owner == kBlack) {
                black_score += 1;
                nn_evals_->black_ownership[idx] = 1;
            } else if (owner == kWhite) {
                black_score -= 1;
                nn_evals_->black_ownership[idx] = -1;
            } else {
                nn_evals_->black_ownership[idx] = 0;
            }
        }

        auto black_final_score = (float)black_score - state.GetKomi();
        nn_evals_-> black_final_score = black_final_score;

        if (black_final_score > 1e-4) {
            nn_evals_->black_wl = 1.0f;
            nn_evals_->draw = 0.0f;
        } else if (black_final_score < -1e-4) {
            nn_evals_->black_wl = 0.0f;
            nn_evals_->draw = 0.0f;
        } else {
            nn_evals_->black_wl = 0.5f;
            nn_evals_->draw = 1.0f;
        }
    }

private:
    std::unique_ptr<NodeEvals> nn_evals_{nullptr};
};

struct ComputationResult {
    int board_size;
    int best_move{kNullVertex};
    int random_move{kNullVertex};

    VertexType to_move;
    float komi;
    float root_eval;
    float root_final_score;

    std::vector<float> root_ownership;
    std::vector<float> root_probabilities;
    std::vector<int> root_visits;

    std::vector<float> target_probabilities;

    std::vector<std::vector<int>> alive_strings;
    std::vector<std::vector<int>> dead_strings;

    int movenum;
};

class Search {
public:
    static constexpr int kMaxPlayouts = 15000000;

    enum OptionTag : int {
        kNullTag  = 0,
        kThinking = 1 << 1,
        kPonder   = 1 << 2,
        kAnalyze  = 1 << 3
    };

    Search(GameState &state, Network &network) : root_state_(state), network_(network) {
        Initialize();
    }
    ~Search();

    void Initialize();

    // Compute the result by monte carlo tree search.
    ComputationResult Computation(int playouts, int interval, OptionTag tag);

    // Get the best move.
    int ThinkBestMove();

    // Get the self play move.
    int GetSelfPlayMove();

    int Analyze(int interval, bool ponder);

    void TryPonder();

    // Set the time control.
    void TimeSettings(const int main_time,
                      const int byo_yomi_time,
                      const int byo_yomi_stones,
                      const int byo_yomi_periods);

    // Set time left.
    void TimeLeft(const int color, const int time, const int stones);

    void SaveTrainingBuffer(std::string filename, GameState &state);

private:
    bool InputPending(Search::OptionTag tag) const;

    void GatherComputationResult(ComputationResult &result) const;

    void GatherData(const GameState &state, ComputationResult &result);

    void PlaySimulation(GameState &currstate, Node *const node,
                        Node *const root_node, SearchResult &search_result);

    std::vector<float> PrepareRootNode();
    void ClearNodes();

    int threads_;
    int max_playouts_; 

    std::vector<Training> training_buffer_;

    std::atomic<bool> running_; 
    std::atomic<int> playouts_; 

    TimeControl time_control_;

    GameState &root_state_;

    Network &network_;

    std::unique_ptr<NodeData> node_data_;
    std::unique_ptr<NodeStats> node_stats_;
    std::unique_ptr<Node> root_node_; 

    std::unique_ptr<Parameters> param_;
    std::unique_ptr<ThreadGroup<void>> group_;
};
