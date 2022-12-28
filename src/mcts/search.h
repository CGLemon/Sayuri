#pragma once

#include "mcts/time_control.h"
#include "mcts/parameters.h"
#include "mcts/node.h"
#include "mcts/rollout.h"
#include "game/game_state.h"
#include "neural/training.h"
#include "utils/threadpool.h"
#include "utils/operators.h"
#include "utils/time.h"

#include <thread>
#include <memory>
#include <atomic>
#include <limits>

struct SearchResult {
public:
    SearchResult() = default;
    bool IsValid() const { return nn_evals_ != nullptr; }
    NodeEvals *GetEvals() const { return nn_evals_.get(); }

    void FromNetEvals(NodeEvals nn_evals) { 
        nn_evals_ = std::make_unique<NodeEvals>(nn_evals);
    }

    void FromRollout(GameState &state) {
        if (nn_evals_ == nullptr) {
            nn_evals_ = std::make_unique<NodeEvals>();
        }
        nn_evals_->black_wl = GetBlackRolloutResult(
                                  state,
                                  nn_evals_->black_ownership.data(),
                                  nn_evals_->black_final_score);
        nn_evals_->draw = 0.f;
    }

    void FromGameOver(GameState &state) {
        assert(state.GetPasses() >= 2);

        if (nn_evals_ == nullptr) {
            nn_evals_ = std::make_unique<NodeEvals>();
        }

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
        nn_evals_->black_final_score = black_final_score;

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
    int gumbel_move{kNullVertex};

    VertexType to_move;
    float komi;
    float root_eval;
    float root_final_score;
    float best_eval;

    std::vector<float> root_ownership;
    std::vector<int> root_visits;

    std::vector<float> root_playouts_dist;
    std::vector<float> target_playouts_dist;

    std::vector<std::vector<int>> alive_strings;
    std::vector<std::vector<int>> dead_strings;

    int movenum;
    int playouts;
    int threads;
    int batch_size;
    float seconds;
};

class Search {
public:
    static constexpr int kMaxPlayouts = std::numeric_limits<int>::max() / 2;

    enum OptionTag : int {
        kNullTag  = 0,
        kThinking = 1 << 1, // use time control
        kPonder   = 1 << 2, // thinking on opponent's time
        kAnalysis = 1 << 3, // use the analysis mode
        kForced   = 1 << 4, // remove all pass move before search
        kUnreused = 1 << 5, // don't reuse the tree
        kNoNoise  = 1 << 6  // disable any noise
    };

    // Enable OptionTag operations.
    ENABLE_FRIEND_BITWISE_OPERATORS_ON(OptionTag);

    Search(GameState &state, Network &network) : root_state_(state), network_(network) {
        Initialize();
    }
    ~Search();

    void Initialize();

    // Compute the result by monte carlo tree search.
    ComputationResult Computation(int playouts, OptionTag tag);

    // Get the best move.
    int ThinkBestMove();

    // Get the self-play move.
    int GetSelfPlayMove();

    // Will dump analysis information.
    int Analyze(bool ponder, AnalysisConfig &analysis_config);

    // Try to do the pondor.
    void TryPonder();

    // Set the time control.
    void TimeSettings(const int main_time,
                      const int byo_yomi_time,
                      const int byo_yomi_stones,
                      const int byo_yomi_periods);

    // Set time left.
    void TimeLeft(const int color, const int time, const int stones);

    // Save the self-play training data.
    void SaveTrainingBuffer(std::string filename, GameState &state);

    // Output the self-play training data.
    void GatherTrainingBuffer(std::vector<Training> &chunk, GameState &state);

    // Clear the training data in the buffer.
    void ClearTrainingBuffer();

    // Release the whole trees.
    void ReleaseTree();

private:
    // Try to reuse the sub-tree.
    bool AdvanceToNewRootState();

    bool InputPending(Search::OptionTag tag) const;

    void GatherComputationResult(ComputationResult &result) const;

    void GatherData(const GameState &state, ComputationResult &result);

    void PlaySimulation(GameState &currstate, Node *const node,
                        const int depth, SearchResult &search_result);

    void PrepareRootNode();
    int GetPonderPlayouts() const;

    int GetExpandThreshold(GameState &state) const;

    AnalysisConfig analysis_config_;

    // Stop the search if current playouts greater this value.
    int max_playouts_; 

    // Self-play training data.
    std::vector<Training> training_buffer_;

    // True if it is searhing.
    std::atomic<bool> running_; 

    // The current playouts.
    std::atomic<int> playouts_; 

    // The tree search time control. 
    TimeControl time_control_;

    // The current game state.
    GameState &root_state_;

    // The game state of last root node.
    GameState last_state_;

    // The forwarding network for this search.
    Network &network_;

    // The root node of tree.
    std::unique_ptr<Node> root_node_; 

    // The tree search parameters.
    std::unique_ptr<Parameters> param_;

    // The tree search threads.
    std::unique_ptr<ThreadGroup<void>> group_;
};
