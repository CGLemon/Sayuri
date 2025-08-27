#pragma once

#include "mcts/time_control.h"
#include "mcts/parameters.h"
#include "mcts/node.h"
#include "game/game_state.h"
#include "neural/training_data.h"
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

    void FromGameOver(Network &network, GameState &state) {
        assert(state.GetPasses() >= 2);

        if (nn_evals_ == nullptr) {
            nn_evals_ = std::make_unique<NodeEvals>();
        }
        auto ownership = state.GetOwnership();
        TryRecoverOwnershipMap(network, state, ownership);

        for (int idx = 0; idx < (int)ownership.size(); ++idx) {
            auto owner = ownership[idx];
            if (owner == kBlack) {
                nn_evals_->black_ownership[idx] = 1;
            } else if (owner == kWhite) {
                nn_evals_->black_ownership[idx] = -1;
            } else {
                nn_evals_->black_ownership[idx] = 0;
            }
        }

        nn_evals_->black_final_score =
            state.GetFinalScore(kBlack, ownership);

        if (nn_evals_->black_final_score > 1e-4) {
            nn_evals_->black_wl = 1.0f;
            nn_evals_->draw = 0.0f;
        } else if (nn_evals_->black_final_score < -1e-4) {
            nn_evals_->black_wl = 0.0f;
            nn_evals_->draw = 0.0f;
        } else {
            nn_evals_->black_wl = 0.5f;
            nn_evals_->draw = 1.0f;
        }
    }

private:
    void TryRecoverOwnershipMap(Network &network,
                                GameState &state,
                                std::vector<int> &ownership) {
        auto fork_state = state;
        while (fork_state.GetPasses() >= 2) {
            fork_state.UndoMove();
        }
        if (fork_state.GetScoringRule() == kArea) {
            return;
        }
        fork_state.SetRule(kArea);
        constexpr float kRawOwnershipThreshold = 0.8f;

        auto netlist = network.GetOutput(fork_state, Network::kRandom);
        auto color = fork_state.GetToMove();
        auto num_intersections = fork_state.GetNumIntersections();
        auto safe_area = fork_state.GetStrictSafeArea();

        for (int idx = 0; idx < num_intersections; ++idx) {
            float black_owner = netlist.ownership[idx]; // -1 ~ 1
            if (color == kWhite) {
                black_owner = 0.f - black_owner;
            }
            if (safe_area[idx]) {
                continue;
            }
            if (black_owner > kRawOwnershipThreshold) {
                ownership[idx] = kBlack;
            } else if (black_owner < -kRawOwnershipThreshold) {
                ownership[idx] = kWhite;
            } else {
                ownership[idx] = kEmpty;
            }
        }
    }

    std::unique_ptr<NodeEvals> nn_evals_{nullptr};
};

struct ComputationResult {
    int board_size;
    int best_move{kNullVertex};
    int best_no_pass_move{kNullVertex};
    int random_move{kNullVertex};
    int gumbel_move{kNullVertex};
    int gumbel_no_pass_move{kNullVertex};
    int capture_all_dead_move{kNullVertex};
    int high_priority_move{kNullVertex};

    VertexType to_move;
    float komi;
    float root_eval;
    float root_score_lead;
    float best_eval;
    float root_score_stddev;
    float root_eval_stddev;

    std::vector<float> root_ownership;
    std::vector<int> root_searched_visits;
    std::vector<float> root_estimated_q;
    std::vector<float> root_visits_dist;
    std::vector<float> target_policy_dist;

    std::vector<std::vector<int>> alive_strings;
    std::vector<std::vector<int>> dead_strings;

    int movenum;
    int visits;
    int playouts;
    int threads;
    int batch_size;
    float elapsed;

    float policy_kld;
    bool side_resign;
};

class Search {
public:
    static constexpr int kMaxPlayouts = std::numeric_limits<int>::max() / 2;

    enum OptionTag : int {
        kNullTag     = 0,
        kThinking    = 1 << 1, // use time control
        kPonder      = 1 << 2, // thinking on opponent's time
        kAnalysis    = 1 << 3, // use the analysis mode
        kForced      = 1 << 4, // remove double pass move before search
        kUnreused    = 1 << 5, // don't reuse the tree
        kNoExploring = 1 << 6, // disable any exploring setting
        kNoBuffer    = 1 << 7  // don't push data to training data buffer
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

    // The Computation function wrapper. Reture the best move only.
    int GetBestMove(int playouts, OptionTag tag);

    // Get the best move.
    int ThinkBestMove();

    // Will dump analysis information.
    int Analyze(bool ponder, AnalysisConfig &analysis_config);

    // Get the self-play move.
    int GetSelfPlayMove(OptionTag tag = Search::kNullTag);

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
    void SaveTrainingBuffer(std::string filename);

    // Output the self-play training data.
    void GatherTrainingBuffer(std::vector<TrainingData> &chunk);

    // Clear the training data in the buffer.
    void ClearTrainingBuffer();

    // Release the whole trees.
    void ReleaseTree();

    // For debug interface, show the selected a path.
    std::string GetDebugMoves(std::vector<int> moves);

    // Keep playing for territory scoring rule and update
    // the territory helpe.
    void UpdateTerritoryHelper();

    // Return search parameters table.
    Parameters *GetParams(bool no_exploring_param = false);

private:
    // Try to reuse the sub-tree.
    bool AdvanceToNewRootState(Search::OptionTag tag);

    // Reture false if there is only one reasonable move and
    // enable the time management.
    bool HaveAlternateMoves(const float elapsed, const float limit,
                            const int cap, Search::OptionTag tag);

    // Reture true if the root achieve visit cap or playout
    // cap.
    bool AchieveCap(const int cap, Search::OptionTag tag);

    bool StoppedByKldGain(ComputationResult &result, Search::OptionTag tag);

    int GetPlayoutsLeft(const int cap, Search::OptionTag tag);

    int GetPonderPlayouts() const;

    bool InputPending(Search::OptionTag tag) const;

    void UpdateComputationResultFast(ComputationResult &result) const;
    void UpdateComputationResult(ComputationResult &result) const;

    void GatherData(const GameState &state,
                    ComputationResult &result,
                    bool discard);

    void UpdateLagBuffer(float thinking_time, float buffer_effect);

    void PlaySinglePlayout();
    void PlaySimulation(GameState &currstate, Node *const node,
                        const int depth, SearchResult &search_result);

    void PrepareRootNode(ComputationResult &result, Search::OptionTag tag);

    AnalysisConfig analysis_config_;

    // Self-play training data.
    std::vector<TrainingData> training_data_buffer_;

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

    // The root networl eval.
    NodeEvals root_evals_;

    // The tree search parameters.
    std::unique_ptr<Parameters> param_;
    std::unique_ptr<Parameters> no_exploring_param_;

    // The tree search threads.
    std::unique_ptr<ThreadGroup<void>> group_;

    // KLD gain
    int prev_kldgain_visits_;
    std::vector<double> prev_kldgain_target_policy_;

    std::vector<float> root_raw_probabilities_;
};
