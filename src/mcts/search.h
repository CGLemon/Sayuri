#include "mcts/time_control.h"
#include "mcts/parameters.h"
#include "mcts/node.h"

#include "game/game_state.h"
#include "utils/threadpool.h"
#include "utils/time.h"

#include <thread>
#include <memory>
#include <atomic>

struct SearchResult {
public:
    SearchResult() = default;
    bool IsValid() const { return nn_evals_ != nullptr; }
    std::shared_ptr<NodeEvals> GetEvals() const { return nn_evals_; }

    void FromNetEvals(NodeEvals nn_evals) { 
        nn_evals_ = std::make_shared<NodeEvals>(nn_evals);
    }

    void FromGameover(GameState &state) {
        if (nn_evals_ == nullptr) {
            nn_evals_ = std::make_shared<NodeEvals>();
        }

        assert(state.GetPasses() >= 2);

        auto black_score = 0;
        auto ownership = state.GetOwnership(20);

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
    std::shared_ptr<NodeEvals> nn_evals_{nullptr};
};

struct ComputationResult {
    int best_move{kNullVertex};
    float root_eval;
    float final_score;
    std::vector<float> ownership;
};

class Search {
public:
    static constexpr int kMaxPlayouts = 15000000;

    Search(GameState &state, Network &network) : root_state_(state), network_(network) {
        Initialize();
    }
    ~Search();

    void Initialize();

    ComputationResult Computation(int playouts);

    int ThinkBestMove();

    void TimeSettings(const int main_time,
                      const int byo_yomi_time,
                      const int byo_yomi_stones);

    void TimeLeft(const int color, const int time, const int stones);


private:
    void PlaySimulation(GameState &currstate, Node *const node,
                        Node *const root_node, SearchResult &search_result);

    void PrepareRootNode();
    void ClearNodes();

    int threads_;
    int max_playouts_; 

    std::atomic<bool> running_; 
    std::atomic<int> playouts_; 

    TimeControl time_control_;

    GameState &root_state_;

    Network &network_;

    std::shared_ptr<NodeStats> node_stats_;
    std::shared_ptr<Node> root_node_; 

    std::shared_ptr<Parameters> param_;
    std::unique_ptr<ThreadGroup<void>> group_;
};
