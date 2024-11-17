#pragma once

#include "game/game_state.h"
#include "game/types.h"
#include "mcts/node_pointer.h"
#include "mcts/parameters.h"
#include "neural/network.h"

#include <array>
#include <vector>
#include <atomic>
#include <string>
#include <mutex>

struct NodeEvals {
    float black_final_score{0.0f};
    float black_wl{0.0f};
    float draw{0.0f};
    std::array<float, kNumIntersections> black_ownership;
};

struct AnalysisConfig {
    enum OutputFormat {
        kSayuri,
        kKata,
        kLeela
    };

    struct MoveToAvoid{
        int vertex{kNullVertex}, color{kInvalid}, until_move{-1};
        bool Valid() const {
            return vertex != kNullVertex &&
                       color != kInvalid &&
                       until_move >= 1;
        }
    };

    OutputFormat output_format{kLeela};
    bool ownership{false};
    bool moves_ownership{false};

    int interval{0};
    int min_moves{0};
    int max_moves{kPotentialMoves};

    std::vector<MoveToAvoid> avoid_moves;
    std::vector<MoveToAvoid> allow_moves;

    bool MoveRestrictions() const {
        return !avoid_moves.empty() ||
                   !allow_moves.empty();
    }

    void Clear() {
        output_format = kLeela;
        ownership =
            moves_ownership = false;
        min_moves = 0;
        max_moves = kPotentialMoves;
        avoid_moves.clear();
        allow_moves.clear();
        interval = 0;
    }

    bool IsLegal(const int vertex, const int color, const int movenum) const {
        for (const auto& move : avoid_moves) {
            if (color == move.color && vertex == move.vertex
                && movenum <= move.until_move) {
                return false;
            }
        }

        auto active_allow = false;
        for (const auto& move : allow_moves) {
            if (color == move.color && movenum <= move.until_move) {
                active_allow = true;
                if (vertex == move.vertex) {
                    return true;
                }
            }
        }
        if (active_allow) {
            return false;
        }
        return true;
    }
};

class Node {
public:
    using Edge = NodePointer<Node>;

    explicit Node(Parameters *param, std::int16_t vertex, float policy);
    ~Node();

    // Expand this node.
    bool ExpandChildren(Network &network,
                        GameState &state,
                        NodeEvals& node_evals,
                        AnalysisConfig &config,
                        const bool is_root);

    // Expand root node children before starting tree search.
    bool PrepareRootNode(Network &network,
                         GameState &state,
                         NodeEvals& node_evals,
                         AnalysisConfig &config);

    // Select the best policy node.
    Node *ProbSelectChild(bool allow_pass);

    // Select the best PUCT value node.
    Node *PuctSelectChild(const int color, const bool is_root);

    // Select the lowest visits node.
    Node *LowestVisitsSelectChild();

    // Randomly select one child by visits.
    int RandomMoveProportionally(float temp,
                                 float min_ratio,
                                 int min_visits);

    // Randomly select one child by visits and Q value.
    int RandomMoveWithLogitsQ(GameState &state, float temp);

    // Update the node.
    void Update(const NodeEvals *evals);

    // Get children's LCB with utility values.
    std::vector<std::pair<float, int>> GetSortedLcbUtilityList(const int color);
    std::vector<std::pair<float, int>> GetSortedLcbUtilityList(const int color,
                                                               const int parentvisits);

    // Get LCB value.
    float GetLcb(const int color) const;

    // Get best move(vertex) with LCB value.
    int GetBestMove(bool allow_pass);

    // Get best move(vertex) with Gumbel-Top-k trick.
    int GetGumbelMove(bool allow_pass);

    int GetRankMove();

    const std::vector<Edge> &GetChildren() const;

    bool HasChildren() const;
    bool SetTerminal();

    // Get the pointer of this node.
    Node *Get();

    // Get the child pointer according to vertex. Return NULL if
    // there is no correspond child.
    Node *GetChild(const int vertex);

    // Get the child pointer and remove it according to vertex. Return
    // NULL if there is no correspond child.
    Node *PopChild(const int vertex);

    // Get the visit number of this node.
    int GetVisits() const;

    // Get the vertex move of this node.
    int GetVertex() const;

    // Get the move probability value of this node.
    float GetPolicy() const;

    // Get the network win-loss value.
    float GetNetWL(const int color) const;

    // Get the network final score value.
    float GetNetScore(const int color) const;

    // Get the average final score value.
    float GetFinalScore(const int color) const;

    // Get score ultility evaluation.
    float GetScoreEval(const int color, float parent_score) const;

    // Get the average win-loss value.
    float GetWL(const int color, const bool use_virtual_loss=false) const;

    // Get the average draw value.
    float GetDraw() const;

    // Get First Play Urgency (FPU) for unvisited children.
    float GetFpu(const int color,
                 const float total_visited_policy,
                 const bool is_root) const;

    float GetCpuct(int parentvisits) const;

    // Get the average ownership value.
    std::array<float, kNumIntersections> GetOwnership(int color);

    // Set the network win-loss value from outside.
    void ApplyEvals(const NodeEvals *evals);

    float GetWLStddev() const;
    float GetScoreStddev() const;

    bool ShouldApplyGumbel() const;
    std::vector<float> GetProbLogitsCompletedQ(GameState &state);

    void SetScoreBouns(float val);

    void IncrementThreads();
    void DecrementThreads();

    bool Expandable() const;
    bool IsExpanding() const;
    bool IsExpanded() const;

    bool IsPruned() const;
    void SetActive(const bool active);
    void Invalidate();
    bool IsActive() const;
    bool IsValid() const;

    std::string GetPathVerboseString(GameState &state, int color, std::vector<int> &moves);
    std::string ToAnalysisString(GameState &state, const int color, AnalysisConfig &config);
    std::string OwnershipToString(GameState &state, const int color, std::string name, Node *node);
    std::string ToVerboseString(GameState &state, const int color);
    std::string GetPvString(GameState &state);

private:
    void RandomPruneRootChildren(Network &network, GameState &state);
    void Recompute(Network &network,
                   GameState &state,
                   const bool is_root);
    float GetDynamicCpuctFactor(Node *node, const int visits, const int parentvisits);
    void ApplyDirichletNoise(const float alpha);
    void ApplyNetOutput(GameState& state,
                        const Network::Result &raw_netlist,
                        NodeEvals& node_evals, const int color);
    void SetPolicy(float p);
    void SetVisits(int v);

    void LinkNodeList(std::vector<Network::PolicyVertexPair> &nodelist);

    int GetValidVisits() const;
    float GetSearchPolicy(Edge& child, bool noise);
    float GetScoreVariance(const float default_var, const int visits) const;
    float GetWLVariance(const float default_var, const int visits) const;

    void Inflate(Edge& child);
    void Release(Edge& child);

    void InflateAllChildren();
    void ReleaseAllChildren();
    int GetThreads() const;
    int GetVirtualLoss() const;

    float GetGumbelEval(int color, float parent_score) const;
    float TransformCompletedQ(const float completed_q,
                              const int max_visits) const;
    void ComputeNodeCount(size_t &nodes, size_t &edges);
    bool ProcessGumbelLogits(std::vector<float> &gumbel_logits,
                             const int color,
                             bool only_max_visits);
    Node *GumbelSelectChild(int color, bool only_max_visits, bool allow_pass);
    void MixLogitsCompletedQ(GameState &state,
                             std::vector<float> &prob);

    void KillRootSuperkos(GameState &state);

    enum class StatusType : std::uint8_t {
        kInvalid, // kInvalid means that this node is illegal, like
                  // superko move.

        kPruned,  // kPruned means that this node is pruned.
        kActive
    };
    std::atomic<StatusType> status_{StatusType::kActive};

    enum class ExpandState : std::uint8_t {
        kInitial = 0,
        kExpanding,
        kExpanded
    };
    std::atomic<ExpandState> expand_state_{ExpandState::kInitial};

    // kInitial -> kExpanding
    bool AcquireExpanding();

    // kExpanding -> done
    void ExpandDone();

    // kExpanding -> kInitial
    void ExpandCancel();

    // wait until we are on kExpanded state
    void WaitExpanded() const;

    // Color of the node. Set kInvalid if there are no children.
    int color_{kInvalid};

    Parameters *param_{nullptr};

    // According to KataGo, to add a tiny bonus for pass move can
    // efficiently end the game. It also does not affect the
    // theoretical optimal play. Only add the bonus during search
    // in order to do not affect the training result. Be care that
    // the bouns is not side to move bouns. It will award any side
    // score utility.
    float score_bouns_{0.f};

    // The network win-loss value.
    float black_wl_{0.5f};

    // The network final score value.
    float black_fs_{0.0f};

    // The accumulated squared difference values.
    std::atomic<double> squared_eval_diff_{1e-4f};
    std::atomic<double> squared_score_diff_{1e-4f};

    // The black accumulated values.
    std::atomic<double> accumulated_black_fs_{0.0f};
    std::atomic<double> accumulated_black_wl_{0.0f};
    std::atomic<double> accumulated_draw_{0.0f};

    // The black average ownership value.
    std::array<float, kNumIntersections> avg_black_ownership_;

    // The ownership lock.
    std::mutex os_mtx_;

    // The visits number of this node.
    std::atomic<int> visits_{0};

    // The threads number below this sub-tree.
    std::atomic<int> running_threads_{0};

    // The children of this node.
    std::vector<Edge> children_;

    // The played move.
    std::int16_t vertex_;

    // The move probability value of this node.
    float policy_;
};
