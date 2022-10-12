#pragma once

#include "game/game_state.h"
#include "game/types.h"
#include "mcts/node_pointer.h"
#include "mcts/parameters.h"
#include "neural/network.h"
#include "utils/operators.h"

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

class Node {
public:
    using Edge = NodePointer<Node>;

    explicit Node(std::int16_t vertex, float policy);
    ~Node();

    // Expand this node.
    bool ExpandChildren(Network &network,
                        GameState &state,
                        const bool is_root);

    // Expand root node children before starting tree search.
    void PrepareRootNode(Network &network,
                         GameState &state,
                         std::vector<float> &dirichlet);

    // Select the best policy node.
    Node *ProbSelectChild();

    // Select the best PUCT value node.
    Node *PuctSelectChild(const int color, const bool is_root);

    // Select the best UCT value node. For no-dcnn mode.
    Node *UctSelectChild(const int color, const bool is_root, const GameState &state);

    void PolicyTargetPruning();

    // Randomly select one child by visits. 
    int RandomizeFirstProportionally(float random_temp);

    // Update the node.
    void Update(const NodeEvals *evals);

    // Get children's LCB values. 
    std::vector<std::pair<float, int>> GetLcbList(const int color);

    // Get best move(vertex) by LCB value.
    int GetBestMove();

    // Do rollout and mix the rollout reuslt with the node evals.
    void MixRolloutEvals(GameState &state,
                             float eval_factor, float owner_factor);

    const std::vector<Edge> &GetChildren() const;
    bool HaveChildren() const;
    bool SetTerminal();

    void SetParameters(Parameters * param);

    Node *Get();
    Node *GetChild(const int vertex);
    Node *PopChild(const int vertex);

    int GetVisits() const;
    int GetVertex() const;
    float GetPolicy() const;

    // GetNetxxx will get raw NN eval from this node.
    float GetNetFinalScore(const int color) const;
    float GetNetEval(const int color) const;
    float GetNetDraw() const;

    float GetFinalScore(const int color) const;
    float GetEval(const int color, const bool use_virtual_loss=true) const;
    float GetDraw() const;

    std::array<float, kNumIntersections> GetOwnership(int color);
    NodeEvals GetNodeEvals() const;
    void ApplyEvals(const NodeEvals *evals);

    void IncrementThreads();
    void DecrementThreads();

    bool Expandable() const;
    bool IsExpanding() const;
    bool IsExpanded() const;

    bool IsPruned() const;
    void SetActive(const bool active);
    void InvaliNode();
    bool IsActive() const;
    bool IsValid() const;

    float ComputeKlDivergence();
    float ComputeTreeComplexity();

    enum class AnalysisTag : int {
        kNullTag        = 0,
        kSayuri         = 1,
        kKata           = 1 << 1,
        kOwnership      = 1 << 2,
        kMovesOwnership = 1 << 3
    };
    ENABLE_FRIEND_BITWISE_OPERATORS_ON(Node::AnalysisTag);

    std::string ToAnalyzeString(GameState &state, const int color, Node::AnalysisTag tag);
    std::string OwnershipToString(GameState &state, const int color, std::string name, Node *node);
    std::string ToVerboseString(GameState &state, const int color);
    std::string GetPvString(GameState &state);

private:
    void ApplyNoDcnnPolicy(GameState &state,
                               const int color,
                               Network::Result &raw_netlist) const;
    void ApplyDirichletNoise(const float alpha);
    void SetPolicy(float p);
    void SetVisits(int v);

    void LinkNodeList(std::vector<Network::PolicyVertexPair> &nodelist);
    void LinkNetOutput(const Network::Result &raw_netlist, const int color);

    float GetSearchPolicy(Edge& child, bool noise);
    float GetScoreUtility(const int color, float factor, float parent_score) const;
    float GetVariance(const float default_var, const int visits) const;
    float GetLcb(const int color) const;

    void Inflate(Edge& child);
    void Release(Edge& child);

    void InflateAllChildren();
    void ReleaseAllChildren();

    int GetThreads() const;
    int GetVirtualLoss() const;

    void ComputeNodeCount(size_t &nodes, size_t &edges);

    Parameters *GetParameters();

    enum class StatusType : std::uint8_t {
        kInvalid, // kInvalid means that the node is illegal.
        kPruned,
        kActive
    };
    std::atomic<StatusType> status_{StatusType::kActive};

    enum class ExpandState : std::uint8_t {
        kInitial = 0,
        kExpanding,
        kExpanded
    };
    std::atomic<ExpandState> expand_state_{ExpandState::kInitial};

    // INITIAL -> EXPANDING
    bool AcquireExpanding();

    // EXPANDING -> DONE
    void ExpandDone();

    // EXPANDING -> INITIAL
    void ExpandCancel();

    // wait until we are on EXPANDED state
    void WaitExpanded() const;

    int color_{kInvalid};

    Parameters *param_;

    // network outputs
    float black_fs_;
    float black_wl_;
    float draw_;
    std::array<float, kNumIntersections> black_ownership_;

    std::mutex os_mtx_; // the ownership accumulated lock

    // accumulated values
    std::atomic<double> squared_eval_diff_{1e-4f};
    std::atomic<double> accumulated_black_fs_{0.0f};
    std::atomic<double> accumulated_black_wl_{0.0f};
    std::atomic<double> accumulated_draw_{0.0f};
    std::array<double, kNumIntersections> accumulated_black_ownership_;

    std::atomic<int> visits_{0};
    std::atomic<int> running_threads_{0};

    std::vector<Edge> children_;

    std::int16_t vertex_;
    float policy_;
};
