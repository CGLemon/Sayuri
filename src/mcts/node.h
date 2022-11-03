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
                        NodeEvals& node_evals,
                        const bool is_root);

    // Expand root node children before starting tree search.
    bool PrepareRootNode(Network &network,
                         GameState &state,
                         NodeEvals& node_evals,
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
    // float GetNetFinalScore(const int color) const;
    float GetNetEval(const int color) const;
    // float GetNetDraw() const;

    float GetFinalScore(const int color) const;
    float GetEval(const int color, const bool use_virtual_loss=true) const;
    float GetDraw() const;

    std::array<float, kNumIntersections> GetOwnership(int color);
    void ApplyEvals(const NodeEvals *evals);

    void IncrementThreads();
    void DecrementThreads();

    bool Expandable() const;
    bool IsExpanding() const;
    bool IsExpanded() const;

    bool IsPruned() const;
    void SetActive(const bool active);
    void InvalidNode();
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
    void ApplyNetOutput(GameState &state,
                        const Network::Result &raw_netlist,
                        NodeEvals& node_evals, const int color);
    void SetPolicy(float p);
    void SetVisits(int v);

    void LinkNodeList(std::vector<Network::PolicyVertexPair> &nodelist);

    float GetSearchPolicy(Edge& child, bool noise);
    float GetScoreUtility(const int color, float factor, float parent_score) const;
    float GetLcbVariance(const float default_var, const int visits) const;
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

    // Color of the node. Set kInvalid if there are no children.
    int color_{kInvalid};

    Parameters *param_;

    // The network win-loss value.
    float black_wl_;

    // The Accumulated values.
    std::atomic<double> squared_eval_diff_{1e-4f};

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

    // Policy value of the node.
    float policy_;
};
