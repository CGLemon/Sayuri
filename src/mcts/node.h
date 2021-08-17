#pragma once

#include "game/game_state.h"
#include "game/types.h"

#include "mcts/node_pointer.h"
#include "mcts/parameters.h"

#include "neural/network.h"

#include <array>
#include <vector>
#include <memory>
#include <atomic>
#include <string>
#include <mutex>

class Node;

struct NodeStats {
    std::atomic<int> nodes{0};
    std::atomic<int> edges{0};
};

struct NodeData {
    float policy{0.0f};
    int vertex{kNullVertex};

    std::shared_ptr<Parameters> parameters{nullptr};
    std::shared_ptr<NodeStats> node_stats{nullptr};

    Node *parent{nullptr};
};

struct NodeEvals {
    float black_final_score{0.0f};
    float black_wl{0.0f};
    float draw{0.0f};

    std::array<float, kNumIntersections> black_ownership;
};

class Node {
public:
    using Edge = NodePointer<Node, NodeData>;

    Node(std::shared_ptr<NodeData> data);
    ~Node();

    bool ExpendChildren(Network &network,
                        GameState &state,
                        const bool is_root);

    NodeEvals PrepareRootNode(Network &network,
                              GameState &state,
                              std::vector<float> &dirichlet);

    Node *ProbSelectChild();
    Node *UctSelectChild(const int color, const bool is_root);

    void PolicyTargetPruning();
    int RandomizeFirstProportionally(float random_temp);

    void Update(std::shared_ptr<NodeEvals> evals);

    std::vector<std::pair<float, int>> GetLcbList(const int color);
    int GetBestMove();

    const std::vector<std::shared_ptr<Edge>> &GetChildren() const;
    bool HaveChildren() const;

    Node *Get();
    Node *GetChild(int vertex);
    int GetVisits() const;
    int GetVertex() const;
    float GetPolicy() const;

    float GetNetFinalScore(const int color) const;
    float GetFinalScore(const int color) const;
    float GetEval(const int color, const bool use_virtual_loss=true) const;
    float GetNetEval(const int color) const;
    float GetNetDraw() const;
    float GetDraw() const;

    size_t GetMemoryUsed() const;

    std::array<float, kNumIntersections> GetOwnership(int color) const;
    NodeEvals GetNodeEvals() const;
    void ApplyEvals(std::shared_ptr<NodeEvals> evals);

    void IncrementThreads();
    void DecrementThreads();

    bool Expandable() const;
    bool IsExpending() const;
    bool IsExpended() const;

    bool IsPruned() const;

    std::string ToString(GameState &state);
    std::string GetPvString(GameState &state);

private:
    void ApplyDirichletNoise(const float alpha);
    void SetPolicy(float p);
    void SetVisits(int v);

    void LinkNodeList(std::vector<Network::PolicyVertexPair> &nodelist);
    void LinkNetOutput(const Network::Result &raw_netlist, const int color);

    float GetUctPolicy(std::shared_ptr<Edge>, bool noise);
    float GetScoreUtility(const int color, float factor, float parent_score) const;
    float GetVariance(const float default_var, const int visits) const;
    float GetLcb(const int color) const;

    void Inflate(std::shared_ptr<Edge> child);
    void Release(std::shared_ptr<Edge> child);

    void InflateAllChildren();
    void ReleaseAllChildren();

    void IncrementNodes();
    void DecrementNodes();

    void IncrementEdges();
    void DecrementEdges();

    int GetThreads() const;
    int GetVirtualLoss() const;

    std::shared_ptr<NodeStats> GetStats() const;
    std::shared_ptr<Parameters> GetParameters() const;

    enum class StatusType : std::uint8_t {
        kInvalid, // kInvalid means that the node is illegal.
        kPruned,
        kActive
    };
    std::atomic<StatusType> status_{StatusType::kActive};

    void SetActive(const bool active);
    void InvaliNode();
    bool IsActive() const;
    bool IsValid() const;

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
    float black_fs_;
    float black_wl_;
    float draw_;
    std::array<float, kNumIntersections> black_ownership_;

    std::mutex update_mtx_;

    std::atomic<float> squared_eval_diff_{1e-4f};
    std::atomic<float> accumulated_black_fs_{0.0f};
    std::atomic<float> accumulated_black_wl_{0.0f};
    std::atomic<float> accumulated_draw_{0.0f};
    std::array<float, kNumIntersections> accumulated_black_ownership_;

    std::atomic<int> visits_{0};
    std::atomic<int> running_threads_{0};

    std::vector<std::shared_ptr<Edge>> children_;
    std::shared_ptr<NodeData> data_;
};
