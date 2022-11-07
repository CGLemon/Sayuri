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
    struct MoveToAvoid{
        int vertex{kNullVertex}, color{kInvalid}, until_move{-1};
        bool Valid() const {
            return vertex != kNullVertex &&
                       color != kInvalid &&
                       until_move >= 1;
        }
    };

    bool is_sayuri{false};
    bool is_kata{false};
    bool is_leelaz{false};
    bool ownership{false};
    bool moves_ownership{false};

    int interval{100};
    int min_moves{0};
    int max_moves{kNumIntersections+1};

    std::vector<MoveToAvoid> avoid_moves;
    std::vector<MoveToAvoid> allow_moves;

    bool MoveRestrictions() const {
        return !avoid_moves.empty() ||
                   !avoid_moves.empty();
    }

    void Clear() {
        is_sayuri =
            is_kata =
            is_leelaz =
            ownership =
            moves_ownership = false;
        min_moves = 0;
        max_moves = kNumIntersections+1;
        avoid_moves.clear();
        allow_moves.clear();
        interval = 100;
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

    explicit Node(std::int16_t vertex, float policy);
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
                             AnalysisConfig &config,
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
    std::vector<std::pair<float, int>> GetLcbUtilityList(const int color);

    // Get best move(vertex) by LCB value.
    int GetBestMove();

    const std::vector<Edge> &GetChildren() const;

    bool HaveChildren() const;
    bool SetTerminal();

    void SetParameters(Parameters * param);

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

    // Get the Network win-loss value. 
    float GetNetWL(const int color) const;

    // Get the average final score value.
    float GetFinalScore(const int color) const;

    // Get the average win-loss value.
    float GetWL(const int color, const bool use_virtual_loss=true) const;

    // Get the average draw value.
    float GetDraw() const;

    // Get the average ownership value.
    std::array<float, kNumIntersections> GetOwnership(int color);

    // Set the network win-loss value from outside.
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

    std::string ToAnalysisString(GameState &state, const int color, AnalysisConfig &config);
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
    float GetScoreUtility(const int color, float div, float parent_score) const;
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

    // The accumulated squared difference value.
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

    // The move probability value of this node.
    float policy_;
};
