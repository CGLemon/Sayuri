#pragma once

#include "game/game_state.h"
#include "game/types.h"

#include "mcts/node_pointer.h"
#include "mcts/parameters.h"

#include <vector>
#include <memory>

class Node;

struct NodeStats {
    std::atomic<int> nodes{0};
    std::atomic<int> edges{0};
};

struct NodeData {
    float policy{0.0f};
    int vertex{-1};

    std::shared_ptr<Parameters> parameters{nullptr};
    std::shared_ptr<NodeStats> node_status{nullptr};

    Node *parent{nullptr};
};

class Node {
public:
    using Edge = NodePointer<Node, NodeData>;

    Node(std::shared_ptr<NodeData> data);
    ~Node();

private:
    std::vector<Edge> children_;
    std::shared_ptr<NodeData> data_;
};
