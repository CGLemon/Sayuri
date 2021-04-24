#pragma once

#include "game/game_state.h"
#include "game/types.h"

#include "search/node_pointer.h"
#include "search/puct/parameters.h"

#include <vector>
#include <memory>

class PuctNode;

struct PuctNodeStats {
    std::atomic<int> nodes{0};
    std::atomic<int> edges{0};
};

struct PuctNodeData {
    float policy{0.0f};
    int vertex{-1};

    std::shared_ptr<Parameters> parameters{nullptr};
    std::shared_ptr<PuctNodeStats> node_status{nullptr};

    PuctNode *parent{nullptr};
};

class PuctNode {
public:
    using Edge = NodePointer<PuctNode, PuctNodeData>;

    PuctNode(std::shared_ptr<PuctNodeData> data);
    ~PuctNode();

private:
    std::vector<Edge> children_;
    std::shared_ptr<PuctNodeData> data_;
};
