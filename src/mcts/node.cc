#include "mcts/node.h"
#include <cassert>

Node::Node(std::shared_ptr<NodeData> data) {
    assert(data->parameters != nullptr);
    data_ = data;
    // increment_nodes();
}

Node::~Node() {
    // assert(get_threads() == 0);
    // decrement_nodes();
    // release_all_children();
    // for (auto i = size_t{0}; i < m_children.size(); ++i) {
    //     decrement_edges();
    // }
}
