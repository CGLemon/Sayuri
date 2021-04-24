#include "search/puct/node.h"
#include <cassert>

PuctNode::PuctNode(std::shared_ptr<PuctNodeData> data) {
    assert(data->parameters != nullptr);
    data_ = data;
    // increment_nodes();
}

PuctNode::~PuctNode() {
    // assert(get_threads() == 0);
    // decrement_nodes();
    // release_all_children();
    // for (auto i = size_t{0}; i < m_children.size(); ++i) {
    //     decrement_edges();
    // }
}
