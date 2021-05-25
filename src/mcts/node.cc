#include "mcts/node.h"
#include "mcts/lcb.h"

#include <cassert>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

#define VIRTUAL_LOSS_COUNT (3)

Node::Node(std::shared_ptr<NodeData> data) {
    assert(data->parameters != nullptr);
    data_ = data;
    IncrementNodes();
}

Node::~Node() {
    assert(GetThreads() == 0);
    DecrementNodes();
    ReleaseAllChildren();
    for (auto i = size_t{0}; i < children_.size(); ++i) {
        DecrementEdges();
    }
}

NodeEvals Node::PrepareRootNode(Network &network,
                                GameState &state) {
    const auto is_root = true;
    const auto success = ExpendChildren(network, state, is_root);
    assert(success);
    assert(HaveChildren());
    if (success) {
        InflateAllChildren();
    }
    return GetNodeEvals();
}

bool Node::ExpendChildren(Network &network,
                          GameState &state,
                          const bool is_root) {
    assert(state.GetPasses() < 2);
    assert(!HaveChildren());

    if (!AcquireExpanding()) {
        return false;
    }

    const auto raw_netlist = network.GetOutput(state, Network::kRandom);
    const auto board_size = state.GetBoardSize();
    const auto num_intersections = state.GetNumIntersections();

    color_ = state.GetToMove();
    linkNetOutput(raw_netlist, color_);

    auto nodelist = std::vector<Network::PolicyVertexPair>{};
    float legal_accumulate = 0.0f;

    for (int idx = 0; idx < num_intersections; ++idx) {
        const auto x = idx % board_size;
        const auto y = idx / board_size;
        const auto vtx = state.GetVertex(x, y);
        const auto policy = raw_netlist.probabilities[idx];

        if (is_root) {}

        nodelist.emplace_back(policy, vtx);
        legal_accumulate += policy;
    }

    nodelist.emplace_back(raw_netlist.pass_probability, kPass);

    for (auto &node : nodelist) {
        node.first /= legal_accumulate;
    }

    LinkNodeList(nodelist);
    ExpandDone();

    return true;
}

void Node::LinkNodeList(std::vector<Network::PolicyVertexPair> &nodelist) {
    std::stable_sort(std::rbegin(nodelist), std::rend(nodelist));

    for (const auto &node : nodelist) {
        auto data = std::make_shared<NodeData>();
        data->vertex = node.second;
        data->policy = node.first;
        data->parameters = GetParameters();
        data->node_stats = GetStats();

        data->parent = Get();

        children_.emplace_back(std::make_shared<Edge>(data));
        IncrementEdges();
    }
    assert(!children_.empty());
}


void Node::linkNetOutput(const Network::Result &raw_netlist, const int color){
    auto wl = raw_netlist.wdl[0] - raw_netlist.wdl[2];
    auto draw = raw_netlist.wdl[1];

    wl = (wl + 1) * 0.5f;
    if (color == kWhite) {
        wl = 1.0f - wl;
    }

    black_wl_ = wl;
    draw_ = draw;

    for (int idx = 0; idx < kNumIntersections; ++idx) {
        auto owner = raw_netlist.ownership[idx];
        if (color == kWhite) {
            owner = 0.f - owner;
        }
        black_ownership_[idx] = owner;
        accumulated_black_ownership_[idx] = 0;
    }
}


Node *Node::ProbSelectChild() {
    WaitExpanded();
    assert(HaveChildren());

    std::shared_ptr<Edge> best_node = nullptr;
    float best_prob = std::numeric_limits<float>::lowest();

    for (const auto &child : children_) {
        const auto node = child->Get();
        const bool is_pointer = node == nullptr ? false : true;

        auto prob = node->GetPolicy();

        // The node was pruned. Skip this time.
        if (is_pointer && !node->IsActive()) {
            continue;
        }

        if (is_pointer && node->IsExpending()) {
            prob = -1.0f + prob;
        }

        if (prob > best_prob) {
            best_prob = prob;
            best_node = child;
        }
    }

    Inflate(best_node);
    return best_node->Get();
}

Node *Node::UctSelectChild(const int color, const bool is_root) {
    WaitExpanded();
    assert(HaveChildren());
    assert(color == color_);

    int parentvisits = 0;
    float total_visited_policy = 0.0f;
    for (const auto &child : children_) {
        const auto node = child->Get();
        if (!node) {
            continue;
        }    
        if (node->IsValid()) {
            const auto visits = node->GetVisits();
            parentvisits += visits;
            if (visits > 0) {
                total_visited_policy += node->GetPolicy();
            }
        }
    }

    const auto fpu_reduction_factor = is_root ? GetParameters()->fpu_root_reduction : GetParameters()->fpu_reduction;
    const auto cpuct_init           = is_root ? GetParameters()->cpuct_root_init    : GetParameters()->cpuct_init;
    const auto cpuct_base           = is_root ? GetParameters()->cpuct_root_base    : GetParameters()->cpuct_base;
    const auto draw_factor          = is_root ? GetParameters()->draw_root_factor   : GetParameters()->draw_factor;

    const float cpuct         = cpuct_init + std::log((float(parentvisits) + cpuct_base + 1) / cpuct_base);
    const float numerator     = std::sqrt(float(parentvisits));
    const float fpu_reduction = fpu_reduction_factor * std::sqrt(total_visited_policy);
    const float fpu_value     = GetNetEval(color) - fpu_reduction;

    std::shared_ptr<Edge> best_node = nullptr;
    float best_value = std::numeric_limits<float>::lowest();

    for (const auto &child : children_) {
        // Check the node is pointer or not.
        // If not, we can not get most data from child.
        const auto node = child->Get();
        const bool is_pointer = node == nullptr ? false : true;

        // The node was pruned. Skip this time.
        if (is_pointer && !node->IsActive()) {
            continue;
        }

        float q_value = fpu_value;
        if (is_pointer) {
            if (node->IsExpending()) {
                q_value = -1.0f - fpu_reduction;
            } else if (node->GetVisits() > 0) {
                const float eval = node->GetEval(color);
                const float draw_value = node->GetDraw() * draw_factor;
                q_value = eval + draw_value;
            }
        }

        float denom = 1.0f;
        if (is_pointer) {
            denom += node->GetVisits();
        }

        const float psa = child->Data()->policy;
        const float puct = cpuct * psa * (numerator / denom);
        const float value = q_value + puct;
        assert(value > std::numeric_limits<float>::lowest());

        if (value > best_value) {
            best_value = value;
            best_node = child;
        }
    }

    Inflate(best_node);
    return best_node->Get();
}

template <typename T> 
void atomic_add(std::atomic<T> &f, T d) {
    T old = f.load();
    while (!f.compare_exchange_weak(old, old + d)) {
        std::this_thread::yield();
    }
}

void Node::Update(std::shared_ptr<NodeEvals> evals) {
    const float eval = evals->black_wl;
    const float old_eval = accumulated_black_wl_.load();
    const float old_visits = visits_.load();

    const float old_delta = old_visits > 0 ? eval - old_eval / old_visits : 0.0f;
    const float new_delta = eval - (old_eval + eval) / (old_visits + 1);

    // Welford's online algorithm for calculating variance.
    const float delta = old_delta * new_delta;

    visits_.fetch_add(1);
    atomic_add(squared_eval_diff_, delta);
    atomic_add(accumulated_black_wl_, eval);
    atomic_add(accumulated_draw_, evals->draw);
    {
        std::lock_guard<std::mutex> lock(update_mtx_);
        for (int idx = 0; idx < kNumIntersections; ++idx) {
            accumulated_black_ownership_[idx] += evals->black_ownership[idx];
        }
    }
}

void Node::ApplyEvals(std::shared_ptr<NodeEvals> evals) {
    black_wl_ = evals->black_wl;
    draw_ = evals->draw;
    std::copy(std::begin(evals->black_ownership),
                  std::end(evals->black_ownership),
                  std::begin(black_ownership_));
}

std::array<float, kNumIntersections> Node::GetOwnerShip(int color) const {
    const auto visits = GetVisits();
    auto out = std::array<float, kNumIntersections>{};
    for (int idx = 0; idx < kNumIntersections; ++idx) {
        auto owner = accumulated_black_ownership_[idx] / visits;
        if (color == kWhite) {
            owner = 0.f - owner;
        }
        out[idx] = owner;
    }
    return out;
}

float Node::GetVariance(const float default_var, const int visits) const {
    return visits > 1 ? squared_eval_diff_.load() / (visits - 1) : default_var;
}

float Node::GetLcb(const int color) const {
    // LCB issues: https://github.com/leela-zero/leela-zero/pull/2290
    // Lower confidence bound of winrate.
    const auto visits = GetVisits();
    if (visits < 2) {
        // Return large negative value if not enough visits.
        return GetPolicy() - 1e6f;
    }

    const auto mean = GetEval(color, false);

    const auto variance = GetVariance(1.0f, visits);
    const auto stddev = std::sqrt(variance / float(visits));
    const auto z = cached_t_quantile(visits - 1);
    
    return mean - z * stddev;
}

size_t Node::GetMemoryUsed() const {
    const auto nodes = GetStats()->nodes.load();
    const auto edges = GetStats()->edges.load();
    const auto node_mem = sizeof(Node) + sizeof(Edge);
    const auto edge_mem = sizeof(Edge);
    return nodes * node_mem + edges * edge_mem;
}

std::string Node::ToString(GameState &state) {
    auto out = std::ostringstream{};
    const auto color = color_;
    const auto lcblist = GetLcbList(color);
    const auto parentvisits = static_cast<float>(GetVisits());

    const auto space = 7;
    out << "Search List:" << std::endl;
    out << std::setw(6) << "move"
            << std::setw(10) << "visits"
            << std::setw(space) << "WL(%)"
            << std::setw(space) << "LCB(%)"
            << std::setw(space) << "D(%)"
            << std::setw(space) << "P(%)"
            << std::setw(space) << "N(%)"
            << std::endl;

    for (auto &lcb : lcblist) {
        const auto lcb_value = lcb.first > 0.0f ? lcb.first : 0.0f;
        const auto vertex = lcb.second;

        auto child = GetChild(vertex);
        const auto visits = child->GetVisits();
        const auto pobability = child->GetPolicy();
        assert(visits != 0);

        const auto eval = child->GetEval(color, false);
        const auto draw = child->GetDraw();

        const auto pv_string = state.VertexToText(vertex) + ' ' + child->GetPvString(state);

        const auto visit_ratio = static_cast<float>(visits) / (parentvisits - 1); // One is root visit.
        out << std::fixed << std::setprecision(2)
                << std::setw(6) << state.VertexToText(vertex)
                << std::setw(10) << visits
                << std::setw(space) << eval * 100.f     // win loss eval
                << std::setw(space) << lcb_value * 100.f   // LCB eval
                << std::setw(space) << draw * 100.f        // draw probability
                << std::setw(space) << pobability * 100.f  // move probability
                << std::setw(space) << visit_ratio * 100.f
                << std::setw(6) << "| PV:" << ' ' << pv_string
                << std::endl;
    }


    const auto mem_used = static_cast<double>(GetMemoryUsed()) / (1024.f * 1024.f);
    const auto nodes = GetStats()->nodes.load();
    const auto edges = GetStats()->edges.load();
    out << "Tree Status:" << std::endl
            << std::setw(9) << "nodes:" << ' ' << nodes  << std::endl
            << std::setw(9) << "edges:" << ' ' << edges  << std::endl
            << std::setw(9) << "memory:" << ' ' << mem_used << ' ' << "(MiB)" << std::endl;


    return out.str();
}

std::string Node::GetPvString(GameState &state) {
    auto pvlist = std::vector<int>{};
    auto *next = this;
    while (next->HaveChildren()) {
        const auto vtx = next->GetBestMove();
        pvlist.emplace_back(vtx);
        next = next->GetChild(vtx);
    }
  
    auto res = std::string{};
    for (const auto &vtx : pvlist) {
        res += state.VertexToText(vtx);
        res += " ";
    }
    return res;
}


Node *Node::Get() {
    return this;
}

Node *Node::GetChild(int vertex) {
    for (const auto & child : children_) {
        const auto node = child->Get();
        if (vertex == node->GetVertex()) {
            return node;
        }
    }
    return nullptr;
}

std::vector<std::pair<float, int>> Node::GetLcbList(const int color) {
    WaitExpanded();
    assert(HaveChildren());
  
    auto list = std::vector<std::pair<float, int>>{};

    for (const auto & child : children_) {
        const auto node = child->Get();
        if (node == nullptr) {
            continue;
        }

        const auto visits = node->GetVisits();
        const auto vertex = node->GetVertex();
        const auto lcb = node->GetLcb(color);
        if (visits > 0) {
            list.emplace_back(lcb, vertex);
        }
    }

    std::stable_sort(std::rbegin(list), std::rend(list));
    return list;
}

int Node::GetBestMove() {
    WaitExpanded();
    assert(HaveChildren());

    InflateAllChildren();

    auto lcblist = GetLcbList(color_);
    float best_value = std::numeric_limits<float>::lowest();
    int best_move = kNullVertex;

    for (auto &entry : lcblist) {
        const auto lcb = entry.first;
        const auto vtx = entry.second;
        if (lcb > best_value) {
            best_value = lcb;
            best_move = vtx;
        }
    }

    if (lcblist.empty() && HaveChildren()) {
        best_move = ProbSelectChild()->GetVertex();
    }

    assert(best_move != kNullVertex);
    return best_move;
}

NodeEvals Node::GetNodeEvals() const {
    auto evals = NodeEvals{};

    evals.black_wl = black_wl_;
    evals.draw = draw_;

    for (int idx = 0; idx < kNumIntersections; ++idx) {
        evals.black_ownership[idx] = black_ownership_[idx];
    }

    return evals;
}


std::shared_ptr<NodeStats> Node::GetStats() const {
    return data_->node_stats;
}

std::shared_ptr<Parameters> Node::GetParameters() const {
    return data_->parameters;
}

int Node::GetThreads() const {
    return running_threads_.load();
}

int Node::GetVirtualLoss() const {
    return GetThreads() * VIRTUAL_LOSS_COUNT;
}

int Node::GetVertex() const {
    return data_->vertex;
}

float Node::GetPolicy() const {
    return data_->policy;
}

int Node::GetVisits() const {
    return visits_.load();
}

float Node::GetDraw() const {
    return draw_;
}

float Node::GetNetEval(const int color) const {
    if (color == kBlack) {
        return black_wl_;
    }
    return 1.0f - black_wl_;
}

float Node::GetEval(const int color, const bool use_virtual_loss) const {
    auto virtual_loss = 0;

    if (use_virtual_loss) {
        // If this node is seaching, punish this node.
        virtual_loss = GetVirtualLoss();
    }

    auto visits = GetVisits() + virtual_loss;
    assert(visits >= 0);

    auto accumulated_wl = accumulated_black_wl_.load();
    if (color == kWhite && use_virtual_loss) {
        accumulated_wl += static_cast<float>(virtual_loss);
    }
    auto eval = accumulated_wl / static_cast<float>(visits);

    if (color == kBlack) {
        return eval;
    }
    return 1.0f - eval;
}

void Node::InflateAllChildren() {
    for (const auto &child : children_) {
         Inflate(child);
    }
}

void Node::ReleaseAllChildren() {
    for (const auto &child : children_) {
         Release(child);
    }
}

void Node::Inflate(std::shared_ptr<Edge> child) {
    if (child->Inflate()) {
        DecrementEdges();
        IncrementNodes();
    }
}

void Node::Release(std::shared_ptr<Edge> child) {
    if (child->Release()) {
        DecrementNodes();
        IncrementEdges();
    }
}

bool Node::HaveChildren() const { 
    return color_ != kInvalid;
}

void Node::IncrementThreads() {
    running_threads_.fetch_add(1);
}

void Node::DecrementThreads() {
    running_threads_.fetch_sub(1);
}

void Node::IncrementNodes() {
    GetStats()->nodes.fetch_add(1);
}

void Node::DecrementNodes() {
    GetStats()->nodes.fetch_sub(1); 
}

void Node::IncrementEdges() {
    GetStats()->edges.fetch_add(1); 
}

void Node::DecrementEdges() {
    GetStats()->edges.fetch_sub(1); 
}

void Node::SetActive(const bool active) {
    if (IsValid()) {
        status_ = active ? StatusType::kActive : StatusType::kPruned;
    }
}

void Node::InvaliNode() {
    if (IsValid()) {
        status_ = StatusType::kInvalid;
    }
}

bool Node::IsPruned() const {
    return status_.load() == StatusType::kPruned;
}

bool Node::IsActive() const {
    return status_.load() == StatusType::kActive;
}

bool Node::IsValid() const {
    return status_.load() != StatusType::kInvalid;
}

bool Node::AcquireExpanding() {
    auto expected = ExpandState::kInitial;
    auto newval = ExpandState::kExpanding;
    return expand_state_.compare_exchange_strong(expected, newval);
}

void Node::ExpandDone() {
    auto v = expand_state_.exchange(ExpandState::kExpanded);
#ifdef NDEBUG
    (void) v;
#endif
    assert(v == ExpandState::kExpanding);
}

void Node::ExpandCancel() {
    auto v = expand_state_.exchange(ExpandState::kInitial);
#ifdef NDEBUG
    (void) v;
#endif
    assert(v == ExpandState::kExpanding);
}

void Node::WaitExpanded() const {
    while (true) {
        auto v = expand_state_.load();
        if (v == ExpandState::kExpanded) {
            break;
        }
    }
}

bool Node::Expandable() const {
    return expand_state_.load() == ExpandState::kInitial;
}

bool Node::IsExpending() const {
    return expand_state_.load() == ExpandState::kExpanding;
}

bool Node::IsExpended() const {
    return expand_state_.load() == ExpandState::kExpanded;
}

