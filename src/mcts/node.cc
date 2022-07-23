#include "mcts/node.h"
#include "mcts/lcb.h"
#include "utils/atomic.h"
#include "utils/random.h"
#include "utils/format.h"
#include "mcts/rollout.h"

#include <cassert>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <stack>

#define VIRTUAL_LOSS_COUNT (3)

Node::Node(NodeData data) {
    assert(data->parameters != nullptr);
    data_ = data;
}

Node::~Node() {
    assert(GetThreads() == 0);
    ReleaseAllChildren();
}

void Node::PrepareRootNode(Network &network,
                           GameState &state,
                           std::vector<float> &dirichlet) {
    const auto is_root = true;
    ExpandChildren(network, state, is_root);
    assert(HaveChildren());

    InflateAllChildren();
    if (GetParameters()->dirichlet_noise) {
        // Generate dirichlet noise and gather it.
        const auto legal_move = children_.size();
        const auto factor = GetParameters()->dirichlet_factor;
        const auto init = GetParameters()->dirichlet_init;
        const auto alpha = init * factor / static_cast<float>(legal_move);

        ApplyDirichletNoise(alpha);

        const auto num_intersections = state.GetNumIntersections();
        const auto board_size = state.GetBoardSize();
        dirichlet.resize(num_intersections+1);

        for (auto idx = 0; idx < num_intersections; ++idx) {
            const auto x = idx % board_size;
            const auto y = idx / board_size;
            const auto vtx = state.GetVertex(x, y);
            dirichlet[idx] = GetParameters()->dirichlet_buffer[vtx];
        }
        dirichlet[num_intersections] = GetParameters()->dirichlet_buffer[kPass];
    }
}

bool Node::ExpandChildren(Network &network,
                          GameState &state,
                          const bool is_root) {
    // The node must be the first time to expand and is not the terminate node.
    assert(state.GetPasses() < 2);
    if (HaveChildren()) {
        return false;
    }

    // First, try to acquire the owner.
    if (!AcquireExpanding()) {
        return false;
    }

    // Second, get network computation result.
    const float temp = is_root ? GetParameters()->root_policy_temp : GetParameters()->policy_temp;

    auto raw_netlist = Network::Result{};
    color_ = state.GetToMove();

    if (GetParameters()->no_dcnn) {
        ApplyNoDcnnPolicy(state, color_, raw_netlist);
    } else {
        raw_netlist = network.GetOutput(state, Network::kRandom, temp);
    }

    LinkNetOutput(raw_netlist, color_);

    auto nodelist = std::vector<Network::PolicyVertexPair>{};
    auto allow_pass = true;
    auto legal_accumulate = 0.0f;

    const auto board_size = state.GetBoardSize();
    const auto num_intersections = state.GetNumIntersections();
    const auto safe_area = state.GetStrictSafeArea();

    // Third, remove the illegal moves or some bad move.
    for (int idx = 0; idx < num_intersections; ++idx) {
        const auto x = idx % board_size;
        const auto y = idx / board_size;
        const auto vtx = state.GetVertex(x, y);
        const auto policy = raw_netlist.probabilities[idx];

        if (!state.IsLegalMove(vtx, color_) || safe_area[idx]) {
            continue;
        }

        if (is_root) {
            auto fork_state = state;
            fork_state.PlayMove(vtx);

            if (fork_state.IsSuperko()) {
                continue;
            }
        }

        nodelist.emplace_back(policy, vtx);
        legal_accumulate += policy;
    }

    if ((int)nodelist.size() > 3*num_intersections/4) {
        allow_pass = false;
    }

    if (allow_pass || nodelist.empty()) {
        nodelist.emplace_back(raw_netlist.pass_probability, kPass);
        legal_accumulate += raw_netlist.pass_probability;
    }

    if (legal_accumulate < 1e-6f) {
        // It will be happened if the policy focuses on illegal moves.
        for (auto &node : nodelist) {
            node.first = 1.f/nodelist.size();
        }
    } else {
        for (auto &node : nodelist) {
            node.first /= legal_accumulate;
        }
    }

    // Fourth, append the nodes.
    LinkNodeList(nodelist);

    // Fifth, release the owner.
    ExpandDone();

    return true;
}

void Node::LinkNodeList(std::vector<Network::PolicyVertexPair> &nodelist) {
    // Besure that the best policy is on the top.
    std::stable_sort(std::rbegin(nodelist), std::rend(nodelist));

    for (const auto &node : nodelist) {
        auto data = NodeData{};

        data.depth = data_.depth + 1;
        data.vertex = node.second;
        data.policy = node.first;
        data.parameters = GetParameters();
        data.parent = Get();

        children_.emplace_back(data);
    }
    assert(!children_.empty());
}

void Node::LinkNetOutput(const Network::Result &raw_netlist, const int color){
    auto wl = (raw_netlist.wdl[0] - raw_netlist.wdl[2] + 1) * 0.5f;
    auto draw = raw_netlist.wdl[1];
    auto final_score = raw_netlist.final_score;

    if (color == kWhite) {
        wl = 1.0f - wl;
        final_score = 0.0f - final_score;
    }

    black_wl_ = wl;
    draw_ = draw;
    black_fs_ = final_score;

    for (int idx = 0; idx < kNumIntersections; ++idx) {
        auto owner = raw_netlist.ownership[idx];
        if (color == kWhite) {
            owner = 0.f - owner;
        }
        black_ownership_[idx] = owner;
        accumulated_black_ownership_[idx] = 0;
    }
}

void Node::MixRolloutEvals(GameState &state, float factor) {
    const auto num_intersections = state.GetNumIntersections();
    auto mcowner = std::vector<float>(num_intersections, 0.f);

    float black_rollout_score;
    float black_rollout_val = GetRolloutWinrate(state, data_.depth, 1, kBlack, mcowner, black_rollout_score);

    black_wl_ = factor * black_rollout_val  + (1-factor) * black_wl_;
    black_fs_ = factor * black_rollout_score + (1-factor) * black_fs_;

    for (int idx = 0; idx < num_intersections; ++idx) {
        black_ownership_[idx] = factor * mcowner[idx] + (1-factor) * black_ownership_[idx];
    }
}

void Node::ApplyNoDcnnPolicy(GameState &state, const int color, Network::Result &raw_netlist) const {
    const auto num_intersections = state.GetNumIntersections();
    auto policy = state.GetGammasPolicy(color);

    for (int idx = 0; idx < num_intersections; ++idx) {
        raw_netlist.probabilities[idx] = policy[idx];
        raw_netlist.ownership[idx] = 0.f;
    }

    // Give a little value for pass policy avoid bug
    // if there is no legal moves.
    raw_netlist.pass_probability = 0.1f/num_intersections;
    raw_netlist.final_score = 0.f;
    raw_netlist.wdl = {0.5f, 0, 0.5f};
}

// Experimental function.
void Node::PolicyTargetPruning() {
    WaitExpanded();
    assert(HaveChildren());
    InflateAllChildren();

    auto buffer = std::vector<std::pair<int, int>>{};

    int parentvisits = 0;
    int most_visits_move = -1;
    int most_visits = 0;

    for (const auto &child : children_) {
        const auto node = child.Get();

        if (!node->IsActive()) {
            continue;
        }

        const auto visits = node->GetVisits();
        const auto vertex = node->GetVertex();
        parentvisits += visits;
        buffer.emplace_back(visits, vertex);

        if (most_visits < visits) {
            most_visits = visits;
            most_visits_move = vertex;
        }
    }

    assert(!buffer.empty());

    const auto forced_policy_factor = GetParameters()->forced_policy_factor;
    for (const auto &x : buffer) {
        const auto visits = x.first;
        const auto vertex = x.second;
        auto node = GetChild(vertex);

        auto forced_playouts = std::sqrt(forced_policy_factor *
                                             node->GetPolicy() *
                                             float(parentvisits));
        auto new_visits = std::max(visits - int(forced_playouts), 0);
        node->SetVisits(new_visits);
    }

    while (true) {
        auto node = PuctSelectChild(color_, false);
        if (node->GetVertex() == most_visits_move) {
            break;
        }
        node->SetActive(false);
    }

    for (const auto &x : buffer) {
        const auto visits = x.first;
        const auto vertex = x.second;
        GetChild(vertex)->SetVisits(visits);
    }
}

Node *Node::ProbSelectChild() {
    WaitExpanded();
    assert(HaveChildren());

    Edge* best_node = nullptr;
    float best_prob = std::numeric_limits<float>::lowest();

    for (auto &child : children_) {
        const auto node = child.Get();
        const bool is_pointer = node == nullptr ? false : true;

        auto prob = child.Data()->policy;

        // The node was pruned. Skip this time.
        if (is_pointer && !node->IsActive()) {
            continue;
        }

        // The node was expending. Give it very bad value.
        if (is_pointer && node->IsExpending()) {
            prob = -1.0f + prob;
        }

        if (prob > best_prob) {
            best_prob = prob;
            best_node = &child;
        }
    }

    Inflate(*best_node);
    return best_node->Get();
}

Node *Node::PuctSelectChild(const int color, const bool is_root) {
    WaitExpanded();
    assert(HaveChildren());
    assert(color == color_);

    // Gather all parent's visits.
    int parentvisits = 0;
    float total_visited_policy = 0.0f;
    for (const auto &child : children_) {
        const auto node = child.Get();
        if (!node) {
            // There is no visits in uninflated node.
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

    const auto fpu_reduction_factor = is_root ? GetParameters()->fpu_root_reduction   : GetParameters()->fpu_reduction;
    const auto cpuct_init           = is_root ? GetParameters()->cpuct_root_init      : GetParameters()->cpuct_init;
    const auto cpuct_base           = is_root ? GetParameters()->cpuct_root_base      : GetParameters()->cpuct_base;
    const auto draw_factor          = is_root ? GetParameters()->draw_root_factor     : GetParameters()->draw_factor;
    const auto forced_policy_factor = is_root ? GetParameters()->forced_policy_factor : 0.0f;
    const auto noise                = is_root ? GetParameters()->dirichlet_noise      : false;
    const auto score_utility_factor = GetParameters()->score_utility_factor;

    const float cpuct         = cpuct_init + std::log((float(parentvisits) + cpuct_base + 1) / cpuct_base);
    const float numerator     = std::sqrt(float(parentvisits));
    const float fpu_reduction = fpu_reduction_factor * std::sqrt(total_visited_policy);
    const float fpu_value     = GetNetEval(color) - fpu_reduction;
    const float score         = GetFinalScore(color);

    Edge* best_node = nullptr;
    float best_value = std::numeric_limits<float>::lowest();

    for (auto &child : children_) {
        const auto node = child.Get();
        const bool is_pointer = node == nullptr ? false : true;

        // The node was pruned. Skip this time.
        if (is_pointer && !node->IsActive()) {
            continue;
        }

        // Apply First Play Urgency(FPU). We tend to search the expended
        // node in the PUCT algorithm. So give the unexpended node a little
        // bad value.
        float q_value = fpu_value;

        if (is_pointer) {
            if (node->IsExpending()) {
                // Like virtual loss, give it a bad value because there is another
                // thread in this node.
                q_value = -1.0f - fpu_reduction;
            } else if (node->GetVisits() > 0) {
                // Transfer Win-Draw-Loss to side-to-move value (Q value).
                const float eval = node->GetEval(color);
                const float draw_value = node->GetDraw() * draw_factor;
                q_value = eval + draw_value;
            }
        }

        float denom = 1.0f;
        float utility = 0.0f; // score utility value
        float bonus = 0.0f; // force playouts bonus
        if (is_pointer) {
            const auto visits = node->GetVisits();
            denom += visits;

            if (visits > 0) {
                // Heuristic value for score lead.
                utility += node->GetScoreUtility(color, score_utility_factor, score);
            }

            // According to Kata Go, we want to improve exploration in
            // the self-play. Give it a large bonus if the visits is not
            // enough.
            int forced_playouts = std::sqrt(forced_policy_factor *
                                                node->GetPolicy() *
                                                float(parentvisits));

            bonus += (int) (forced_playouts - denom + 1.0f);
            bonus *= 10;
            bonus = std::max(bonus, 0.0f);
        }

        // PUCT algorithm
        const float psa = GetUctPolicy(child, noise);
        const float puct = cpuct * psa * (numerator / denom);
        const float value = q_value + puct + utility + bonus;
        assert(value > std::numeric_limits<float>::lowest());

        if (value > best_value) {
            best_value = value;
            best_node = &child;
        }
    }

    Inflate(*best_node);
    return best_node->Get();
}

int Node::RandomizeFirstProportionally(float random_temp) {
    auto select_vertex = -1;
    auto accum = float{0.0f};
    auto accum_vector = std::vector<std::pair<float, int>>{};

    for (const auto &child : children_) {
        auto node = child.Get();
        const auto visits = node->GetVisits();
        const auto vertex = node->GetVertex();
        if (visits > GetParameters()->random_min_visits) {
            accum += std::pow((float)visits, (1.0 / random_temp));
            accum_vector.emplace_back(std::pair<float, int>(accum, vertex));
        }
    }

    auto distribution = std::uniform_real_distribution<float>{0.0, accum};
    auto pick = distribution(Random<kXoroShiro128Plus>::Get());
    auto size = accum_vector.size();

    for (auto idx = size_t{0}; idx < size; ++idx) {
        if (pick < accum_vector[idx].first) {
            select_vertex = accum_vector[idx].second;
            break;
        }
    }

    return select_vertex;
}

void Node::Update(const NodeEvals *evals) {
    const float eval = evals->black_wl;
    const float old_eval = accumulated_black_wl_.load(std::memory_order_relaxed);
    const float old_visits = visits_.load(std::memory_order_relaxed);

    const float old_delta = old_visits > 0 ? eval - old_eval / old_visits : 0.0f;
    const float new_delta = eval - (old_eval + eval) / (old_visits + 1);

    // TODO: According to Kata Go, It is not necessary to use
    //       Welford's online algorithm. The accuracy of simplify
    //       algorithm is enough.
    // Welford's online algorithm for calculating variance.
    const float delta = old_delta * new_delta;

    visits_.fetch_add(1, std::memory_order_relaxed);
    AtomicFetchAdd(squared_eval_diff_, delta);
    AtomicFetchAdd(accumulated_black_wl_, eval);
    AtomicFetchAdd(accumulated_draw_, evals->draw);
    AtomicFetchAdd(accumulated_black_fs_, evals->black_final_score);

    {
        std::lock_guard<std::mutex> lock(update_mtx_);
        for (int idx = 0; idx < kNumIntersections; ++idx) {
            accumulated_black_ownership_[idx] += evals->black_ownership[idx];
        }
    }
}

void Node::ApplyEvals(const NodeEvals *evals) {
    black_wl_ = evals->black_wl;
    draw_ = evals->draw;
    black_fs_ = evals->black_final_score;

    std::copy(std::begin(evals->black_ownership),
                  std::end(evals->black_ownership),
                  std::begin(black_ownership_));
}

std::array<float, kNumIntersections> Node::GetOwnership(int color) const {
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

float Node::GetScoreUtility(const int color, float factor, float parent_score) const {
    return std::tanh(factor * (GetFinalScore(color) - parent_score));
}

float Node::GetVariance(const float default_var, const int visits) const {
    return visits > 1 ? squared_eval_diff_.load(std::memory_order_relaxed) / (visits - 1) : default_var;
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
    const auto z = LcbEntries::Get().CachedTQuantile(visits - 1);
    
    return mean - z * stddev;
}

std::string Node::ToVerboseString(GameState &state, const int color) {
    auto out = std::ostringstream{};
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
            << std::setw(space) << "S"
            << std::endl;

    for (auto &lcb : lcblist) {
        const auto lcb_value = lcb.first > 0.0f ? lcb.first : 0.0f;
        const auto vertex = lcb.second;

        auto child = GetChild(vertex);
        const auto visits = child->GetVisits();
        const auto pobability = child->GetPolicy();
        assert(visits != 0);

        const auto final_score = child->GetFinalScore(color);
        const auto eval = child->GetEval(color, false);
        const auto draw = child->GetDraw();

        const auto pv_string = state.VertexToText(vertex) + ' ' + child->GetPvString(state);

        const auto visit_ratio = static_cast<float>(visits) / (parentvisits - 1); // One is root visit.
        out << std::fixed << std::setprecision(2)
                << std::setw(6) << state.VertexToText(vertex)  // move
                << std::setw(10) << visits                     // visits
                << std::setw(space) << eval * 100.f            // win loss eval
                << std::setw(space) << lcb_value * 100.f       // LCB eval
                << std::setw(space) << draw * 100.f            // draw probability
                << std::setw(space) << pobability * 100.f      // move probability
                << std::setw(space) << visit_ratio * 100.f     // visits ratio
                << std::setw(space) << final_score             // score leading
                << std::setw(6) << "| PV:" << ' ' << pv_string // principal variation
                << std::endl;
    }

    auto nodes = size_t{0};
    auto edges = size_t{0};
    ComputeNodeCount(nodes, edges);

    const auto node_mem = sizeof(Node) + sizeof(Edge);
    const auto edge_mem = sizeof(Edge);

    // There is some error to compute memory used. It is because that
    // we may not collect all node conut. 
    const auto mem_used = static_cast<double>(nodes * node_mem + edges * edge_mem) / (1024.f * 1024.f);

    out << "Tree Status:" << std::endl
            << std::setw(9) << "nodes:"  << ' ' << nodes    << std::endl
            << std::setw(9) << "edges:"  << ' ' << edges    << std::endl
            << std::setw(9) << "memory:" << ' ' << mem_used << ' ' << "(MiB)" << std::endl;

    return out.str();
}

std::string Node::ToAnalyzeString(GameState &state, const int color) {
    // Gather analyzing string, you can see the detail here
    // https://github.com/SabakiHQ/Sabaki/blob/master/docs/guides/engine-analysis-integration.md

    auto out = std::ostringstream{};
    const auto lcblist = GetLcbList(color);

    int i = 0;

    for (auto &lcb : lcblist) {
        const auto lcb_value = lcb.first > 0.0f ? lcb.first : 0.0f;
        const auto vertex = lcb.second;

        auto child = GetChild(vertex);
        const auto final_score = child->GetFinalScore(color);
        const auto winrate = std::min(10000, (int)(10000 * child->GetEval(color, false)));
        const auto visits = child->GetVisits();
        const auto prior = std::min(10000, (int)(10000 * child->GetPolicy()));
        const auto pv_string = state.VertexToText(vertex) + ' ' + child->GetPvString(state);

        out << Format("info move %s visits %d winrate %d scoreLead %f prior %d lcb %d order %d pv %s",
                         state.VertexToText(vertex).c_str(),
                         visits,
                         winrate,
                         final_score,
                         prior,
                         std::min(10000, (int)(10000 * lcb_value)),
                         i++,
                         pv_string.c_str()
                     );
    }

    out << std::endl;

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

Node *Node::GetChild(const int vertex) {
    for (auto & child : children_) {
        if (vertex == child.Data()->vertex) {
            Inflate(child);
            return child.Get();
        }
    }
    return nullptr;
}

Node *Node::PopChild(const int vertex) {
    auto node = GetChild(vertex);
    if (node) {
        auto ite = std::remove_if(std::begin(children_), std::end(children_),
                                  [node](Edge &ele) {
                                     return ele.Get() == node;
                                  });
        children_.erase(ite, std::end(children_));
    }
    return node;
}

std::vector<std::pair<float, int>> Node::GetLcbList(const int color) {
    WaitExpanded();
    assert(HaveChildren());

    auto lcb_reduction = GetParameters()->lcb_reduction;
    auto parents_visits = (float)GetVisits();
    auto list = std::vector<std::pair<float, int>>{};

    for (const auto & child : children_) {
        const auto node = child.Get();
        const bool is_pointer = node == nullptr ? false : true;

        if (!is_pointer || !node->IsActive()) {
            continue;
        }

        const auto visits = node->GetVisits();
        if (visits > 0) {
            const auto lcb = node->GetLcb(color) * (1.f - lcb_reduction) + 
                                 lcb_reduction * ((float)visits/parents_visits);
            list.emplace_back(lcb, node->GetVertex());
        }
    }

    std::stable_sort(std::rbegin(list), std::rend(list));
    return list;
}

int Node::GetBestMove() {
    WaitExpanded();
    assert(HaveChildren());

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
    evals.black_final_score = black_fs_;

    for (int idx = 0; idx < kNumIntersections; ++idx) {
        evals.black_ownership[idx] = black_ownership_[idx];
    }

    return evals;
}

const std::vector<Node::Edge> &Node::GetChildren() const {
    return children_;
}

Parameters *Node::GetParameters() {
    return data_.parameters;
}

int Node::GetThreads() const {
    return running_threads_.load(std::memory_order_relaxed);
}

int Node::GetVirtualLoss() const {
    return GetThreads() * VIRTUAL_LOSS_COUNT;
}

int Node::GetVertex() const {
    return data_.vertex;
}

float Node::GetPolicy() const {
    return data_.policy;
}

int Node::GetVisits() const {
    return visits_.load(std::memory_order_relaxed);
}

float Node::GetNetFinalScore(const int color) const {
    if (color == kBlack) {
        return black_fs_;
    }
    return 0.0f - black_fs_;
}

float Node::GetFinalScore(const int color) const {
    auto score = accumulated_black_fs_.load(std::memory_order_relaxed) / GetVisits();

    if (color == kBlack) {
        return score;
    }
    return 0.0f - score;
}

float Node::GetNetDraw() const {
    return draw_;
}

float Node::GetDraw() const {
    return accumulated_draw_.load(std::memory_order_relaxed) / GetVisits();
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
        // If the node is seaching, punish it.
        virtual_loss = GetVirtualLoss();
    }

    auto visits = GetVisits() + virtual_loss;
    assert(visits >= 0);

    auto accumulated_wl = accumulated_black_wl_.load(std::memory_order_relaxed);
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
    for (auto &child : children_) {
         Inflate(child);
    }
}

void Node::ReleaseAllChildren() {
    for (auto &child : children_) {
         Release(child);
    }
}

void Node::Inflate(Edge& child) {
    if (child.Inflate()) {
        // do nothing...
    }
}

void Node::Release(Edge& child) {
    if (child.Release()) {
        // do nothing...
    }
}

bool Node::HaveChildren() const { 
    return color_ != kInvalid;
}

void Node::IncrementThreads() {
    running_threads_.fetch_add(1, std::memory_order_relaxed);
}

void Node::DecrementThreads() {
    running_threads_.fetch_sub(1, std::memory_order_relaxed);
}

void Node::SetActive(const bool active) {
    if (IsValid()) {
        StatusType v = active ? StatusType::kActive : StatusType::kPruned;
        status_.store(v, std::memory_order_relaxed);
    }
}

void Node::InvaliNode() {
    if (IsValid()) {
        status_.store(StatusType::kInvalid, std::memory_order_relaxed);
    }
}

bool Node::IsPruned() const {
    return status_.load(std::memory_order_relaxed) == StatusType::kPruned;
}

bool Node::IsActive() const {
    return status_.load(std::memory_order_relaxed) == StatusType::kActive;
}

bool Node::IsValid() const {
    return status_.load(std::memory_order_relaxed) != StatusType::kInvalid;
}

bool Node::AcquireExpanding() {
    auto expected = ExpandState::kInitial;
    auto newval = ExpandState::kExpanding;
    return expand_state_.compare_exchange_strong(expected, newval, std::memory_order_acquire);
}

void Node::ExpandDone() {
    auto v = expand_state_.exchange(ExpandState::kExpanded, std::memory_order_release);
#ifdef NDEBUG
    (void) v;
#endif
    assert(v == ExpandState::kExpanding);
}

void Node::ExpandCancel() {
    auto v = expand_state_.exchange(ExpandState::kInitial, std::memory_order_release);
#ifdef NDEBUG
    (void) v;
#endif
    assert(v == ExpandState::kExpanding);
}

void Node::WaitExpanded() const {
    while (true) {
        //TODO: Sleep some time because it is not busy lock. Implement
        //      it and test the performance.

        auto v = expand_state_.load(std::memory_order_acquire);
        if (v == ExpandState::kExpanded) {
            break;
        }
    }
}

bool Node::Expandable() const {
    return expand_state_.load(std::memory_order_relaxed) == ExpandState::kInitial;
}

bool Node::IsExpending() const {
    return expand_state_.load(std::memory_order_relaxed) == ExpandState::kExpanding;
}

bool Node::IsExpended() const {
    return expand_state_.load(std::memory_order_relaxed) == ExpandState::kExpanded;
}

void Node::ApplyDirichletNoise(const float alpha) {
    auto child_cnt = children_.size();
    auto buffer = std::vector<float>(child_cnt);
    auto gamma = std::gamma_distribution<float>(alpha, 1.0f);

    std::generate(std::begin(buffer), std::end(buffer),
                      [&gamma] () { return gamma(Random<kXoroShiro128Plus>::Get()); });

    auto sample_sum =
        std::accumulate(std::begin(buffer), std::end(buffer), 0.0f);

    auto &dirichlet = GetParameters()->dirichlet_buffer;
    dirichlet.fill(0.0f);

    // If the noise vector sums to 0 or a denormal, then don't try to
    // normalize.
    if (sample_sum < std::numeric_limits<float>::min()) {
        std::fill(std::begin(buffer), std::end(buffer), 0.0f);
        return;
    }

    for (auto &v : buffer) {
        v /= sample_sum;
    }

    InflateAllChildren();
    for (auto i = size_t{0}; i < child_cnt; ++i) {
        const auto vertex = children_[i].Data()->vertex;
        dirichlet[vertex] = buffer[i];
    }
}

float Node::GetUctPolicy(Node::Edge& child, bool noise) {
    auto policy = child.Data()->policy;
    if (noise) {
        const auto vertex = child.Data()->vertex;
        const auto epsilon = GetParameters()->dirichlet_epsilon;
        const auto eta_a = GetParameters()->dirichlet_buffer[vertex];
        policy = policy * (1 - epsilon) + epsilon * eta_a;
    }
    return policy;
}

void Node::SetVisits(int v) {
    visits_.store(v, std::memory_order_relaxed);
}

void Node::SetPolicy(float p) {
    data_.policy = p;
}

void Node::ComputeNodeCount(size_t &nodes, size_t &edges) {
    // Use DFS to search all nodes.
    auto stk = std::stack<Node *>{};

    // Start search from this node.
    stk.emplace(Get());
    nodes++;

    while (!stk.empty()) {
        Node * node = stk.top();
        stk.pop();

        const auto &children = node->GetChildren();

        // Because we want compute the memory used, collect
        // all types of nodes. Including pruned and invalid node.
        for (const auto &child : children) {
            node = child.Get();
            const bool is_pointer = node == nullptr ? false : true;

            if (is_pointer) {
                if (!(node->IsExpending())) {
                    // If the node is expending, skip the
                    // the node.
                    stk.emplace(node);
                }
                nodes++;
            } else {
                edges++;
            }
        }
    }
}
