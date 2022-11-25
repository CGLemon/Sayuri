#include "mcts/node.h"
#include "mcts/lcb.h"
#include "mcts/rollout.h"
#include "utils/atomic.h"
#include "utils/random.h"
#include "utils/format.h"
#include "game/symmetry.h"

#include <cassert>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <stack>
#include <iostream>

#define VIRTUAL_LOSS_COUNT (3)

Node::Node(std::int16_t vertex, float policy) {
    vertex_ = vertex;
    policy_ = policy;
}

Node::~Node() {
    assert(GetThreads() == 0);
    ReleaseAllChildren();
}

bool Node::PrepareRootNode(Network &network,
                               GameState &state,
                               NodeEvals &node_evals,
                               AnalysisConfig &config,
                               std::vector<float> &dirichlet) {
    const auto is_root = true;
    const auto success = ExpandChildren(network, state, node_evals, config, is_root);
    assert(HaveChildren());

    InflateAllChildren();
    if (param_->dirichlet_noise) {
        // Generate the dirichlet noise and gather it.
        const auto legal_move = children_.size();
        const auto factor = param_->dirichlet_factor;
        const auto init = param_->dirichlet_init;
        const auto alpha = init * factor / static_cast<float>(legal_move);

        ApplyDirichletNoise(alpha);

        const auto num_intersections = state.GetNumIntersections();
        const auto board_size = state.GetBoardSize();
        dirichlet.resize(num_intersections+1);

        for (auto idx = 0; idx < num_intersections; ++idx) {
            const auto x = idx % board_size;
            const auto y = idx / board_size;
            const auto vtx = state.GetVertex(x, y);
            dirichlet[idx] = param_->dirichlet_buffer[vtx];
        }
        dirichlet[num_intersections] = param_->dirichlet_buffer[kPass];
    }

    return success;
}

bool Node::ExpandChildren(Network &network,
                              GameState &state,
                              NodeEvals &node_evals,
                              AnalysisConfig &config,
                              const bool is_root) {
    // The node must be the first time to expand and is not the terminate node.
    assert(state.GetPasses() < 2);
    if (HaveChildren()) {
        return false;
    }

    // Try to acquire the owner.
    if (!AcquireExpanding()) {
        return false;
    }

    // Get network computation result.
    const float temp = is_root ? param_->root_policy_temp : param_->policy_temp;

    auto raw_netlist = Network::Result{};
    color_ = state.GetToMove();

    if (param_->no_dcnn &&
            !(param_->root_dcnn && is_root)) {
        ApplyNoDcnnPolicy(state, color_, raw_netlist);
    } else {
        raw_netlist = network.GetOutput(state, Network::kRandom, temp);
    }

    // Store the network reuslt.
    ApplyNetOutput(state, raw_netlist, node_evals, color_);

    // For children...
    auto nodelist = std::vector<Network::PolicyVertexPair>{};
    auto allow_pass = true;
    auto legal_accumulate = 0.0f;

    const auto board_size = state.GetBoardSize();
    const auto num_intersections = state.GetNumIntersections();
    const auto safe_area = state.GetStrictSafeArea();

    // For symmetry pruning.
    bool apply_symm_pruning = param_->symm_pruning &&
                                  board_size >= state.GetMoveNumber();
    auto moves_hash = std::vector<std::uint64_t>{};
    auto symm_base_hash = std::vector<std::uint64_t>(Symmetry::kNumSymmetris, 0ULL);

    for (int symm = Symmetry::kIdentitySymmetry;
             apply_symm_pruning && symm < Symmetry::kNumSymmetris; ++symm) {
        symm_base_hash[symm] = state.ComputeSymmetryHash(symm);
    }

    // Prune the illegal moves or some bad move.
    for (int idx = 0; idx < num_intersections; ++idx) {
        const auto x = idx % board_size;
        const auto y = idx / board_size;
        const auto vtx = state.GetVertex(x, y);
        const auto policy = raw_netlist.probabilities[idx];

        // Prune the illegal and unwise move.
        int movenum = state.GetMoveNumber();
        if (!state.IsLegalMove(vtx, color_,
                [movenum, &config](int vtx, int color){
                    return !config.IsLegal(vtx, color, movenum);
                }) 
                    || safe_area[idx]) {
            continue;
        }

        // Prune the symmetry moves. May reduce some perfomance.
        if (apply_symm_pruning) {
            bool hash_found = false;
            for (int symm = Symmetry::kIdentitySymmetry+1;
                     symm < Symmetry::kNumSymmetris && !hash_found; ++symm) {
                const auto symm_vtx = Symmetry::Get().TransformVertex(board_size, symm, vtx);
                const auto symm_hash = symm_base_hash[symm] ^ state.GetMoveHash(symm_vtx, color_);
                hash_found = (std::find(std::begin(moves_hash), std::end(moves_hash), symm_hash) != std::end(moves_hash));
            }

            if (!hash_found) {
                moves_hash.emplace_back(state.GetHash() ^ state.GetMoveHash(vtx, color_));
            } else {
                legal_accumulate += policy; // The pruned node is a legal move.
                continue;
            }
        }

        // Prune the super ko move.
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

    // There ara too many legal moves. Disable the pass move.
    if ((int)nodelist.size() > 3*num_intersections/4) {
        allow_pass = false;
    }

    // The pass is always legal. If there is no legal move except for pass, forcing
    // to open the pass node.
    if (allow_pass || nodelist.empty()) {
        nodelist.emplace_back(raw_netlist.pass_probability, kPass);
        legal_accumulate += raw_netlist.pass_probability;
    }

    if (legal_accumulate < 1e-8f) {
        // It will be happened if the policy focuses on the illegal moves.
        for (auto &node : nodelist) {
            node.first = 1.f/nodelist.size();
        }
    } else {
        for (auto &node : nodelist) {
            // Adjust the policy.
            node.first /= legal_accumulate;
        }
    }

    // Extend the nodes.
    LinkNodeList(nodelist);

    // Release the owner.
    ExpandDone();

    return true;
}

void Node::LinkNodeList(std::vector<Network::PolicyVertexPair> &nodelist) {
    // Besure that the best policy is on the top.
    std::stable_sort(std::rbegin(nodelist), std::rend(nodelist));

    for (const auto &node : nodelist) {
        const auto vertex = (std::uint16_t)node.second;
        const auto policy = node.first;
        children_.emplace_back(vertex, policy);
    }
    assert(!children_.empty());
}

void Node::ApplyNetOutput(GameState &state,
                        const Network::Result &raw_netlist,
                        NodeEvals& node_evals, const int color) {
    auto black_ownership = std::array<float, kNumIntersections>{};
    auto black_fs = float(0.f);
    auto draw =raw_netlist.wdl[1];

    // Compute the black side to move evals.
    auto wl = float(0.5f);

    if (param_->use_stm_winrate) {
        wl = raw_netlist.stm_winrate;
    } else {
        wl = (raw_netlist.wdl[0] - raw_netlist.wdl[2] + 1) / 2;
    }

    auto final_score = raw_netlist.final_score;

    if (color == kWhite) {
        wl = 1.0f - wl;
        final_score = 0.0f - final_score;
    }

    black_wl_ = wl;
    black_fs = final_score;

    for (int idx = 0; idx < kNumIntersections; ++idx) {
        auto owner = raw_netlist.ownership[idx];
        if (color == kWhite) {
            owner = 0.f - owner;
        }
        black_ownership[idx] = owner;
        avg_black_ownership_[idx] = 0.f;
    }

    // Do rollout if we disable the DCNN or the DCNN does not
    // support the ownership.
    if (param_->use_rollout || param_->no_dcnn) {
        float mc_black_rollout_score;
        float mc_black_rollout_res = GetBlackRolloutResult(
                                         state,
                                         black_ownership.data(),
                                         mc_black_rollout_score);
        if (param_->no_dcnn) {
            black_wl_ = mc_black_rollout_res;
            black_fs = mc_black_rollout_score;
        }
    }

    // Store the network evals.
    node_evals.black_wl = black_wl_;
    node_evals.draw = draw;
    node_evals.black_final_score = black_fs;

    for (int idx = 0; idx < kNumIntersections; ++idx) {
        node_evals.black_ownership[idx] = black_ownership[idx];
    }
}

void Node::ApplyNoDcnnPolicy(GameState &state, const int color,
                                 Network::Result &raw_netlist) const {
    const auto num_intersections = state.GetNumIntersections();
    auto policy = state.GetGammasPolicy(color);

    for (int idx = 0; idx < num_intersections; ++idx) {
        raw_netlist.probabilities[idx] = policy[idx];
        raw_netlist.ownership[idx] = 0.f; // set zero...
    }

    raw_netlist.board_size = state.GetBoardSize();
    raw_netlist.komi = state.GetKomi();

    // Give the pass move a little value in order to avoid the 
    // bug if there is no legal moves.
    raw_netlist.pass_probability = 0.1f/num_intersections;
    raw_netlist.final_score = 0.f; // set zeros...
    raw_netlist.wdl = {0.5f, 0, 0.5f}; // set draw value...
    raw_netlist.wdl_winrate = 0.5f; // set draw value...
    raw_netlist.stm_winrate = 0.5f; // set draw value...
}

bool Node::SetTerminal() {
    if (!AcquireExpanding()) {
        return false;
    }

    color_ = kInvalid; // no children

    ExpandDone();
    return true;
}

float Node::ComputeKlDivergence() {
    const auto vtx = GetBestMove();
    int parentvisits = 0;
    int best_visits = 0;

    for (const auto &child : children_) {
        const auto node = child.Get();
        if (node && node->IsActive()) {
            const auto visits = node->GetVisits();

            parentvisits += visits;
            if (node->GetVertex() == vtx) {
                best_visits = visits;
            }
        }
    }

    if (parentvisits == best_visits) {
        return 0;
    }
    if (parentvisits == 0 || best_visits == 0) {
        return -1;
    }

    return -std::log((float)best_visits / parentvisits);
}

float Node::ComputeTreeComplexity() {
    const auto visits = GetVisits();
    if (visits <= 1) {
        return 0;
    }

    const auto variance = GetLcbVariance(1.0f, visits);
    const auto stddev = std::sqrt(100 * variance);

    return stddev;
}

Node *Node::ProbSelectChild() {
    WaitExpanded();
    assert(HaveChildren());

    Edge* best_node = nullptr;
    float best_prob = std::numeric_limits<float>::lowest();

    for (auto &child : children_) {
        const auto node = child.Get();
        const bool is_pointer = node != nullptr;

        // The node is pruned or invalid. Skip it.
        if (is_pointer && !node->IsActive()) {
            continue;
        }

        auto prob = child.GetPolicy();

        // The node is expanding. Give it very bad value.
        if (is_pointer && node->IsExpanding()) {
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
    // assert(color == color_);

    const auto noise  = is_root ? param_->dirichlet_noise : false;
    const auto gumbel = is_root ? param_->gumbel_noise    : false;
    auto gumbel_type1 = std::extreme_value_distribution<float>(0, 1);

    // Gather all parent's visits.
    auto policy_buf = std::vector<float>(kNumVertices + 10, -1e6f);
    int parentvisits = 0;
    float total_visited_policy = 0.0f;
    for (auto &child : children_) {
        const auto node = child.Get();

        float vpol = GetSearchPolicy(child, noise);
        if (gumbel) {
            vpol = std::log(vpol) + gumbel_type1(Random<>::Get());
        }
        policy_buf[child.GetVertex()] = vpol;

        if (!node) {
            // There is no visits in uninflated node.
            continue;
        }
        if (node->IsValid()) {
            // The node status is pruned or active.
            const auto visits = node->GetVisits();
            parentvisits += visits;
            if (visits > 0) {
                total_visited_policy += child.GetPolicy();
            }
        }
    }

    if (gumbel) {
        policy_buf = Network::Softmax(policy_buf, 1.f);
    }

    const auto fpu_reduction_factor = is_root ? param_->fpu_root_reduction   : param_->fpu_reduction;
    const auto cpuct_init           = is_root ? param_->cpuct_root_init      : param_->cpuct_init;
    const auto cpuct_base           = is_root ? param_->cpuct_root_base      : param_->cpuct_base;
    const auto draw_factor          = is_root ? param_->draw_root_factor     : param_->draw_factor;
    const auto score_utility_factor = param_->score_utility_factor;
    const auto score_utility_div    = param_->score_utility_div;

    const float cpuct         = cpuct_init + std::log((float(parentvisits) + cpuct_base + 1) / cpuct_base);
    const float numerator     = std::sqrt(float(parentvisits));
    const float fpu_reduction = fpu_reduction_factor * std::sqrt(total_visited_policy);
    const float fpu_value     = GetNetWL(color) - fpu_reduction;
    const float score         = GetFinalScore(color);

    Edge* best_node = nullptr;
    float best_value = std::numeric_limits<float>::lowest();

    for (auto &child : children_) {
        const auto node = child.Get();
        const bool is_pointer = node != nullptr;

        // The node is pruned or invalid. Skip it.
        if (is_pointer && !node->IsActive()) {
            continue;
        }

        // Apply First Play Urgency(FPU). We tend to search the expanded
        // node in the PUCT algorithm. So give the unexpanded node a little
        // bad value.
        float q_value = fpu_value;

        if (is_pointer) {
            if (node->IsExpanding()) {
                // Like virtual loss, give it a bad value because there are other
                // threads in this node.
                q_value = -1.0f - fpu_reduction;
            } else if (node->GetVisits() > 0) {
                // Transfer win-draw-loss to side-to-move value (Q value).
                const float eval = node->GetWL(color);
                const float draw_value = node->GetDraw() * draw_factor;
                q_value = eval + draw_value;
            }
        }

        float denom = 1.0f;
        float utility = 0.0f; // the utility value

        if (is_pointer) {
            const auto visits = node->GetVisits();
            denom += visits;

            if (visits > 0) {
                // Heuristic value for score lead.
                utility += score_utility_factor *
                               node->GetScoreUtility(color, score_utility_div, score);
            }
        }

        // PUCT algorithm
        const float psa = policy_buf[child.GetVertex()];
        const float puct = cpuct * psa * (numerator / denom);
        const float value = q_value + puct + utility;
        assert(value > std::numeric_limits<float>::lowest());

        if (value > best_value) {
            best_value = value;
            best_node = &child;
        }
    }

    Inflate(*best_node);
    return best_node->Get();
}

Node *Node::UctSelectChild(const int color, const bool is_root, const GameState &state) {
    WaitExpanded();
    assert(HaveChildren());
    // assert(color == color_);

    Edge* best_node = nullptr;
    float best_value = std::numeric_limits<float>::lowest();

    const int parentvisits = std::max(1, GetVisits());
    const float numerator = std::log((float)parentvisits);
    const float cpuct = is_root ? param_->cpuct_root_init : param_->cpuct_init;
    const float parent_qvalue = GetWL(color, false);

    std::vector<Edge*> edge_buf;

    for (auto &child : children_) {
        edge_buf.emplace_back(&child);
    }

    int width = std::max(ComputeWidth(parentvisits), 1);
    int i = 0;

    for (auto edge_ptr : edge_buf) {
        auto &child = *edge_ptr;

        if (state.board_.IsCaptureMove(edge_ptr->GetVertex(), color)) {
            width += 1;
        }

        if (++i > width) {
            break;
        }

        const auto node = child.Get();
        const bool is_pointer = node != nullptr;

        // The node is pruned or invalid. Skip it.
        if (is_pointer && !node->IsActive()) {
            continue;
        }

        float q_value = parent_qvalue;
        int child_visits = 0;

        if (is_pointer) {
            child_visits = node->GetVisits();

            if (node->IsExpanding()) {
                q_value = -1.0f; // Give it a bad value.
            } else if (child_visits > 0) {
                q_value = node->GetWL(color);
            }
        }

        // UCT algorithm
        const float denom = 1.0f + child_visits;
        const float psa = child.GetPolicy();
        const float bouns = 1.0f * std::sqrt(1000.f / ((float)parentvisits + 1000.f)) * psa;
        const float uct = cpuct * std::sqrt(numerator / denom);
        float value = q_value + uct + bouns;
        assert(value > std::numeric_limits<float>::lowest());

        if (value > best_value) {
            best_value = value;
            best_node = edge_ptr;
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
        if (visits > param_->random_min_visits) {
            accum += std::pow((float)visits, (1.0 / random_temp));
            accum_vector.emplace_back(std::pair<float, int>(accum, vertex));
        }
    }

    auto distribution = std::uniform_real_distribution<float>{0.0, accum};
    auto pick = distribution(Random<>::Get());
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
    auto WelfordDelta = [](double eval,
                               double old_acc_eval,
                               int old_visits) {
        const double old_delta = old_visits > 0 ? eval - old_acc_eval / old_visits : 0.0f;
        const double new_delta = eval - (old_acc_eval + eval) / (old_visits+1);
        const double delta = old_delta * new_delta;
        return delta;
    };

    // type casting
    const double eval = evals->black_wl;
    const double draw = evals->draw;
    const double black_final_score = evals->black_final_score;
    const double old_acc_eval = accumulated_black_wl_.load(std::memory_order_relaxed);

    const int old_visits = visits_.load(std::memory_order_relaxed);

    // TODO: According to Kata Go, It is not necessary to use
    //       Welford's online algorithm. The accuracy of simplify
    //       algorithm is enough.
    // Welford's online algorithm for calculating variance.
    const double delta = WelfordDelta(eval, old_acc_eval, old_visits);

    visits_.fetch_add(1, std::memory_order_relaxed);
    AtomicFetchAdd(squared_eval_diff_   , delta);
    AtomicFetchAdd(accumulated_black_wl_, eval);
    AtomicFetchAdd(accumulated_draw_    , draw);
    AtomicFetchAdd(accumulated_black_fs_, black_final_score);

    {
        std::lock_guard<std::mutex> lock(os_mtx_);
        for (int idx = 0; idx < kNumIntersections; ++idx) {
            const double eval_owner = evals->black_ownership[idx];
            const double avg_owner  = avg_black_ownership_[idx];
            const double diff_owner = (eval_owner - avg_owner) / (old_visits+1);

            avg_black_ownership_[idx] += diff_owner;
        }
    }
}

void Node::ApplyEvals(const NodeEvals *evals) {
    black_wl_ = evals->black_wl;
}

std::array<float, kNumIntersections> Node::GetOwnership(int color) {
    std::lock_guard<std::mutex> lock(os_mtx_);

    auto out = std::array<float, kNumIntersections>{};
    for (int idx = 0; idx < kNumIntersections; ++idx) {
        auto owner = avg_black_ownership_[idx];
        if (color == kWhite) {
            owner = 0.f - owner;
        }
        out[idx] = owner;
    }
    return out;
}

float Node::GetScoreUtility(const int color, float div, float parent_score) const {
    return std::tanh(((GetFinalScore(color) - parent_score))/div);
}

float Node::GetLcbVariance(const float default_var, const int visits) const {
    return visits > 1 ?
               squared_eval_diff_.load(std::memory_order_relaxed) / (visits - 1) :
               default_var;
}

float Node::GetLcb(const int color) const {
    // The Lower confidence bound of winrate.
    // See the LCB issues here: https://github.com/leela-zero/leela-zero/pull/2290

    const auto visits = GetVisits();
    if (visits <= 1) {
        // We can not get the variance in the first visit. Return
        // the large negative value.
        return GetPolicy() - 1e6f;
    }

    const auto mean = GetWL(color, false);
    const auto variance = GetLcbVariance(1.0f, visits);
    const auto stddev = std::sqrt(variance / float(visits));
    const auto z = LcbEntries::Get().CachedTQuantile(visits - 1);

    return mean - z * stddev;
}

std::string Node::ToVerboseString(GameState &state, const int color) {
    auto out = std::ostringstream{};
    const auto lcblist = GetLcbUtilityList(color);
    const auto parentvisits = GetVisits() - 1; // One is root visit.

    if (lcblist.empty()) {
         out << " * Search List: N/A" << std::endl;
        return out.str();
    }

    const auto space1 = 7;
    out << " * Search List:" << std::endl;
    out << std::setw(6) << "move"
            << std::setw(10) << "visits"
            << std::setw(space1) << "WL(%)"
            << std::setw(space1) << "LCB(%)"
            << std::setw(space1) << "D(%)"
            << std::setw(space1) << "P(%)"
            << std::setw(space1) << "N(%)"
            << std::setw(space1) << "S"
            << std::endl;

    for (auto &lcb_pair : lcblist) {
        const auto lcb = lcb_pair.first > 0.0f ? lcb_pair.first : 0.0f;
        const auto vertex = lcb_pair.second;

        auto child = GetChild(vertex);
        const auto visits = child->GetVisits();
        const auto pobability = child->GetPolicy();
        assert(visits != 0);

        const auto final_score = child->GetFinalScore(color);
        const auto eval = child->GetWL(color, false);
        const auto draw = child->GetDraw();

        const auto pv_string = state.VertexToText(vertex) + ' ' + child->GetPvString(state);

        const auto visit_ratio = static_cast<float>(visits) / (parentvisits);
        out << std::fixed << std::setprecision(2)
                << std::setw(6) << state.VertexToText(vertex)  // move
                << std::setw(10) << visits                     // visits
                << std::setw(space1) << eval * 100.f           // win loss eval
                << std::setw(space1) << lcb * 100.f            // LCB eval
                << std::setw(space1) << draw * 100.f           // draw probability
                << std::setw(space1) << pobability * 100.f     // move probability
                << std::setw(space1) << visit_ratio * 100.f    // visits ratio
                << std::setw(space1) << final_score            // score lead
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

    const auto space2 = 10;
    out << " * Tree Status:" << std::endl
            << std::fixed << std::setprecision(4)
            << std::setw(space2) << "root KL:" << ' ' << ComputeKlDivergence() << std::endl
            << std::setw(space2) << "root C:"  << ' ' << ComputeTreeComplexity() << std::endl
            << std::setw(space2) << "nodes:"   << ' ' << nodes    << std::endl
            << std::setw(space2) << "edges:"   << ' ' << edges    << std::endl
            << std::setw(space2) << "memory:"  << ' ' << mem_used << ' ' << "(MiB)" << std::endl;

    return out.str();
}

std::string Node::OwnershipToString(GameState &state, const int color, std::string name, Node *node) {
    auto out = std::ostringstream{};
    const auto board_size = state.GetBoardSize();

    auto ownership = node->GetOwnership(color);

    out << name << ' ';
    for (int y = board_size-1; y >= 0; --y) {
        for (int x = 0; x < board_size; ++x) {
            out << Format("%.6f ", ownership[state.GetIndex(x,y)]);
        }
    }

    return out.str();
}

std::string Node::ToAnalysisString(GameState &state,
                                       const int color,
                                       AnalysisConfig &config) {
    // Gather the analysis string. You can see the detail here
    // https://github.com/SabakiHQ/Sabaki/blob/master/docs/guides/engine-analysis-integration.md

    auto out = std::ostringstream{};
    const auto lcblist = GetLcbUtilityList(color);

    if (lcblist.empty()) {
        return std::string{};
    }

    const auto root_visits = static_cast<float>(GetVisits() - 1);

    bool is_sayuri = config.is_sayuri;
    bool is_kata = config.is_kata;
    bool use_ownership = config.ownership;
    bool use_moves_ownership = config.moves_ownership;

    int order = 0;
    for (auto &lcb_pair : lcblist) {
        if (order+1 > config.max_moves) {
            break;
        }

        const auto lcb = lcb_pair.first > 0.0f ? lcb_pair.first : 0.0f;
        const auto vertex = lcb_pair.second;

        auto child = GetChild(vertex);
        const auto final_score = child->GetFinalScore(color);
        const auto winrate = child->GetWL(color, false);
        const auto visits = child->GetVisits();
        const auto prior = child->GetPolicy();
        const auto pv_string = state.VertexToText(vertex) + ' ' + child->GetPvString(state);

        if (param_->no_dcnn &&
                visits/root_visits < 0.01f) { // cut off < 1% children...
            continue;
        }

        if (is_sayuri) {
            const auto kl = child->ComputeKlDivergence();
            const auto complexity = child->ComputeTreeComplexity();
            out << Format("info move %s visits %d winrate %.6f scorelead %.6f prior %.6f lcb %.6f kl %.6f complexity %.6f order %d pv %s",
                             state.VertexToText(vertex).c_str(),
                             visits,
                             winrate,
                             final_score,
                             prior,
                             lcb,
                             kl,
                             complexity,
                             order,
                             pv_string.c_str()
                         );
        } else if (is_kata) {
            out << Format("info move %s visits %d winrate %.6f scoreLead %.6f prior %.6f lcb %.6f order %d pv %s",
                             state.VertexToText(vertex).c_str(),
                             visits,
                             winrate,
                             final_score,
                             prior,
                             lcb,
                             order,
                             pv_string.c_str()
                         );
        } else {
            out << Format("info move %s visits %d winrate %d scoreLead %.6f prior %d lcb %d order %d pv %s",
                             state.VertexToText(vertex).c_str(),
                             visits,
                             std::min(10000, (int)(10000 * winrate)),
                             final_score,
                             std::min(10000, (int)(10000 * prior)),
                             std::min(10000, (int)(10000 * lcb)),
                             order,
                             pv_string.c_str()
                         );
        }
        if (use_moves_ownership) {
            if (is_sayuri) {
                out << OwnershipToString(state, color, "movesownership", child);
            } else {
                out << OwnershipToString(state, color, "movesOwnership", child);
            }
        }
        order += 1;
    }

    if (use_ownership) {
        out << OwnershipToString(state, color, "ownership", this->Get());
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
        if (vertex == child.GetVertex()) {
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

std::vector<std::pair<float, int>> Node::GetLcbUtilityList(const int color) {
    WaitExpanded();
    assert(HaveChildren());

    const auto lcb_utility_factor = std::max(0.f, param_->lcb_utility_factor);
    const auto lcb_reduction = std::min(
                                   std::max(0.f, param_->lcb_reduction), 1.f);
    int parentvisits = 0;
    const auto score = GetFinalScore(color);
    const auto score_utility_div = param_->score_utility_div;

    auto list = std::vector<std::pair<float, int>>{};

    for (const auto & child : children_) {
        const auto node = child.Get();
        const bool is_pointer = node != nullptr;

        if (is_pointer && node->IsActive()) {
            parentvisits += node->GetVisits();
        }
    }

    for (const auto & child : children_) {
        const auto node = child.Get();
        const bool is_pointer = node != nullptr;

        // The node is uninflated, pruned or invalid. Skip it.
        if (!is_pointer || !node->IsActive()) {
            continue;
        }

        const auto visits = node->GetVisits();
        if (visits > 0) {
            auto lcb = node->GetLcb(color);
            auto utility = lcb_utility_factor *
                               node->GetScoreUtility(color, score_utility_div, score);
            const auto ulcb = (lcb + utility) * (1.f - lcb_reduction) + 
                                  lcb_reduction * ((float)visits/parentvisits);
            list.emplace_back(ulcb, node->GetVertex());
        }
    }

    std::stable_sort(std::rbegin(list), std::rend(list));
    return list;
}

int Node::GetBestMove() {
    WaitExpanded();
    assert(HaveChildren());

    auto lcblist = GetLcbUtilityList(color_);
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

const std::vector<Node::Edge> &Node::GetChildren() const {
    return children_;
}

void Node::SetParameters(Parameters * param) {
    param_ = param;
}

int Node::GetThreads() const {
    return running_threads_.load(std::memory_order_relaxed);
}

int Node::GetVirtualLoss() const {
    return GetThreads() * VIRTUAL_LOSS_COUNT;
}

int Node::GetVertex() const {
    return vertex_;
}

float Node::GetPolicy() const {
    return policy_;
}

int Node::GetVisits() const {
    return visits_.load(std::memory_order_relaxed);
}

float Node::GetFinalScore(const int color) const {
    auto score = accumulated_black_fs_.load(std::memory_order_relaxed) / GetVisits();

    if (color == kBlack) {
        return score;
    }
    return 0.0f - score;
}

float Node::GetDraw() const {
    return accumulated_draw_.load(std::memory_order_relaxed) / GetVisits();
}

float Node::GetNetWL(const int color) const {
    if (color == kBlack) {
        return black_wl_;
    }
    return 1.0f - black_wl_;
}

float Node::GetWL(const int color, const bool use_virtual_loss) const {
    auto virtual_loss = 0;

    if (use_virtual_loss) {
        // Punish the node if there are some threads in this 
        // sub-tree.
        virtual_loss = GetVirtualLoss();
    }

    auto visits = GetVisits() + virtual_loss;
    assert(visits >= 0);

    auto accumulated_wl = accumulated_black_wl_.load(std::memory_order_relaxed);
    if (color == kWhite && use_virtual_loss) {
        accumulated_wl += static_cast<double>(virtual_loss);
    }
    auto eval = accumulated_wl / visits;

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
        child.Get()->SetParameters(param_);
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

void Node::InvalidNode() {
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
        auto v = expand_state_.load(std::memory_order_acquire);
        if (v == ExpandState::kExpanded) {
            break;
        }

        // Wait some time to avoid busy waiting.
        std::this_thread::yield();
    }
}

bool Node::Expandable() const {
    return expand_state_.load(std::memory_order_relaxed) == ExpandState::kInitial;
}

bool Node::IsExpanding() const {
    return expand_state_.load(std::memory_order_relaxed) == ExpandState::kExpanding;
}

bool Node::IsExpanded() const {
    return expand_state_.load(std::memory_order_relaxed) == ExpandState::kExpanded;
}

void Node::ApplyDirichletNoise(const float alpha) {
    auto child_cnt = children_.size();
    auto buffer = std::vector<float>(child_cnt);
    auto gamma = std::gamma_distribution<float>(alpha, 1.0f);

    std::generate(std::begin(buffer), std::end(buffer),
                      [&gamma] () { return gamma(Random<>::Get()); });

    auto sample_sum =
        std::accumulate(std::begin(buffer), std::end(buffer), 0.0f);

    auto &dirichlet = param_->dirichlet_buffer;
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

    for (auto i = size_t{0}; i < child_cnt; ++i) {
        const auto vertex = children_[i].GetVertex();
        dirichlet[vertex] = buffer[i];
    }
}

float Node::GetSearchPolicy(Node::Edge& child, bool noise) {
    auto policy = child.GetPolicy();
    if (noise) {
        const auto vertex = child.GetVertex();
        const auto epsilon = param_->dirichlet_epsilon;
        const auto eta_a = param_->dirichlet_buffer[vertex];
        policy = policy * (1 - epsilon) + epsilon * eta_a;
    }
    return policy;
}

void Node::SetVisits(int v) {
    visits_.store(v, std::memory_order_relaxed);
}

void Node::SetPolicy(float p) {
    policy_ = p;
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
            const bool is_pointer = node != nullptr;

            if (is_pointer) {
                if (!(node->IsExpanding())) {
                    // If the node is expanding, skip the
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

void Node::MixLogitsCompletedQ(GameState &state, std::vector<float> &prob) {
    const auto num_intersections = state.GetNumIntersections();
    const auto color = state.GetToMove();

    if (num_intersections != (int)prob.size()) {
        return;
    }

    auto logits_q = std::vector<float>(num_intersections+1, -1e6f);
    float value_pi = 0.f;
    int max_visits = 0;

    // Compute completed Q value.
    for (auto & child : children_) {
        const auto node = child.Get();
        const bool is_pointer = node != nullptr;
        if (is_pointer && !node->IsActive()) {
            continue;
        }
        max_visits = std::max(max_visits, node->GetVisits());
        value_pi += child.GetPolicy() *
                        node->GetWL(color, false);
    }

    for (auto & child : children_) {
        const auto node = child.Get();
        const bool is_pointer = node != nullptr;
        const auto vtx = child.GetVertex();

        int idx = num_intersections; // pass move
        if (vtx != kPass) {
            idx = state.GetIndex(
                      state.GetX(vtx), state.GetY(vtx));
        }
        if (is_pointer && !node->IsActive()) {
            continue;
        }
        const float logits = std::log(prob[idx] + 1e-8f);
        const int visits = node->GetVisits();
        float completed_q;
        if (visits == 0) {
            completed_q = value_pi;
        } else {
            completed_q = node->GetWL(color, false);
        }
        logits_q[idx] = logits + (50 * max_visits) * 0.1f * completed_q;
    }
    prob = Network::Softmax(logits_q, 1.f);
}
