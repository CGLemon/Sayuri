#include "mcts/node.h"
#include "mcts/lcb.h"
#include "mcts/rollout.h"
#include "utils/atomic.h"
#include "utils/random.h"
#include "utils/format.h"
#include "utils/logits.h"
#include "utils/kldivergence.h"
#include "game/symmetry.h"
#include "pattern/gammas_dict.h"

#include <cassert>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <stack>
#include <numeric>

Node::Node(Parameters *param, std::int16_t vertex, float policy) {
    param_ = param;
    vertex_ = vertex;
    policy_ = policy;
}

Node::~Node() {
    ReleaseAllChildren();
}

bool Node::PrepareRootNode(Network &network,
                           GameState &state,
                           NodeEvals &node_evals,
                           AnalysisConfig &config) {
    const auto is_root = true;
    const auto success = ExpandChildren(network, state, node_evals, config, is_root);
    assert(HasChildren());

    InflateAllChildren();
    if (!success) {
        // Reusing node settings may not reflect what we want, eg. policy temperature,
        // so we recompute the policy and update it accordingly.
        RecomputePolicy(network, state, node_evals, is_root);
    }
    if (param_->dirichlet_noise) {
        // Generate the dirichlet noise and gather it.
        const auto legal_move = children_.size();
        const auto factor = param_->dirichlet_factor;
        const auto init = param_->dirichlet_init;
        const auto alpha = init * factor / static_cast<float>(legal_move);

        ApplyDirichletNoise(alpha);
    }

    // Remove all superkos at the root. In the most case,
    // it will help simplify the state.
    KillRootSuperkos(state);

    // Compute the score bonus for children.
    UpdateScoreBonus(state, node_evals);

    return success;
}

void Node::UpdateScoreBonus(GameState &state, NodeEvals &node_evals) {
    if (!param_->first_pass_bonus) {
        return;
    }

    assert(HasChildren());

    // reset
    black_sb_ = 0.0f;

    InflateAllChildren();
    for (auto &child : children_) {
        const auto node = child.GetPointer();
        node->ComputeScoreBonus(state, node_evals);
    }
}

void Node::RecomputePolicy(Network &network,
                           GameState &state,
                           NodeEvals &node_evals,
                           const bool is_root) {
    WaitExpanded();
    if (!HasChildren()) {
        return;
    }

    const auto raw_netlist = GetNetOutput(network, state, is_root);

    // Fill the evals buffer even if the node is complete. We may
    // use it for compute score bonus.
    FillNodeEvalsFromNet(state, raw_netlist, node_evals, state.GetToMove());

    auto buffer = std::vector<float>{};
    for (auto &child : children_) {
         const auto vtx = child.GetVertex();
         if (vtx == kPass) {
             buffer.emplace_back(raw_netlist.pass_probability);
         } else {
             buffer.emplace_back(raw_netlist.probabilities[state.VertexToIndex(vtx)]);
         }
    }

    // rescaling policy
    const auto legal_accumulate =
        std::accumulate(std::begin(buffer), std::end(buffer), 0.0f);
    if (legal_accumulate < 1e-8f) {
        for (auto &p: buffer) {
            p /= legal_accumulate;
        }
    } else {
        for (auto &p: buffer) {
            p /= legal_accumulate;
        }
    }

    // Assume we already inflated all children. Refill the new policy.
    int idx = 0;
    for (auto &child : children_) {
        child.GetPointer()->SetPolicy(buffer[idx++]);
    }
}

Network::Result Node::GetNetOutput(Network &network,
                                   GameState &state,
                                   const bool is_root) {
    // Root node policy always normal policy. Therefore, we check whether the current
    // network is using the default normal policy. If not, certain settings will be
    // disabled afterward.
    const auto default_using_normal_policy =
        network.GetDefaultPolicyOffset() == PolicyBufferOffset::kNormal;
    const auto policy_offset = is_root ?
        PolicyBufferOffset::kNormal : PolicyBufferOffset::kDefault;

    // Policy softmax temperature. If 't' is greater than 1, policy
    // will be broader. If 't' is less than 1, policy will be sharper.
    policy_temp_ = is_root ?
        param_->root_policy_temp : param_->policy_temp;

    // The network cache only stores a single policy and does not recognize different
    // types of policies. Therefore, if the policy used differs from the default one,
    // the cache should be disabled.
    const auto query = Network::Query::Get().SetTemperature(policy_temp_).
                                                 SetCache(!default_using_normal_policy).
                                                 SetOffset(policy_offset);
    auto result = network.GetOutput(state, Network::kRandom, query);

    // TODO: We are curious whether using a weaker policy can significantly and reasonably
    //       reduce the playing strength. If feasible, I can train a neural network with multiple
    //       strength levels to replace it.
    if (param_->gammas_policy_factor > 0.f && GammasDict::Get().Valid()) {
        auto gammas_policy = state.GetGammasPolicy(
                                 state.GetToMove(), result.ownership.data());
        const float reduction = (1.0f - result.pass_probability);
        const float factor = param_->gammas_policy_factor;
        int num_intersections = state.GetNumIntersections();
        for (int idx = 0; idx < num_intersections; ++idx) {
            result.probabilities[idx] = (1.0f - factor) * result.probabilities[idx] +
                                            factor * reduction * gammas_policy[idx];
        }
    }
    return result;
}

bool Node::ExpandChildren(Network &network,
                          GameState &state,
                          NodeEvals &node_evals,
                          AnalysisConfig &config,
                          const bool is_root) {
    // The node must be the first time to expand and is not the terminate node.
    assert(state.GetPasses() < 2);

    // Try to acquire the owner.
    if (!AcquireExpanding()) {
        return false;
    }

    color_ = state.GetToMove();

    // Get network computation result.
    const auto raw_netlist = GetNetOutput(network, state, is_root);

    // Store the network reuslt.
    ApplyNetOutput(state, raw_netlist, node_evals, color_);

    // For children...
    auto nodelist = std::vector<Network::PolicyVertexPair>{};
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

    for (int idx = 0; idx < num_intersections; ++idx) {
        const auto vtx = state.IndexToVertex(idx);
        const auto policy = raw_netlist.probabilities[idx];

        // Discard moves that are illegal, forbidden by configuration,
        // or otherwise undesirable.
        int movenum = state.GetMoveNumber();
        if (!state.IsLegalMove(vtx, color_,
                [movenum, &config](int vtx, int color){
                    return !config.IsLegal(vtx, color, movenum);
                })
                    || safe_area[idx]) {
            continue;
        }

        // Prune symmetric moves to eliminate redundant computations.
        // This may slightly hurt performance due to excluding certain
        // moves.
        if (apply_symm_pruning) {
            bool hash_found = false;
            for (int symm = Symmetry::kIdentitySymmetry+1;
                     symm < Symmetry::kNumSymmetris && !hash_found; ++symm) {
                const auto symm_vtx = Symmetry::Get().TransformVertex(board_size, symm, vtx);
                const auto symm_hash = symm_base_hash[symm] ^ state.GetMoveHash(symm_vtx, color_);
                hash_found = std::end(moves_hash) !=
                                 std::find(std::begin(moves_hash),
                                               std::end(moves_hash), symm_hash);
            }

            if (!hash_found) {
                // Compute the hash of the next game state. 
                // This hash may be inaccurate for capture moves,
                // but since it's only used during the opening stage,
                // and captures are rare in that phase, the inaccuracy 
                // is acceptable.
                moves_hash.emplace_back(
                    state.GetHash() ^ state.GetMoveHash(vtx, color_));
            } else {
                // This pruned node represents a legal move, so its policy
                // value should be accumulated into the total for all legal moves.
                legal_accumulate += policy;
                continue;
            }
        }

        nodelist.emplace_back(policy, vtx);
        legal_accumulate += policy;
    }

    // In the early stages, pass is usually not an option that should be
    // considered, so we remove it directly."
    const auto left_threshold = std::max(
        0, static_cast<int>((1.0f - param_->suppress_pass_factor) * num_intersections));
    const bool should_suppress_pass =
        !nodelist.empty() && static_cast<int>(nodelist.size()) > left_threshold;
    if (!should_suppress_pass) {
        nodelist.emplace_back(raw_netlist.pass_probability, kPass);
        legal_accumulate += raw_netlist.pass_probability;
    }

    if (legal_accumulate < 1e-8f) {
        // This can occur if the policy assigns most of its probability
        // mass to illegal moves. In that case, fall back to a uniform
        // distribution over all nodes.
        for (auto &node : nodelist) {
            node.first = 1.f/nodelist.size();
        }
    } else {
        // Normalize the policy values over legal moves.
        for (auto &node : nodelist) {
            node.first /= legal_accumulate;
        }
    }

    // Extend the nodes.
    LinkNodeList(nodelist);

    // Release the lock owner.
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
    children_.shrink_to_fit();
    assert(!children_.empty());
}

void Node::ApplyNetOutput(GameState &state,
                          const Network::Result &raw_netlist,
                          NodeEvals& node_evals, const int color) {
    FillNodeEvalsFromNet(state, raw_netlist, node_evals, color);

    black_wl_ = node_evals.black_wl;
    black_fs_ = node_evals.black_final_score;
    avg_black_ownership_.fill(0.f);
}

void Node::FillNodeEvalsFromNet(GameState &state,
                                const Network::Result &raw_netlist,
                                NodeEvals& node_evals, const int color) const {
    auto black_ownership = std::array<float, kNumIntersections>{};
    auto draw = raw_netlist.wdl[1];

    // Compute the black side to move evals.
    auto wl = 0.5f;

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

    for (int idx = 0; idx < kNumIntersections; ++idx) {
        auto owner = raw_netlist.ownership[idx];
        if (color == kWhite) {
            owner = 0.f - owner;
        }
        black_ownership[idx] = owner;
    }

    if (param_->use_rollout) {
        // Use random rollout instead of net's ownership
        // outputs.
        float mc_black_score;
        GetBlackRolloutResult(
            state, black_ownership.data(), mc_black_score);
    }

    // Store the network evals.
    node_evals.black_wl = wl;
    node_evals.draw = draw;
    node_evals.black_final_score = final_score;

    for (int idx = 0; idx < kNumIntersections; ++idx) {
        node_evals.black_ownership[idx] = black_ownership[idx];
    }
}

bool Node::SetTerminal() {
    if (!AcquireExpanding()) {
        return false;
    }

    color_ = kInvalid; // no children

    ExpandDone();
    return true;
}

Node *Node::DescentSelectChild(const int color, const bool is_root) {
    // At the root, attempt to select a child using the Gumbel-Top-k
    // trick. Fall back to regular PUCT if no Gumbel move is found.
    if (is_root && param_->gumbel) {
        auto node = GumbelSelectChild(color, false, true);
        if (node) {
            return node;
        }
    }
    return PuctSelectChild(color, is_root);
}

Node *Node::ProbSelectChild(bool allow_pass) {
    WaitExpanded();
    assert(HasChildren());

    Edge* best_node = nullptr;
    float best_prob = std::numeric_limits<float>::lowest();

    for (auto &child : children_) {
        const auto node = child.GetPointer();
        const bool is_pointer = node != nullptr;

        // Skip nodes that are unvisited, pruned, or inactive. 
        if (is_pointer && !node->IsActive()) {
            continue;
        }

        auto prob = child.GetPolicy();

        // Penalize nodes currently being expanded to avoid selection.
        if (is_pointer && node->IsExpanding()) {
            prob = prob - 1.0f;
        }

        // Strongly discourage selecting the pass move if not allowed.
        if (!allow_pass && child.GetVertex() == kPass) {
            prob = prob - 1e6f;
        }

        // Track the child with the highest adjusted policy probability.
        if (prob > best_prob) {
            best_prob = prob;
            best_node = &child;
        }
    }

    Inflate(*best_node);
    return best_node->GetPointer();
}

float Node::GetFpu(const int color,
                   const float total_visited_policy,
                   const bool is_root) const {
    // Computes the First Play Urgency (FPU) value for unvisited moves.
    // FPU gives a starting evaluation for child nodes before they are
    // explored, based on a combination of neural network predictions
    // and search statistics.
    const auto visits = GetVisits();
    const auto fpu_reduction_max = is_root ? param_->root_fpu_reduction : param_->fpu_reduction;
    const auto fpu_reduction = fpu_reduction_max * std::sqrt(total_visited_policy);

    if (visits <= 0) {
        return GetNetWL(color) - fpu_reduction;
    }
    const auto avg_factor = std::pow(total_visited_policy, 2.0f);
    const auto fpu_value = (1.0f - avg_factor) * GetNetWL(color) + avg_factor * GetWL(color, false);
    return fpu_value - fpu_reduction;
}

float Node::GetDynamicCpuctFactor(Node *node, const int visits, const int children_visits) {
    // Imported form http://www.yss-aya.com/bbs/patio.cgi?read=33&ukey=0

    bool cpuct_dynamic = param_->cpuct_dynamic;
    if (!cpuct_dynamic ||
            node == nullptr ||
            visits <= 1) {
        return 1.0f;
    }

    double cpuct_dynamic_k_factor = param_->cpuct_dynamic_k_factor;
    double cpuct_dynamic_k_base = param_->cpuct_dynamic_k_base;

    double variance = node->GetWLVariance(1.0f, visits);
    double stddev = std::sqrt(variance);
    double k = cpuct_dynamic_k_factor * (stddev / visits);

    k = std::max(0.5, k);
    k = std::min(1.4, k);

    double alpha = 1.0 / (1.0 + std::sqrt(children_visits/cpuct_dynamic_k_base));
    k = alpha*k + (1.0-alpha) * 1.0;
    return k;
}

float Node::GetCpuct(const int children_visits) const {
    const auto cpuct_init = param_->cpuct_init;
    const auto cpuct_base_factor = param_->cpuct_base_factor;
    const auto cpuct_base = param_->cpuct_base;

    const auto cpuct = cpuct_init + cpuct_base_factor *
                           std::log((float(children_visits) + cpuct_base + 1) / cpuct_base);
    return cpuct;
}

int Node::GetForcedVisits(const float policy,
                          const int children_visits,
                          const bool is_root) const {
    const auto forced_playouts_k = is_root ? param_->forced_playouts_k : 0.f;

    // Forced Playouts: Encourage exploration of low visit, high priority
    // children. We think 20% is big enough and high priority child is easy
    // to be explored with PUCT. We don't need to add any bonus for these
    // kind of children.
    const float forced_n_factor =
        std::max(1e-4f, forced_playouts_k *
        std::min(0.2f, policy) *
        static_cast<float>(children_visits));
    const int forced_n = std::sqrt(forced_n_factor);
    return forced_n;
}

float Node::GetSearchPolicy(Node::Edge& child,
                            const bool is_root) {
    const auto noise = is_root ?
                           param_->dirichlet_noise : false;
    auto policy = child.GetPolicy();
    if (noise) {
        const auto vertex = child.GetVertex();
        const auto epsilon = param_->dirichlet_epsilon;
        const auto eta_a = param_->dirichlet_buffer[vertex];
        policy = policy * (1 - epsilon) + epsilon * eta_a;
    }
    return policy;
}

Node *Node::PuctSelectChild(const int color, const bool is_root) {
    WaitExpanded();
    assert(HasChildren());
    // assert(color == color_);

    // Gather all parent's visits.
    int children_visits = 0;
    float total_visited_policy = 0.0f;
    for (auto &child : children_) {
        const auto node = child.GetPointer();
        const bool is_pointer = node != nullptr;

        if (is_pointer && node->IsValid()) {
            // The node status is pruned or active.
            const auto visits = node->GetVisits();
            children_visits += visits;
            if (visits > 0) {
                total_visited_policy += child.GetPolicy();
            }
        }
    }

    const float raw_cpuct     = GetCpuct(children_visits);
    const float numerator     = std::sqrt(float(children_visits));
    const float fpu_value     = GetFpu(color, total_visited_policy, is_root);
    const float parent_score  = GetFinalScore(color);

    Edge* best_node = nullptr;
    float best_value = std::numeric_limits<float>::lowest();

    for (auto &child : children_) {
        const auto node = child.GetPointer();
        const bool is_pointer = node != nullptr;

        // Skip nodes that are unvisited, pruned, or inactive.
        if (is_pointer && !node->IsActive()) {
            continue;
        }

        float q_value = fpu_value;
        const float psa = GetSearchPolicy(child, is_root);

        float cpuct = raw_cpuct;
        float denom = 1.0f;

        if (is_pointer) {
            const auto visits = node->GetVisits();

            if (node->IsExpanding()) {
                // Assign a low Q value to nodes currently being expanded to avoid
                // selection by other threads.
                q_value = std::min(fpu_value, -1.0f);
            } else if (visits > 0) {
                // Combine win/loss evaluation and score lead to optimize path selection.
                q_value = node->GetWL(color, true) +
                              node->GetScoreEval(color, parent_score);
                // Apply forced playouts.
                const int forced_n = GetForcedVisits(psa, children_visits, is_root);
                if (forced_n - visits > 0) {
                    q_value += (forced_n - visits) * 1e6;
                }
            }
            cpuct *= GetDynamicCpuctFactor(node, visits, children_visits);
            denom += visits;
        }

        // PUCT formula
        const float puct = cpuct * psa * (numerator / denom);
        const float value = q_value + puct;
        assert(value > std::numeric_limits<float>::lowest());

        if (value > best_value) {
            best_value = value;
            best_node = &child;
        }
    }

    Inflate(*best_node);
    return best_node->GetPointer();
}

int Node::GetRandomMoveProportionally(float temp,
                                      float min_ratio,
                                      int min_visits) {
    // Selects a move at random, with probability proportional to its visit count.
    // Moves with more visits are more likely to be chosen, and the "temp" parameter
    // controls how strongly the selection favors high-visit moves.
    //
    // Very low-visit moves are ignored based on:
    //   - min_ratio: relative threshold compared to the most-visited move
    //   - min_visits: absolute threshold on visits
    auto selected_vertex = kNullVertex;
    auto norm_factor= double{0};
    auto accum = double{0};
    auto accum_vector = std::vector<std::pair<decltype(accum), int>>{};
    auto max_n = int{0};

    for (const auto &child : children_) {
        max_n = std::max(max_n, child.GetVisits());
    }
    min_visits = std::max(
        static_cast<int>(std::round(max_n * min_ratio)), min_visits);

    for (const auto &child : children_) {
        auto node = child.GetPointer();
        const auto visits = node->GetVisits();
        const auto vertex = node->GetVertex();
        if (visits > min_visits) {
            if (norm_factor == 0.0) {
                norm_factor = visits;
            }
            double val = visits / norm_factor;
            accum += std::pow(val, (1.0 / temp));
            accum_vector.emplace_back(
                std::pair<decltype(accum), int>(accum, vertex));
        }
    }

    // If no moves remain after pruning, choose the best move directly.
    if (accum_vector.empty()) {
        return GetBestMove(true);
    }

    // Draw a random number and select the corresponding move.
    auto distribution =
        std::uniform_real_distribution<decltype(accum)>{0.0, accum};
    auto pick = distribution(Random<>::Get());
    auto size = accum_vector.size();

    for (auto idx = size_t{0}; idx < size; ++idx) {
        if (pick < accum_vector[idx].first) {
            selected_vertex = accum_vector[idx].second;
            break;
        }
    }

    return selected_vertex;
}

int Node::GetRandomMoveWithLogitsQ(GameState &state, float temp) {
    const auto num_intersections = state.GetNumIntersections();
    auto prob = std::vector<float>(num_intersections+1, 0.f);
    auto vertices_table = std::vector<int>(num_intersections+1, kNullVertex);
    int accum_visists = 0;

    for (const auto &child : children_) {
        auto node = child.GetPointer();
        const auto visits = node->GetVisits();
        const auto vtx = child.GetVertex();
        int idx = state.VertexToIndexIncludingPass(vtx);

        if (visits != 0) {
            accum_visists += visits;
            prob[idx] = visits;
            vertices_table[idx] = vtx;
        }
    }

    // If there is no visits, choose the best move directly.
    if (accum_visists == 0) {
        return GetBestMove(true);
    }
    for (float &p : prob) {
        p /= (float)accum_visists;
    }
    MixLogitsCompletedQ(state, prob);

    auto selected_vertex = kNullVertex;
    auto accum = double{0};
    auto accum_vector = std::vector<std::pair<decltype(accum), int>>{};

    for (int idx = 0; idx < num_intersections+1; ++idx) {
        // Prune the unvisited moves.
        int vtx = vertices_table[idx];
        if (vtx != kNullVertex) {
            accum += std::pow((decltype(accum))prob[idx], (1.0 / temp));
            accum_vector.emplace_back(
                std::pair<decltype(accum), int>(accum, vtx));
        }
    }

    // What happened? Is it possible?
    if (accum_vector.empty()) {
        return GetRandomMoveProportionally(temp, 0.f, 0);
    }

    // Draw a random number and select the corresponding move
    auto distribution =
        std::uniform_real_distribution<decltype(accum)>{0.0, accum};
    auto pick = distribution(Random<>::Get());
    auto size = accum_vector.size();

    for (auto idx = size_t{0}; idx < size; ++idx) {
        if (pick < accum_vector[idx].first) {
            selected_vertex = accum_vector[idx].second;
            break;
        }
    }

    return selected_vertex;
}

void Node::Update(const NodeEvals *evals) {
    auto WelfordDelta = [](double eval,
                           double old_acc_eval,
                           int old_visits) {
        // Welford's online algorithm for calculating variance.
        const double old_delta = old_visits > 0 ? eval - old_acc_eval / old_visits : 0.0f;
        const double new_delta = eval - (old_acc_eval + eval) / (old_visits+1);
        const double delta = old_delta * new_delta;
        return delta;
    };

    // type casting
    const double eval = evals->black_wl;
    const double draw = evals->draw;
    const double score = evals->black_final_score;

    const double old_acc_eval = accumulated_black_wl_.load(std::memory_order_relaxed);
    const double old_acc_score = accumulated_black_fs_.load(std::memory_order_relaxed);

    const int old_visits = visits_.load(std::memory_order_relaxed);

    // TODO: According to Kata Go, It is not necessary to use
    //       Welford's online algorithm. The accuracy of simplify
    //       algorithm is enough.
    const double eval_delta = WelfordDelta(eval, old_acc_eval, old_visits);
    const double score_delta = WelfordDelta(score, old_acc_score, old_visits);

    visits_.fetch_add(1, std::memory_order_relaxed);
    AtomicFetchAdd(accumulated_black_wl_, eval);
    AtomicFetchAdd(accumulated_draw_    , draw);
    AtomicFetchAdd(accumulated_black_fs_, score);
    AtomicFetchAdd(squared_eval_diff_   , eval_delta);
    AtomicFetchAdd(squared_score_diff_  , score_delta);

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
    black_fs_ = evals->black_final_score;
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

float Node::GetScoreBonus(const int color) const {
    if (color == kBlack) {
        return black_sb_;
    }
    return 0.f - black_sb_;
}

float Node::GetScoreEval(const int color, float parent_score) const {
    const auto factor = param_->score_utility_factor;
    const auto div = param_->score_utility_div;
    const auto score = GetFinalScore(color) + GetScoreBonus(color);
    return factor * std::tanh((score - parent_score)/div);
}

float Node::GetScoreVariance(const float default_var, const int visits) const {
    return visits > 1 ?
               squared_score_diff_.load(std::memory_order_relaxed) / (visits - 1) :
               default_var;
}

float Node::GetScoreStddev() const {
    const auto visits = GetVisits();
    const auto variance = GetScoreVariance(1.0f, visits);
    return std::sqrt(variance);
}

float Node::GetWLVariance(const float default_var, const int visits) const {
    return visits > 1 ?
               squared_eval_diff_.load(std::memory_order_relaxed) / (visits - 1) :
               default_var;
}

float Node::GetWLStddev() const {
    const auto visits = GetVisits();
    const auto variance = GetWLVariance(1.0f, visits);
    return std::sqrt(variance);
}

float Node::GetLcb(const int color) const {
    // The Lower confidence bound of winrate.
    // See the LCB issue here: https://github.com/leela-zero/leela-zero/pull/2290

    const auto visits = GetVisits();
    if (visits <= 1) {
        // We can not get the variance at the first visit. Return
        // the large negative value.
        return GetPolicy() - 1e6f;
    }

    const auto mean = GetWL(color, false);
    const auto variance = GetWLVariance(1.0f, visits);
    const auto stddev = std::sqrt(variance);
    const auto z = LcbEntries::Get().CachedTQuantile(visits - 1);

    // The variance divide the visits in order to empirically
    // make the bound decrease slower.
    return mean - z * (stddev/visits);
}

void Node::ComputeScoreBonus(GameState &state, NodeEvals &parent_node_evals) {
    if (!param_->first_pass_bonus ||
            state.GetKoMove() != kNullVertex) {
        black_sb_ = 0.0f;
        return;
    }

    constexpr float kRawOwnershipThreshold = 0.8f; // ~90%
    constexpr float kTail = 1.0f - kRawOwnershipThreshold;
    constexpr float kEndBonus = 0.5f;
    const auto vtx = GetVertex();
    const auto color = state.GetToMove();
    float black_bonus = 0.0f;

    if (state.GetScoringRule() == kArea) {
        // Under the scoring area, simply encourage the passing, so the player try to
        // pass first.
        if (vtx == kPass) {
            black_bonus += kEndBonus;
        } else if (state.IsSeki(vtx)) {
            black_bonus += kEndBonus;
        } else {
            const auto idx = state.VertexToIndex(vtx);
            const auto black_owner = parent_node_evals.black_ownership[idx];

            if ((black_owner > kRawOwnershipThreshold && color == kBlack) ||
                    (black_owner < -kRawOwnershipThreshold && color == kWhite)) {
                if (state.IsNeighborColor(vtx, !color)) {
                    black_bonus += kEndBonus;
                }
            }
        }
        if (color == kWhite) {
            black_bonus = 0.0f - black_bonus;
        }
    } else if (state.GetScoringRule() == kTerritory) {
        // Under the scoring, slightly encourage dame-filling by discouraging passing, so
        // that the player will try to do everything non-point-losing first, like filling
        // dame. But cosmetically, it's also not great if we just encourage useless threat
        // moves in the opponent's territory to prolong the game. So also discourage those
        // moves to.
        if (vtx == kPass) {
            black_bonus -= (2.f/3.f) * kEndBonus;
        } else {
            const auto idx = state.VertexToIndex(vtx);
            const auto black_owner = parent_node_evals.black_ownership[idx];
            float owner_penalty_factor = 0.0f;

            if (black_owner > kRawOwnershipThreshold ||
                    black_owner < -kRawOwnershipThreshold) {
                owner_penalty_factor = (
                    std::abs(black_owner) - kRawOwnershipThreshold) / kTail;
            }
            black_bonus -= owner_penalty_factor * kEndBonus;
        }
        if (color == kWhite) {
            black_bonus = 0.0f - black_bonus;
        }
    }
    black_sb_ = black_bonus;
}

std::string Node::GetPathVerboseString(GameState &state, int color,
                                       std::vector<int> &moves) {
    auto curr_node = Get();
    auto out = std::ostringstream{};
    int depth = 0;

    while (depth < (int)moves.size() && curr_node) {
        curr_node = curr_node->GetChild(moves[depth++]);
        if (curr_node) {
            const auto vertex = curr_node->GetVertex();
            const auto winrate = curr_node->GetWL(color, false);
            const auto score = curr_node->GetFinalScore(color);
            const auto policy = curr_node->GetPolicy();
            const auto raw_winrate = curr_node->GetNetWL(color);
            const auto raw_score = curr_node->GetNetScore(color);
            out << Format("%s -> avg-WL: %.2f(\%), avg-S: %.2f, P: %.2f(\%), WL: %.2f(\%), S: %.2f\n",
                       state.VertexToText(vertex).c_str(),
                       100 * winrate, score,
                       100 * policy, 100 * raw_winrate, raw_score);
        }
        color = !color;
    }

    if (curr_node) {
        if (curr_node->GetVisits() < 1) {
            out << "edge";
        } else if (curr_node->GetVisits() == 1) {
            out << "node";
        } else {
            auto verbose = curr_node->ToVerboseString(state, color);
            auto vsize = verbose.size();
            verbose.resize(vsize-1);
            out << "To move color is "
                    << (color == kBlack ? "BLACK" : "WHITE")
                    << std::endl;
            out << verbose;
        }
    } else {
        out << "not a node/edge";
    }
    return out.str();
}

std::string Node::ToVerboseString(GameState &state, const int color) {
    auto out = std::ostringstream{};
    const auto children_visits = GetChildrenVisits();
    const auto lcblist = GetSortedLcbUtilityList(color, children_visits);

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

        const auto visit_ratio = static_cast<float>(visits) / children_visits;
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

    // Here are some errors to compute memory used.
    const auto mem_used = static_cast<double>(
        nodes * node_mem + edges * edge_mem) / (1024.f * 1024.f);

    const auto space2 = 14;
    out << " * Tree Status:" << std::endl
            << std::fixed << std::setprecision(4)
            << std::setw(space2) << "nodes:"   << ' ' << nodes    << std::endl
            << std::setw(space2) << "edges:"   << ' ' << edges    << std::endl
            << std::setw(space2) << "memory:"  << ' ' << mem_used << ' ' << "(MiB)" << std::endl;

    return out.str();
}

std::string Node::ToAnalysisString(GameState &state,
                                   const int color,
                                   AnalysisConfig &config) {
    const auto OwnershipToString = [](GameState &state,
                                      const int color,
                                      std::string name,
                                      Node *node) -> std::string {
        auto out = std::ostringstream{};
        const auto num_intersections = state.GetNumIntersections();
        auto ownership = node->GetOwnership(color);

        out << name << ' ';
        for (int idx = 0; idx < num_intersections; ++idx) {
            out << Format("%.6f ", ownership[state.IndexToRowMajorIndex(idx)]);
        }
        return out.str();
    };

    // Gather the analysis string. You can see the detail here
    // https://github.com/SabakiHQ/Sabaki/blob/master/docs/guides/engine-analysis-integration.md

    auto out = std::ostringstream{};
    const auto lcblist = GetSortedLcbUtilityList(color);
    auto root = Get();
    if (config.output_format == AnalysisConfig::kSayuri) {
        out << Format("info move null visits %d winrate %.6f drawrate %.6f scorelead %.6f ",
                         root->GetVisits(),
                         root->GetWL(color, false),
                         root->GetDraw(),
                         root->GetFinalScore(color)
                     );
        if (config.ownership) {
            out << OwnershipToString(state, color, "ownership", root);
        }
    }

    int order = 0;
    for (auto &lcb_pair : lcblist) {
        if (order+1 > config.max_moves) {
            break;
        }

        const auto lcb = lcb_pair.first > 0.0f ? lcb_pair.first : 0.0f;
        const auto vertex = lcb_pair.second;

        auto child = GetChild(vertex);
        const auto visits = child->GetVisits();
        const auto winrate = child->GetWL(color, false);
        const auto final_score = child->GetFinalScore(color);
        const auto prior = child->GetPolicy();
        const auto pv_string = state.VertexToText(vertex) + ' ' + child->GetPvString(state);

        if (config.output_format == AnalysisConfig::kSayuri) {
            const auto drawrate = child->GetDraw();
            out << Format("info move %s visits %d winrate %.6f drawrate %.6f scorelead %.6f prior %.6f lcb %.6f order %d pv %s",
                             state.VertexToText(vertex).c_str(),
                             visits,
                             winrate,
                             drawrate,
                             final_score,
                             prior,
                             std::min(1.0f, lcb),
                             order,
                             pv_string.c_str()
                         );
        } else if (config.output_format == AnalysisConfig::kKata) {
            out << Format("info move %s visits %d winrate %.6f scoreLead %.6f prior %.6f lcb %.6f order %d pv %s",
                             state.VertexToText(vertex).c_str(),
                             visits,
                             winrate,
                             final_score,
                             prior,
                             std::min(1.0f, lcb),
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
        if (config.ownership && config.output_format == AnalysisConfig::kSayuri) {
            out << OwnershipToString(state, color, "ownership", child);
        }
        if (config.moves_ownership && config.output_format != AnalysisConfig::kSayuri) {
            out << OwnershipToString(state, color, "movesOwnership", child);
        }
        order += 1;
    }

    if (config.ownership && config.output_format != AnalysisConfig::kSayuri) {
        out << OwnershipToString(state, color, "ownership", root);
    }

    // There may be no output information so only add new line character for
    // non-empty string.
    out.seekp(0, std::ios::end);
    if (out.tellp() > 0) {
        out << std::endl;
    }
    out.seekp(0, std::ios::end);


    return out.str();
}

std::string Node::GetPvString(GameState &state) {
    auto pvlist = std::vector<int>{};
    auto *next = this;
    while (next->HasChildren()) {
        const auto vtx = next->GetBestMove(true);
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
            return child.GetPointer();
        }
    }
    return nullptr;
}

Node *Node::PopChild(const int vertex) {
    auto node = GetChild(vertex);
    if (node) {
        auto ite = std::remove_if(std::begin(children_), std::end(children_),
                                  [node](Edge &ele) {
                                      return ele.GetPointer() == node;
                                  });
        children_.erase(ite, std::end(children_));
    }
    return node;
}

int Node::GetChildrenVisits() const {
    // Returns the total number of visits across all active child
    // nodes. Only counts visits from children that are marked active.
    int validvisits = 0;
    for (const auto & child : children_) {
        const auto node = child.GetPointer();
        const bool is_pointer = node != nullptr;

        if (is_pointer && node->IsActive()) {
            validvisits += node->GetVisits();
        }
    }
    return validvisits;
}

std::vector<std::pair<float, int>> Node::GetSortedLcbUtilityList(const int color) {
    return GetSortedLcbUtilityList(color, GetChildrenVisits());
}

std::vector<std::pair<float, int>> Node::GetSortedLcbUtilityList(const int color,
                                                                 const int children_visits) {
    WaitExpanded();
    assert(HasChildren());

    // Clamp lcb_reduction parameter to the [0, 1] range.
    const auto lcb_reduction = std::min(
        std::max(0.f, param_->lcb_reduction), 1.f);
    const auto parent_score = GetFinalScore(color);
    auto lcblist = std::vector<std::pair<float, int>>{};

    for (const auto & child : children_) {
        const auto node = child.GetPointer();
        const bool is_pointer = node != nullptr;

        // Skip nodes that are unvisited, pruned, or inactive.
        if (!is_pointer || !node->IsActive()) {
            continue;
        }

        const auto visits = node->GetVisits();
        if (visits > 0) {
            // Compute mixed LCB by combining the node's LCB with its score evaluation.
            // This ensures the bias of LCB with scoring matches that of Q evaluation
            // in the PUCT phase.
            const auto mixed_lcb = node->GetLcb(color) +
                                       node->GetScoreEval(color, parent_score);

            // Adjust the mixed LCB to penalize moves with fewer visits.
            // For example, a node with 100 visits and 90% LCB might be less stable
            // than a node with 1,000,000 visits and 89% LCB. This adjustment
            // favors more stable nodes with higher visit counts.
            const auto rlcb = mixed_lcb * (1.0f - lcb_reduction) +
                                  lcb_reduction * ((float)visits/children_visits);
            lcblist.emplace_back(rlcb, node->GetVertex());
        }
    }

    std::stable_sort(std::rbegin(lcblist), std::rend(lcblist));
    return lcblist;
}

int Node::GetBestMove(bool allow_pass) {
    WaitExpanded();
    assert(HasChildren());

    auto lcblist = GetSortedLcbUtilityList(color_);
    float best_value = std::numeric_limits<float>::lowest();
    int best_move = kNullVertex;

    for (auto &lcb_pair : lcblist) {
        const auto lcb = lcb_pair.first;
        const auto vtx = lcb_pair.second;
        if (lcb > best_value) {
            if (!allow_pass && vtx == kPass) {
                continue;
            }
            best_value = lcb;
            best_move = vtx;
        }
    }

    // If no valid move was found, select a child based on raw policy
    // probabilities instead.
    if (best_move == kNullVertex) {
        best_move = ProbSelectChild(allow_pass)->GetVertex();
    }

    assert(best_move != kNullVertex);
    return best_move;
}

const std::vector<Node::Edge> &Node::GetChildren() const {
    return children_;
}

int Node::GetVirtualLoss() const {
    return param_->virtual_loss_count *
               running_threads_.load(std::memory_order_relaxed);
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

float Node::GetNetScore(const int color) const {
    if (color == kBlack) {
        return black_fs_;
    }
    return 0.0f - black_fs_;
}

float Node::GetWL(const int color, const bool use_virtual_loss) const {
    auto virtual_loss = 0;

    if (use_virtual_loss) {
        // When we visit a node, add this amount of virtual losses
        // to it to encourage other CPUs to explore other parts of the
        // search tree.
        virtual_loss = GetVirtualLoss();
    }

    auto visits = GetVisits() + virtual_loss;
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
    if (child.Inflate(param_)) {
        // do nothing...
    }
}

void Node::Release(Edge& child) {
    if (child.Release()) {
        // do nothing...
    }
}

bool Node::HasChildren() const {
    return IsExpanded() && color_ != kInvalid;
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

void Node::Invalidate() {
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
            node = child.GetPointer();
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

float Node::GetGumbelEval(int color, float parent_score) const {
    // Return the non-transformed complete Q value. In the original
    // Gumbel AlphaZero paper, this corresponds to the Q value. Here,
    // we combine the Q value with the score lead to refine move probability
    // optimization, encouraging the selection of the best move.
    const auto mixed_q = GetWL(color, false) +
                             GetScoreEval(color, parent_score);
    return mixed_q;
}

float Node::TransformCompletedQ(const float completed_q,
                                const int max_visits) const {
    // The transformation progressively increases the scale for
    // Q value and reduces the effect of the prior policy.
    const auto c_visit = param_->gumbel_c_visit;
    const auto c_scale = param_->gumbel_c_scale;
    const auto threshold = param_->gumbel_playouts_threshold;

    return (c_visit + std::min(threshold, max_visits)) *
               c_scale * completed_q;
}

std::vector<float> Node::GetProbLogitsCompletedQ(GameState &state) {
    const auto num_intersections = state.GetNumIntersections();
    auto prob = std::vector<float>(num_intersections+1, 0.f);
    float acc = 0.f;

    for (auto & child : children_) {
        const auto vtx = child.GetVertex();
        const auto idx = state.VertexToIndexIncludingPass(vtx);
        acc += child.GetPolicy();
        prob[idx] = child.GetPolicy();
    }

    for (auto &v : prob) {
        v /= acc;
    }

    MixLogitsCompletedQ(state, prob);
    return prob;
}

void Node::MixLogitsCompletedQ(GameState &state,
                               std::vector<float> &prob) {
    const auto num_intersections = state.GetNumIntersections();
    const auto color = state.GetToMove();

    // Ensure that 'prob' has a length equal to the number of board intersections
    // plus pass move.
    if (num_intersections + 1 != static_cast<int>(prob.size())) {
        return;
    }

    const auto parent_score = GetFinalScore(color);
    auto logits_q = GetZeroLogits<float>(num_intersections+1);

    int max_visits = 0;
    int children_visits = 0;;
    float weighted_q = 0.f;
    float weighted_pi = 0.f;

    // Compute weigted Q and sum of weigted policy.
    for (auto & child : children_) {
        const auto node = child.GetPointer();
        const bool is_pointer = node != nullptr;

        int visits = 0;
        if (is_pointer && node->IsActive()) {
            visits = node->GetVisits();
        }
        children_visits += visits;
        max_visits = std::max(max_visits, visits);

        if (visits > 0) {
            weighted_q += child.GetPolicy() *
                              node->GetGumbelEval(color, parent_score);
            weighted_pi += child.GetPolicy();
        }
    }

    // Compute the all children's completed Q.
    auto completed_q_list = std::vector<float>{};

    // Mix the raw value and approximate Q value. It may help
    // to improve the performance when the children_visits is very
    // low (<= 4).
    const float raw_value = GetNetWL(color);
    const float approximate_q = (raw_value + (children_visits/weighted_pi) *
                                    weighted_q) / (1 + children_visits);
    for (auto & child : children_) {
        const auto node = child.GetPointer();
        const bool is_pointer = node != nullptr;

        int visits = 0;
        if (is_pointer && node->IsActive()) {
            visits = node->GetVisits();
        }

        float completed_q;
        if (visits == 0) {
            // The unvisited node has no Q value. Give it a
            // virtual approximate Q value.
            completed_q = approximate_q;
        } else {
            completed_q = node->GetGumbelEval(color, parent_score);
        }
        completed_q_list.emplace_back(completed_q);
    }

    // Apply the completed Q with policy.
    int completed_q_idx = 0;
    for (auto & child : children_) {
        const auto vtx = child.GetVertex();
        const auto idx = state.VertexToIndexIncludingPass(vtx);

        const float logits = SafeLog(prob[idx]);
        const float completed_q = completed_q_list[completed_q_idx++];

        // Transform the Completed Q value because it makes
        // policy logit and Q value balance.
        logits_q[idx] = logits + TransformCompletedQ(
                                     completed_q, max_visits);
    }
    prob = Softmax(logits_q, 1.f);

    // Prune moves with negligible probability.
    double psize = prob.size();
    double noise_threshold = 1./(100. + psize);
    double o = 0.;
    for (auto &v : prob) {
        if (v < noise_threshold) {
            v = 0.;
        } else {
            o += v;
        }
    }

    for (auto &v : prob) {
        v /= o;
    }
}

bool Node::ShouldApplyGumbel() const {
    return param_->gumbel &&
               param_->gumbel_playouts_threshold > GetChildrenVisits();
}

bool Node::ProcessGumbelLogits(std::vector<float> &gumbel_logits,
                               const int color,
                               const bool only_max_visits) {
    // This is a variant of the Sequential Halving algorithm.
    // The input N (number of playouts) is always calculated as:
    //   (promotion visits) * (log2(considered moves) + 1) * (considered moves)
    // for each epoch.
    // The variant behaves the same as the standard Sequential Halving algorithm
    // if the total number of playouts is lower than this value.
    //
    // * Round 0.
    //  accumulation: { 0, 0, 0, 0 }
    //
    // * Round 1.
    //  distribution: { 1, 1, 1, 1 }
    //  accumulation: { 1, 1, 1, 1 }
    //
    // * Round 2.
    //  distribution: { 2, 2, 0, 0 }
    //  accumulation: { 3, 3, 1, 1 }
    //
    // * Round 3.(1st epoch is end)
    //  distribution: { 4, 0, 0, 0 }
    //  accumulation: { 7, 3, 1, 1 }
    //
    // * Round 4.
    //  distribution: { 1, 1, 1, 1 }
    //  accumulation: { 8, 4, 2, 2 }
    //
    // * Round 5.
    //  distribution: {  2, 2, 0, 0 }
    //  accumulation: { 10, 6, 2, 2 }
    //
    // * Round 6. (2nd epoch is end)
    //  distribution: {  4, 0, 0, 0 }
    //  accumulation: { 14, 6, 2, 2 }

    const int size = children_.size();
    auto table = std::vector<std::pair<int, int>>(size);
    gumbel_logits.resize(size, LOGIT_ZERO);

    for (int i = 0; i < size; ++i) {
        auto &child = children_[i];
        const auto node = child.GetPointer();
        const bool is_pointer = node != nullptr;

        if (is_pointer && node->IsValid()) {
            // The node status is pruned or active.
            const auto visits = node->GetVisits();

            if (node->IsActive()) {
                table[i].first = visits;
                table[i].second = child.GetVertex();
            }
        }
    }

    std::stable_sort(std::rbegin(table), std::rend(table));
    const int max_visists = table[0].first;

    const int considered_moves =
        std::min(param_->gumbel_considered_moves, (int)children_.size());
    int playouts_thres = param_->gumbel_playouts_threshold;
    const int prom_visits = std::max(1, param_->gumbel_prom_visits);
    const int n = std::log2(std::max(1, considered_moves)) + 1;
    const int adj_considered_moves = std::pow(2, n-1); // Be sure that it is pow of 2.

    int target_visits = 0;
    int width = adj_considered_moves;
    int level = prom_visits;

    if (only_max_visits) {
        // The case is to output the best move.
        playouts_thres = std::max(playouts_thres, 1);
        target_visits = max_visists;
        goto end_loop;
    }

    // We may reuse the sub-tree. Try to fill the old distribution  
    // so that it covers the Sequential Halving distribution. For example:
    //
    // Original distribution (sorted):
    //   { 9, 2, 0, 0 }
    //
    // Target Sequential Halving distribution:
    //   { 7, 3, 1, 1 }
    //
    // Addition playouts distribution:
    //   { 0, 1, 1, 1 }
    //
    // Result distribution:
    //   { 9, 3, 1, 1 }
    while (true) {
        for (int i = 0; i < level; ++i) {
            // Keep to minus current root visits based on Sequential Halving
            // distribution until finding first zero visits child.
            for (int j = 0; j < width; ++j) {
                if (table[j].first <= 0) {
                    auto vtx = table[j].second;
                    target_visits = GetChild(vtx)->GetVisits();
                    goto end_loop;
                }
                table[j].first -= 1;
                playouts_thres -= 1;

                // If the total playout threshold is met, terminate the process.
                // and disable the Gumbel noise.
                if (playouts_thres <= 0) {
                    goto end_loop;
                }
            }
        }
        if (width == 1) {
            // Move to the next epoch: reset width and level for the new round.
            width = adj_considered_moves;
            level = prom_visits;
        } else {
            // Narrow the candidate set by half and double the allocation per child.
            width /= 2;
            level *= 2;
        }
    }

end_loop:;
    if (playouts_thres <= 0) {
        return false;
    }

    int count = 0;
    const auto parent_score = GetFinalScore(color);
    auto gumbel_type1 = std::extreme_value_distribution<float>(0, 1);

    for (int i = 0; i < size; ++i) {
        auto &child = children_[i];
        const auto node = child.GetPointer();
        const bool is_pointer = node != nullptr;

        // Skip nodes that are unvisited, pruned, or inactive.
        if (is_pointer && !node->IsActive()) {
            continue;
        }
        if (target_visits == child.GetVisits()) {
            auto logit = gumbel_type1(Random<>::Get()) +
                             SafeLog(child.GetPolicy());
            auto completed_q = 0.f;
            if (is_pointer && target_visits > 0) {
                completed_q = TransformCompletedQ(
                    node->GetGumbelEval(color, parent_score), max_visists);
            }
            gumbel_logits[i] = logit + completed_q;
            count += 1;
        }
    }
    if (count == 0) {
        // It may occur when multi-threads search.
        return false;
    }
    return true;
}

Node *Node::GumbelSelectChild(int color, bool only_max_visits, bool allow_pass) {
    WaitExpanded();
    assert(HasChildren());

    auto gumbel_logits = std::vector<float>{};
    if (!ProcessGumbelLogits(gumbel_logits, color, only_max_visits)) {
        // Fail to find the next node.
        return nullptr;
    }

    Edge* best_node = nullptr;
    Edge* best_node_no_pass = nullptr;
    float best_value = std::numeric_limits<float>::lowest();
    const int size = children_.size();

    for (int i = 0; i < size; ++i) {
        auto &child = children_[i];
        const auto value = gumbel_logits[i];

        if (value > best_value) {
            best_value = value;
            best_node = &child;
            if (child.GetVertex() != kPass) {
                best_node_no_pass = &child;
            }
        }
    }

    if (!allow_pass && best_node_no_pass) {
        Inflate(*best_node_no_pass);
        return best_node_no_pass->GetPointer();
    }
    Inflate(*best_node);
    return best_node->GetPointer();
}

int Node::GetGumbelMove(bool allow_pass) {
    WaitExpanded();
    assert(HasChildren());

    int num_candidates = 0;
    for (auto &child : children_) {
        const auto node = child.GetPointer();
        const int visits = child.GetVisits();
        if (visits > 0 && node->IsValid()) {
            num_candidates += 1;
        }
    }

    if (!allow_pass && num_candidates == 1) {
        // It there is only one candidate move, it may be the pass
        // move. So alway enable pass move.
        allow_pass = true;
    }

    auto node = GumbelSelectChild(color_, true, allow_pass);
    if (!node) {
        return GetBestMove(allow_pass);
    }
    return node->GetVertex();
}

void Node::KillRootSuperkos(GameState &state) {
    for (const auto &child : children_) {
        const auto vtx = child.GetVertex();

        auto fork_state = state;
        fork_state.PlayMove(vtx);

        if (vtx != kPass &&
                fork_state.IsSuperko()) {
            // Kill all superko moves.
            child.GetPointer()->Invalidate();
        }
    }

    auto ite = std::remove_if(std::begin(children_), std::end(children_),
                              [](Edge &ele) {
                                  return !ele.GetPointer()->IsValid();
                              });
    children_.erase(ite, std::end(children_));
}
