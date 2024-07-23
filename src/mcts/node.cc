#include "mcts/node.h"
#include "mcts/lcb.h"
#include "mcts/rollout.h"
#include "utils/atomic.h"
#include "utils/random.h"
#include "utils/format.h"
#include "utils/logits.h"
#include "utils/kldivergence.h"
#include "utils/ai_style.h"
#include "game/symmetry.h"

#include <cassert>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <stack>
#include <numeric>

#define VIRTUAL_LOSS_COUNT (3)

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
        // The setting of root policy and children may be different,
        // like softmax temperature, so we refill the children policy.
        Recompute(network, state, is_root);
    }
    if (param_->dirichlet_noise) {
        // Generate the dirichlet noise and gather it.
        const auto legal_move = children_.size();
        const auto factor = param_->dirichlet_factor;
        const auto init = param_->dirichlet_init;
        const auto alpha = init * factor / static_cast<float>(legal_move);

        ApplyDirichletNoise(alpha);
    }

    // Adjust the strength by pruning children.
    RandomPruneRootChildren(state);

    // Remove all superkos at the root. In the most case,
    // it will help simplify the state.
    KillRootSuperkos(state);

    // Reset the bouns.
    SetScoreBouns(0.f);
    for (auto &child : children_) {
        auto node = child.Get();
        if (param_->first_pass_bonus &&
                child.GetVertex() == kPass) {
            // Half komi bouns may efficiently end the game.
            node->SetScoreBouns(0.5f);
        } else {
            node->SetScoreBouns(0.f);
        }
    }

    return success;
}

void Node::Recompute(Network &network,
                     GameState &state,
                     const bool is_root) {
    WaitExpanded();
    if (!HasChildren()) {
        return;
    }

    const float temp = is_root ?
                param_->root_policy_temp : param_->policy_temp;
    auto raw_netlist = network.GetOutput(
        state, Network::kRandom, Network::Query::SetTemperature(temp));

    const auto num_intersections = state.GetNumIntersections();
    auto legal_accumulate = 0.f;

    // Filter the illegal or pruned nodes.
    for (int idx = 0; idx < num_intersections; ++idx) {
        const auto vtx = state.IndexToVertex(idx);
        auto node = GetChild(vtx);
        if (node) {
            legal_accumulate +=
                raw_netlist.probabilities[idx];
        }
    }
    auto passnode = GetChild(kPass);
    if (passnode) {
        legal_accumulate += raw_netlist.pass_probability;
    }

    // Assign the new policy.
    for (int idx = 0; idx < num_intersections; ++idx) {
        const auto vtx = state.IndexToVertex(idx);
        auto node = GetChild(vtx);
        if (node) {
            const auto policy =
                raw_netlist.probabilities[idx]/legal_accumulate;
            node->SetPolicy(policy);
        }
    }
    if (passnode) {
        passnode->SetPolicy(
            raw_netlist.pass_probability/legal_accumulate);
    }
}

void Node::RandomPruneRootChildren(GameState &state) {
    if (param_->relative_rank < 0) {
        return;
    }
    if (state.GetBoardSize() != 19) {
        return;
    }

    auto GetCoord = [](GameState &state, int vtx) {
        auto coord = std::array<int, 2>({-1, -1});
        if (vtx != kPass
                && vtx != kResign
                && vtx != kNullVertex) {
            coord[0] = state.GetX(vtx);
            coord[1] = state.GetY(vtx);
        }
        return coord;
    };

    auto selection = SelectionVector<Node *>{};
    for (const auto &child : children_) {
        const auto node = child.Get();
        if (node->IsActive()) {
            auto coord = GetCoord(state, node->GetVertex());
            selection.emplace_back(
                child.GetPolicy(), coord, node);
        }
    }

    selection = GetRelativeRankVector(
        selection, param_->relative_rank,
        state.GetBoardSize(),
        GetCoord(state, state.GetLastMove()));

    // prune all active node first
    for (const auto &child : children_) {
        const auto node = child.Get();
        if (!node->IsActive()) {
            continue;
        }
        node->SetActive(false);
    }

    // activate the selection children
    for (auto &it : selection) {
        const auto node = std::get<2>(it);

        if (node->IsPruned()) {
            node->SetActive(true);
        }
    }
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

    // Policy softmax temperature. If 't' is greater than 1, policy
    // will be broader. If 't' is less than 1, policy will be sharper.
    const float temp = is_root ?
                    param_->root_policy_temp : param_->policy_temp;
    auto raw_netlist = network.GetOutput(
        state, Network::kRandom, Network::Query::SetTemperature(temp));

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

    // Prune the illegal moves or some bad move.
    for (int idx = 0; idx < num_intersections; ++idx) {
        const auto vtx = state.IndexToVertex(idx);
        const auto policy = raw_netlist.probabilities[idx];

        // Prune the illegal, unwise and forbidden move.
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
                hash_found = std::end(moves_hash) !=
                                 std::find(std::begin(moves_hash),
                                               std::end(moves_hash), symm_hash);
            }

            if (!hash_found) {
                // Get next game state hash. Is is not always correct
                // if move is capture. It is ok because we only need
                // move hash in the opening stage. The capture move is
                // unusual in the opening stage.
                moves_hash.emplace_back(
                    state.GetHash() ^ state.GetMoveHash(vtx, color_));
            } else {
                // The pruned node is a legal move. We need accumulate
                // the all legal moves policy.
                legal_accumulate += policy;
                continue;
            }
        }

        nodelist.emplace_back(policy, vtx);
        legal_accumulate += policy;
    }

    // The pass is always legal.
    nodelist.emplace_back(raw_netlist.pass_probability, kPass);
    legal_accumulate += raw_netlist.pass_probability;

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

void Node::ApplyNetOutput(GameState& state,
                          const Network::Result &raw_netlist,
                          NodeEvals& node_evals, const int color) {
    auto black_ownership = std::array<float, kNumIntersections>{};
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
    black_fs_ = final_score;

    for (int idx = 0; idx < kNumIntersections; ++idx) {
        auto owner = raw_netlist.ownership[idx];
        if (color == kWhite) {
            owner = 0.f - owner;
        }
        black_ownership[idx] = owner;
        avg_black_ownership_[idx] = 0.f;
    }

    if (param_->use_rollout) {
        // Use random rollout instead of net's ownership
        // outputs.
        float mc_black_score;
        GetBlackRolloutResult(
            state, black_ownership.data(), mc_black_score);
    }

    // Store the network evals.
    node_evals.black_wl = black_wl_;
    node_evals.draw = draw;
    node_evals.black_final_score = black_fs_;

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

Node *Node::ProbSelectChild(bool allow_pass) {
    WaitExpanded();
    assert(HasChildren());

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
            prob = prob - 1.0f;
        }

        // Try to forbid the pass move. Give it a crazy
        // bad value.
        if (!allow_pass && child.GetVertex() == kPass) {
            prob = prob - 1e6f;
        }

        if (prob > best_prob) {
            best_prob = prob;
            best_node = &child;
        }
    }

    Inflate(*best_node);
    return best_node->Get();
}

float Node::GetDynamicCpuctFactor(Node *node, const int visits, const int parentvisits) {
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

    double alpha = 1.0 / (1.0 + std::sqrt(parentvisits/cpuct_dynamic_k_base));
    k = alpha*k + (1.0-alpha) * 1.0;
    return k;
}

Node *Node::PuctSelectChild(const int color, const bool is_root) {
    WaitExpanded();
    assert(HasChildren());
    // assert(color == color_);

    // Apply the Gumbel-Top-k trick here. Mix it with PUCT
    // search. Use the PUCT directly if we fail to find the
    // next Gumbel move.
    if (is_root && param_->gumbel) {
        auto node = GumbelSelectChild(color, false, true);
        if (node) {
            return node;
        }
    }

    // Gather all parent's visits.
    int parentvisits = 0;
    float total_visited_policy = 0.0f;
    for (auto &child : children_) {
        const auto node = child.Get();
        const bool is_pointer = node != nullptr;

        if (is_pointer && node->IsValid()) {
            // The node status is pruned or active.
            const auto visits = node->GetVisits();
            parentvisits += visits;
            if (visits > 0) {
                total_visited_policy += child.GetPolicy();
            }
        }
    }

    // Cache the hyper-parameters.
    const auto cpuct_init           = param_->cpuct_init;
    const auto cpuct_base_factor    = param_->cpuct_base_factor;
    const auto cpuct_base           = param_->cpuct_base;
    const auto draw_factor          = param_->draw_factor;
    const auto score_utility_factor = param_->score_utility_factor;
    const auto score_utility_div    = param_->score_utility_div;
    const auto noise                = is_root ? param_->dirichlet_noise : false;
    const auto forced_playouts_k    = is_root ? param_->forced_playouts_k : 0.f;

    const float raw_cpuct     = cpuct_init + cpuct_base_factor *
                                    std::log((float(parentvisits) + cpuct_base + 1) / cpuct_base);
    const float numerator     = std::sqrt(float(parentvisits));
    const float fpu_value     = GetFpu(color, total_visited_policy, is_root);
    const float parent_score  = GetFinalScore(color);

    Edge* best_node = nullptr;
    float best_value = std::numeric_limits<float>::lowest();

    for (auto &child : children_) {
        const auto node = child.Get();
        const bool is_pointer = node != nullptr;

        // The node is pruned or invalid. Skip it.
        if (is_pointer && !node->IsActive()) {
            continue;
        }

        float q_value = fpu_value;
        const float psa = GetSearchPolicy(child, noise);

        float denom = 1.0f;
        float utility = 0.0f; // the utility value
        float cpuct = raw_cpuct;

        if (is_pointer) {
            const auto visits = node->GetVisits();

            if (node->IsExpanding()) {
                // Like virtual loss, give it a bad value because there are other
                // threads in this node.
                q_value = std::min(fpu_value, -1.0f);
            } else if (visits > 0) {
                // Transfer win-draw-loss to side-to-move value (Q value).
                const float eval = node->GetWL(color);
                const float draw_value = node->GetDraw() * draw_factor;
                q_value = eval + draw_value;

                // Heuristic value for score lead.
                utility += score_utility_factor *
                               node->GetScoreUtility(
                                   color, score_utility_div, parent_score);

                // Forced Playouts method. It can help to explore the low priority
                // child with high noise value. We think 20% is big enough and high
                // priority child is easy to be explored with PUCT. We don't need to
                // add any bouns for these kind of children.
                const float psa_factor = std::min(0.2f, psa);
                const float forced_n_factor =
                    std::max(1e-4f, forced_playouts_k * psa_factor * (float)parentvisits);
                const int forced_n = std::sqrt(forced_n_factor);
                if (forced_n - visits > 0) {
                    utility += (forced_n - visits) * 1e6;
                }
            }
            cpuct *= GetDynamicCpuctFactor(node, visits, parentvisits);
            denom += visits;
        }

        // PUCT algorithm
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

int Node::RandomMoveProportionally(float temp,
                                   float min_ratio,
                                   int min_visits) {
    auto select_vertex = kNullVertex;
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
        auto node = child.Get();
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

    if (accum_vector.empty()) {
        // All moves are pruned. In this case, we think
        // the random move is unsafe. Only return the best
        // move, the safest move.
        return GetBestMove(true);
    }

    auto distribution =
        std::uniform_real_distribution<decltype(accum)>{0.0, accum};
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

int Node::RandomMoveWithLogitsQ(GameState &state, float temp) {
    const auto num_intersections = state.GetNumIntersections();
    auto prob = std::vector<float>(num_intersections+1, 0.f);
    auto vertices_table = std::vector<int>(num_intersections+1, kNullVertex);
    int accm_visists = 0;

    for (const auto &child : children_) {
        auto node = child.Get();
        const auto visits = node->GetVisits();
        const auto vtx = child.GetVertex();
        int idx = num_intersections; // pass move

        // Do not need to prune the low visits move because
        // the Q value will reduce the probabilities of
        // bad moves.
        if (vtx != kPass) {
            idx = state.GetIndex(
                      state.GetX(vtx), state.GetY(vtx));
        }
        if (visits != 0) {
            accm_visists += visits;
            prob[idx] = visits;
            vertices_table[idx] = vtx;
        }
    }

    if (accm_visists == 0) {
        // There is no visits. Reture the best policy move.
        return GetBestMove(true);
    }
    for (float &p : prob) {
        p /= (float)accm_visists;
    }
    MixLogitsCompletedQ(state, prob);

    auto select_vertex = kNullVertex;
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

    if (accum_vector.empty()) {
        // What happened? Is it possible?
        return RandomMoveProportionally(temp, 0.f, 0);
    }

    auto distribution =
        std::uniform_real_distribution<decltype(accum)>{0.0, accum};
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

float Node::GetScoreUtility(const int color,
                            float div,
                            float parent_score) const {
    const auto score =
        GetFinalScore(color) + score_bouns_;
    return std::tanh((score - parent_score)/div);
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
        // We can not get the variance in the first visit. Return
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

    // Here are some errors to compute memory used.
    const auto mem_used = static_cast<double>(
        nodes * node_mem + edges * edge_mem) / (1024.f * 1024.f);

    const auto space2 = 10;
    out << " * Tree Status:" << std::endl
            << std::fixed << std::setprecision(4)
            << std::setw(space2) << "nodes:"   << ' ' << nodes    << std::endl
            << std::setw(space2) << "edges:"   << ' ' << edges    << std::endl
            << std::setw(space2) << "memory:"  << ' ' << mem_used << ' ' << "(MiB)" << std::endl;

    return out.str();
}

std::string Node::OwnershipToString(GameState &state, const int color, std::string name, Node *node) {
    auto out = std::ostringstream{};
    const auto board_size = state.GetBoardSize();

    auto ownership = node->GetOwnership(color);

    // TODO: A wrapper for row major iterator staring
    //       from A19.
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

        if (is_sayuri) {
            out << Format("info move %s visits %d winrate %.6f scorelead %.6f prior %.6f lcb %.6f order %d pv %s",
                             state.VertexToText(vertex).c_str(),
                             visits,
                             winrate,
                             final_score,
                             prior,
                             lcb,
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
    assert(HasChildren());

    const auto lcb_reduction = std::min(
        std::max(0.f, param_->lcb_reduction), 1.f);
    int parentvisits = 0;
    const auto parent_score = GetFinalScore(color);
    const auto score_utility_factor = param_->score_utility_factor;
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

        // The node is unvisited, pruned or invalid. Skip it.
        if (!is_pointer || !node->IsActive()) {
            continue;
        }

        const auto visits = node->GetVisits();
        if (visits > 0) {
            auto lcb = node->GetLcb(color);
            auto utility = score_utility_factor *
                               node->GetScoreUtility(
                                   color, score_utility_div, parent_score);
            const auto ulcb = (lcb + utility) * (1.f - lcb_reduction) +
                                  lcb_reduction * ((float)visits/parentvisits);
            list.emplace_back(ulcb, node->GetVertex());
        }
    }

    std::stable_sort(std::rbegin(list), std::rend(list));
    return list;
}

int Node::GetBestMove(bool allow_pass) {
    WaitExpanded();
    assert(HasChildren());

    auto lcblist = GetLcbUtilityList(color_);
    float best_value = std::numeric_limits<float>::lowest();
    int best_move = kNullVertex;

    for (auto &entry : lcblist) {
        const auto lcb = entry.first;
        const auto vtx = entry.second;
        if (lcb > best_value) {
            if (!allow_pass && vtx == kPass) {
                continue;
            }
            best_value = lcb;
            best_move = vtx;
        }
    }

    if (best_move == kNullVertex) {
        // There is no visited (non-pass) move. We use raw probabilities
        // instead of LCB list.
        best_move = ProbSelectChild(allow_pass)->GetVertex();
    }

    assert(best_move != kNullVertex);
    return best_move;
}

const std::vector<Node::Edge> &Node::GetChildren() const {
    return children_;
}

int Node::GetVirtualLoss() const {
    return VIRTUAL_LOSS_COUNT *
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
        // Punish the node if there are some threads under this
        // sub-tree.
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

float Node::GetFpu(const int color,
                   const float total_visited_policy,
                   const bool is_root) const {
    // Apply First Play Urgency (FPU). We should think the value of the
    // unvisited nodes are same as parent's. The NN-based MCTS favors
    // the visited node. So give the unvisited node a little bad favour
    // (FPU reduction) in order to reduce the priority.
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

float Node::GetGumbelQValue(int color, float parent_score) const {
    // Get non-transform complete Q value. In the original
    // paper, it is Q value. We mix Q value and score lead
    // in order to optimize the move probabilities. Make it
    // playing the best move.
    const auto score_utility_div = param_->score_utility_div;
    const auto score_utility_factor = param_->score_utility_factor;
    const auto utility = score_utility_factor *
                                 GetScoreUtility(
                                     color, score_utility_div, parent_score);
    const auto mixed_q = GetWL(color, false) + utility;
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
        int idx = num_intersections; // pass move
        if (vtx != kPass) {
            idx = state.GetIndex(
                      state.GetX(vtx), state.GetY(vtx));
        }
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

    // The 'prob' size should be intersections number plus
    // one pass move.
    if (num_intersections + 1 != static_cast<int>(prob.size())) {
        return;
    }

    const auto parent_score = GetFinalScore(color);
    auto logits_q = GetZeroLogits<float>(num_intersections+1);

    int max_visits = 0;
    int parentvisits = 0;;
    float weighted_q = 0.f;
    float weighted_pi = 0.f;

    // Gather some basic informations.
    for (auto & child : children_) {
        const auto node = child.Get();
        const bool is_pointer = node != nullptr;

        int visits = 0;
        if (is_pointer && node->IsActive()) {
            visits = node->GetVisits();
        }
        parentvisits += visits;
        max_visits = std::max(max_visits, visits);

        if (visits > 0) {
            weighted_q += child.GetPolicy() *
                              node->GetGumbelQValue(color, parent_score);
            weighted_pi += child.GetPolicy();
        }
    }

    // Compute the all children's completed Q.
    auto completed_q_list = std::vector<float>{};

    // Mix the raw value and approximate Q value. It may help
    // to improve the performance when the parentvisits is very
    // low (<= 4).
    const float raw_value = GetNetWL(color);
    const float approximate_q = (raw_value + (parentvisits/weighted_pi) *
                                    weighted_q) / (1 + parentvisits);
    for (auto & child : children_) {
        const auto node = child.Get();
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
            completed_q = node->GetGumbelQValue(color, parent_score);
        }
        completed_q_list.emplace_back(completed_q);
    }

    // Apply the completed Q with policy.
    int completed_q_idx = 0;
    for (auto & child : children_) {
        const auto vtx = child.GetVertex();
        int idx = num_intersections; // pass move
        if (vtx != kPass) {
            idx = state.GetIndex(
                      state.GetX(vtx), state.GetY(vtx));
        }

        const float logits = SafeLog(prob[idx]);
        const float completed_q = completed_q_list[completed_q_idx++];

        // Transform the Completed Q value because it makes
        // policy logit and Q value balance.
        logits_q[idx] = logits + TransformCompletedQ(
                                     completed_q, max_visits);
    }
    prob = Softmax(logits_q, 1.f);

    // Prune the bad policy and rescale the policy.
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
    // We simply think the parent's visits is current
    // visits. Include the pruned and invalid nodes.
    const int visits = GetVisits() - 1;
    return param_->gumbel &&
               param_->gumbel_playouts_threshold > visits;
}

bool Node::ProcessGumbelLogits(std::vector<float> &gumbel_logits,
                               const int color,
                               const bool only_max_visits) {
    // The variant of Sequential Halving algorithm. The input N playouts is always
    // '(promotion visits) * (log2(considered moves) + 1) * (considered moves)' for
    // each epoch. The variant algorithm is same as Sequential Halving if the total
    // playous is lower than this value. Following is a example,
    //
    // promotion visits = 1
    // considered moves = 4
    //
    // * Round 0.
    //  accumulation -> { 0, 0, 0, 0 }
    //
    // * Round 1.
    //  distribution -> { 1, 1, 1, 1 }
    //  accumulation -> { 1, 1, 1, 1 }
    //
    // * Round 2.
    //  distribution -> { 2, 2, 0, 0 }
    //  accumulation -> { 3, 3, 1, 1 }
    //
    // * Round 3.(1st epoch is end)
    //  distribution -> { 4, 0, 0, 0 }
    //  accumulation -> { 7, 3, 1, 1 }
    //
    // * Round 4.
    //  distribution -> { 1, 1, 1, 1 }
    //  accumulation -> { 8, 4, 2, 2 }
    //
    // * Round 5.
    //  distribution -> {  2, 2, 0, 0 }
    //  accumulation -> { 10, 6, 2, 2 }
    //
    // * Round 6. (2nd epoch is end)
    //  distribution -> {  4, 0, 0, 0 }
    //  accumulation -> { 14, 6, 2, 2 }

    const int size = children_.size();
    auto table = std::vector<std::pair<int, int>>(size);
    gumbel_logits.resize(size, LOGIT_ZERO);

    for (int i = 0; i < size; ++i) {
        auto &child = children_[i];
        const auto node = child.Get();
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

    // We may reuse the sub-tree. Try fill old distribution in order
    // to cover the Sequential Halving distribution. For example,
    //
    // Original distribution (sorted)
    // { 9, 2, 0, 0 }
    //
    // Target Sequential Halving distribution
    // { 7, 3, 1, 1 }
    //
    // Addition playouts distribution
    // { 0, 1, 1, 1 }
    //
    // Result distribution
    // { 9, 3, 1, 1 }
    while (true) {
        for (int i = 0; i < level; ++i) {
            // Keep to minus current distribution according to Sequential
            // Halving, the first zero visits is target child.
            for (int j = 0; j < width; ++j) {
                if (table[j].first <= 0) {
                    auto vtx = table[j].second;
                    target_visits = GetChild(vtx)->GetVisits();
                    goto end_loop;
                }
                table[j].first -= 1;
                playouts_thres -= 1;

                if (playouts_thres <= 0) {
                    // The current distribution cover the Sequential Halving
                    // distribution. Disable the Gumbel noise.
                    goto end_loop;
                }
            }
        }
        if (width == 1) {
            // Go to next epoch.
            width = adj_considered_moves;
            level = prom_visits;
        } else {
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
        const auto node = child.Get();
        const bool is_pointer = node != nullptr;

        if (is_pointer && !node->IsActive()) {
            // The node is pruned or invalid. Skip it.
            continue;
        }
        if (target_visits == child.GetVisits()) {
            auto logit = gumbel_type1(Random<>::Get()) +
                             SafeLog(child.GetPolicy());
            auto completed_q = 0.f;
            if (is_pointer && target_visits > 0) {
                completed_q = TransformCompletedQ(
                    node->GetGumbelQValue(color, parent_score), max_visists);
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
        return best_node_no_pass->Get();
    }
    Inflate(*best_node);
    return best_node->Get();
}

int Node::GetGumbelMove(bool allow_pass) {
    WaitExpanded();
    assert(HasChildren());

    int num_candidates = 0;
    for (auto &child : children_) {
        const auto node = child.Get();
        const int visits = child.GetVisits();
        if (visits > 0 && node->IsValid()) {
            num_candidates += 1;
        }
    }

    if (!allow_pass && num_candidates == 1) {
        // Only one candidate move. It may be the pass move.
        allow_pass = true;
    }

    return GumbelSelectChild(color_, true, allow_pass)->GetVertex();
}

void Node::SetScoreBouns(float val) {
    score_bouns_ = val;
}

void Node::KillRootSuperkos(GameState &state) {
    for (const auto &child : children_) {
        const auto vtx = child.GetVertex();

        auto fork_state = state;
        fork_state.PlayMove(vtx);

        if (vtx != kPass &&
                fork_state.IsSuperko()) {
            // Kill the superko move.
            child.Get()->Invalidate();
        }
    }

    auto ite = std::remove_if(std::begin(children_), std::end(children_),
                              [](Edge &ele) {
                                  return !ele.Get()->IsValid();
                              });
    children_.erase(ite, std::end(children_));
}
