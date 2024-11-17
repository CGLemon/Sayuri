#include "mcts/node.h"
#include "neural/network_basic.h"
#include "utils/random.h"

#include <random>

#include <iostream>
#include <iomanip>

float TransferRelativeRank(int relative_rank) {
    constexpr int kMaxRelativeRank = 25;
    return static_cast<float>(std::min(relative_rank, kMaxRelativeRank)) / kMaxRelativeRank;
}

float GetMinimalWLEvalDiff(int relative_rank) {
    float factor = TransferRelativeRank(relative_rank);
    return (1.0f - factor) * 0.05f + factor * 0.1f;
}

float GetMinimalScorelDiff(int relative_rank) {
    float factor = TransferRelativeRank(relative_rank);
    return (1.0f - factor) * 5.0f + factor * 10.f;
}

float ComputeAccumProbUpperBound(int relative_rank) {
    float factor = TransferRelativeRank(relative_rank);
    return (1.0f - factor) * 0.35f + factor * 0.55f;
}

float ComputeProbLowerBound(int relative_rank) {
    float factor = TransferRelativeRank(relative_rank);
    return (1.0f - factor) * 0.0025f + factor * 0.0015f;
}

int Node::GetRankMove() {
    auto edgelist = std::vector<const Edge*>{};
    for (const auto & child : children_) {
        const auto node = child.Get();
        const bool is_pointer = node != nullptr;

        if (!is_pointer || !node->IsActive()) {
            continue;
        }
        if (child.GetVisits() <= 0) {
            continue;
        }
        edgelist.emplace_back(&child);
    }

    int best_move = GetBestMove(true);
    if (edgelist.empty()) {
        return best_move;
    }
    std::shuffle(std::begin(edgelist),
        std::end(edgelist), Random<>::Get());

    Node* best_node = GetChild(best_move);
    const auto best_wl = best_node->GetWL(color_);
    const auto best_score = best_node->GetFinalScore(color_);
    assert(best_node->GetVisits() > 0);

    const auto min_wl_diff = GetMinimalWLEvalDiff(param_->relative_rank);
    const auto min_score_diff = GetMinimalScorelDiff(param_->relative_rank);
    for (const auto & child : edgelist) {
        const auto node = child->Get();
        const auto wl = node->GetWL(color_);
        const auto score = node->GetFinalScore(color_);

        if (best_wl - wl < min_wl_diff &&
                best_score - score < min_score_diff) {
            return node->GetVertex();
        }
    }
    return best_move;
}

void Node::RandomPruneRootChildren(Network &network, GameState &state) {
    if (param_->relative_rank < 0) {
        return;
    }
    if (state.GetBoardSize() != 19) {
        return;
    }

    auto raw_netlist = network.GetOutput(
        state, Network::kRandom, Network::Query::Get().SetOffset(PolicyBufferOffset::kSoft));
    auto movelist = std::vector<Network::PolicyVertexPair>{};
    const auto num_intersections = state.GetNumIntersections();

    assert(HasChildren());
    InflateAllChildren();
    for (int idx = 0; idx < num_intersections; ++idx) {
        const auto vtx = state.IndexToVertex(idx);
        const auto policy = raw_netlist.probabilities[idx];
        const auto node = GetChild(vtx);
        const bool is_pointer = node != nullptr;

        if (!is_pointer || !node->IsActive()) {
            continue;
        }
        movelist.emplace_back(policy, vtx);
    }
    auto passnode = GetChild(kPass);
    if (passnode) {
        movelist.emplace_back(raw_netlist.pass_probability, kPass);
    }
    std::stable_sort(std::rbegin(movelist), std::rend(movelist));

    const auto upper_bound_accum_porb = ComputeAccumProbUpperBound(param_->relative_rank);
    const auto lower_bound_prob = ComputeProbLowerBound(param_->relative_rank);
    auto accum_porb = 0.0f;
    int upper_index = 0;
    for (auto it: movelist) {
        const auto policy = it.first;
        if (accum_porb >= upper_bound_accum_porb) {
            break;
        }
        if (lower_bound_prob >= policy) {
            break;
        }
        accum_porb += policy;
        upper_index += 1;
    }
    movelist.resize(upper_index);

    // prune all active node first
    for (const auto &child : children_) {
        const auto node = child.Get();
        if (!node->IsActive()) {
            continue;
        }
        node->SetActive(false);
    }

    // activate the selection children
    for (auto it: movelist) {
        const auto vtx = it.second;
        const auto node = GetChild(vtx);

        if (node->IsPruned()) {
            node->SetActive(true);
        }
    }
}
