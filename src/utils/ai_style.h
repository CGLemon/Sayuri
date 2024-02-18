#pragma once

#include "utils/random.h"

#include <algorithm>
#include <utility>
#include <random>
#include <unordered_set>
#include <unordered_map>
#include <cmath>

enum class SytleType {
    kNormal,
    kRandom
};

template<typename T>
using SelectionVector = std::vector<std::pair<double, T>>;

template<typename T>
SelectionVector<T> WeightedSelection(SelectionVector<T> selection_vector,
                                     int n_moves) {
    // Random select n moves with policy weighted.
    auto distribution =
        std::uniform_real_distribution<double>{1e-8, 1.0};

    for (auto &it : selection_vector) {
        double val = std::log(distribution(Random<>::Get()));
        double denom = 1.0 + it.first;
        it.first = val / denom; // magic
    }
    std::sort(std::rbegin(selection_vector),
                  std::rend(selection_vector));
    selection_vector.resize(n_moves);

    return selection_vector;
}

template<int TOPRANK>
int GetMaxMoves(int relative_rank, int num_intersections, int num_legal_moves) {
    // The formula is imported from katrain. Random select
    // k candidate moves. We can choose the low k number to
    // limit veiw of network. It may be effective to reduce
    // the netowork strength. You may see the issues here,
    // 1. https://github.com/sanderland/katrain/issues/44
    // 2. https://github.com/sanderland/katrain/issues/74
    const auto rank = 15.f - std::min(TOPRANK, std::max(0, relative_rank));

    const auto orig_calib_avemodrank =
        0.063015f + 0.7624f * num_intersections /
            std::pow(10.f, -0.05737f * rank + 1.9482f);
    const auto norm_leg_moves =
        static_cast<float>(num_legal_moves) / num_intersections;
    const auto avemodrank_exp =
        3.002f * norm_leg_moves * norm_leg_moves -
        norm_leg_moves -
        0.034889f * rank -
        0.5097f;
    const auto modified_calib_avemodrank = (
        0.3931f +
        0.6559f *
        norm_leg_moves *
        std::exp(-1 * (avemodrank_exp * avemodrank_exp)) -
        0.01093f * rank) * orig_calib_avemodrank;
    int n_moves = std::round(
        num_intersections * norm_leg_moves /
        (1.31165f * (modified_calib_avemodrank + 1.f) - 0.082653f));
    n_moves = std::max(1, n_moves);
    n_moves = std::min(num_legal_moves, n_moves);
    return n_moves;
}

template<typename T>
void RewritePriority(SelectionVector<T> &selection, SytleType sytle) {
    if (sytle == SytleType::kNormal) {
        // origin policy only
        return;
    }

    if (sytle == SytleType::kRandom) {
        for (auto &it : selection) {
            it.first = 0.f;
        }
    }
}

template<typename T>
SelectionVector<T> SaveHighPriorityItem(SelectionVector<T> &selection,
                                        int num_intersections,
                                        int num_legal_moves) {
    const auto override_top =
        0.8 * (1.0 - 0.5 * (num_intersections - num_legal_moves) / num_intersections);
    auto saved = SelectionVector<T>{};

    for (auto &it : selection) {
        double priority = it.first;
        if (priority > override_top) {
            saved.emplace_back(it);
        } 
    }

    // FIXME: Seem the std::remove_if has some bugs.
    //
    // auto it = std::remove_if(std::begin(selection), std::end(selection),
    //                              [override_top](auto &it) {
    //                                  double priority = it.first;
    //                                  return priority > override_top;
    //                              });
    // saved.insert(std::begin(saved), it, std::end(selection));
    // selection.erase(it, std::end(selection));

    return saved;
}

template<typename T>
SelectionVector<T> GetRelativeRankVector(SelectionVector<T> selection,
                                         int relative_rank,
                                         int num_intersections,
                                         SytleType sytle = SytleType::kNormal) {
    int num_legal_moves = selection.size();
    auto saved = SaveHighPriorityItem(
        selection, num_intersections, num_legal_moves);

    RewritePriority(selection, sytle);

    int n_moves = GetMaxMoves<25>(
        relative_rank, num_intersections, num_legal_moves);

    selection = WeightedSelection<T>(selection, n_moves);
    selection.insert(std::begin(selection), std::begin(saved), std::end(saved));

    std::sort(std::rbegin(selection), std::rend(selection));
    selection.erase(std::unique(std::begin(selection), std::end(selection)),
                    std::end(selection));

    return selection;
}
