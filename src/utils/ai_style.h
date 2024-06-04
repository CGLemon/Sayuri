#pragma once

#include "utils/random.h"

#include <algorithm>
#include <array>
#include <utility>
#include <random>
#include <unordered_set>
#include <unordered_map>
#include <cmath>
#include <tuple>

enum class StyleType {
    kDefault,
    kPriority
};

template<typename T>
using SelectionItem = std::tuple<double, T>; // priority, item

template<typename T>
using SelectionVector = std::vector<SelectionItem<T>>;

template<int TOPRANK>
int GetMaxNumMoves(int relative_rank, int num_intersections, int num_legal_moves) {
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

template<int TOPRANK>
double GetMaxAccumulationPriority(int relative_rank) {
    relative_rank = std::min(TOPRANK, std::max(0, relative_rank));
    double prior = std::min(
        0.75 + 0.25 * std::pow(relative_rank/static_cast<double>(TOPRANK), 1.5),
        1.0
    );
    return prior;
}

template<typename T>
SelectionVector<T> RandomSelection(SelectionVector<T> selection_vector,
                                   int relative_rank,
                                   int num_intersections) {
    int num_legal_moves = selection_vector.size();
    int n_moves = GetMaxNumMoves<25>(
        relative_rank, num_intersections, num_legal_moves);

    std::shuffle (std::begin(selection_vector), std::end(selection_vector), Random<>::Get());
    selection_vector.resize(n_moves);

    return selection_vector;
}

template<typename T>
SelectionVector<T> PrioritySelection(SelectionVector<T> selection_vector,
                                     int relative_rank,
                                     int num_intersections) {
    double max_accm_prior = GetMaxAccumulationPriority<25>(relative_rank);
    double accm_prior = 0.0;
    int num_legal_moves = selection_vector.size();
    int n_moves = 0;

    std::sort(std::rbegin(selection_vector),
                  std::rend(selection_vector));
    for (auto &it : selection_vector) {
        ++n_moves;
        accm_prior += std::get<0>(it);
        if (accm_prior > max_accm_prior) {
            break;
        }
    }

    int max_n_moves = std::sqrt(
            1.5 * 
            static_cast<double>(num_intersections)/num_legal_moves *
            GetMaxNumMoves<25>(relative_rank, num_intersections, num_legal_moves));
    max_n_moves = std::max(2, max_n_moves);
    n_moves = std::min(n_moves, max_n_moves);

    selection_vector.resize(n_moves);
    return selection_vector;
}


template<typename T>
SelectionVector<T> GetRelativeRankVector(SelectionVector<T> selection,
                                         int relative_rank,
                                         int num_intersections,
                                         StyleType style) {
    if (style == StyleType::kPriority) {
        selection = PrioritySelection<T>(
            selection, relative_rank, num_intersections);
    } else {
        selection = RandomSelection<T>(
            selection, relative_rank, num_intersections);
    }

    std::sort(std::rbegin(selection), std::rend(selection));
    selection.erase(std::unique(std::begin(selection), std::end(selection)),
                    std::end(selection));

    return selection;
}
