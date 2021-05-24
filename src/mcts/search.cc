#include "mcts/search.h"

Search::~Search() {
    group_->WaitToJoin();
}

void Search::Initialize() {
    param_ = std::make_shared<Parameters>();
    param_->Reset();

    threads_ = param_->threads;

    group_ = std::make_unique<ThreadGroup<void>>(&ThreadPool::Get(threads_));


    max_playouts_ = param_->playouts;
    playouts_.store(0);
}

Search::Result Search::Think() {
    auto result = Result{};

    auto nn_output = network_.GetOutput(root_state_, Network::RANDOM_SYMMETRY);
    auto num_intersections = root_state_.GetNumIntersections();
    auto max_index = 0;
    for (int idx = 0; idx < num_intersections; ++idx) {
        if (nn_output.probabilities[max_index] < nn_output.probabilities[idx]) {
            max_index = idx;
        }
    }

    auto x = max_index % nn_output.board_size;
    auto y = max_index / nn_output.board_size;
    result.best_move = root_state_.GetVertex(x, y);

    return result;
}

int Search::ThinkBestMove() {
    auto result = Think();
    return result.best_move;
}
