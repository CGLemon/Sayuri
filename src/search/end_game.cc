#include "search/end_game.h"
#include "game/types.h"
#include "utils/random.h"

#include <algorithm>

#include <iostream>

EndGame &EndGame::Get(GameState &state) {
    static EndGame end_game(state);
    return end_game;
}

std::vector<int> EndGame::GetFinalOwnership() const {
    auto root_board = root_state_.board_;

    // We assume that the game is end. But still some stone
    // not be captured.  

    auto num_intersections = root_board.GetNumIntersections();
    auto buffer = std::vector<int>(num_intersections, 0);
    auto cnt = 0;

    static constexpr int kPlayoutsCount = 100;

    for (int p = 0; p < kPlayoutsCount; ++p) {
        auto final_ownership = RandomRollout(root_board);
        if (!final_ownership.empty()) {
            cnt += 1;
            for (int idx = 0; idx < num_intersections; ++idx) {
                auto owner = final_ownership[idx];
                if (owner == kBlack) {
                    buffer[idx] += 1;
                } else if (owner == kWhite) {
                    buffer[idx] -= 1;
                } else if (owner == kEmpty) {
                    buffer[idx] += 0;
                } 
            }
        }
    }

    auto reuslt = std::vector<int>(num_intersections, kInvalid);

    if (cnt == 0) {
        return reuslt;
    }

    for (int idx = 0; idx < num_intersections; ++idx) {
        if (buffer[idx] >= 0.9 * kPlayoutsCount) {
            reuslt[idx] = kBlack;
        } else if (buffer[idx] <= (-0.9 * kPlayoutsCount)) {
            reuslt[idx] = kWhite;
        } else if (buffer[idx] == 0) {
            reuslt[idx] = kEmpty;
        }
    }

    return reuslt;
}

std::vector<int> EndGame::RandomRollout(Board board) const {
    auto num_intersections = board.GetNumIntersections();
    auto boardsize = board.GetBoardSize();

    while(true) {
        auto color = board.GetToMove();
        auto simple_ownership = board.GetSimpleOwnership();
        auto movelist = std::vector<int>{};
        auto buffer = std::vector<bool>{};

        for (int idx = 0; idx < num_intersections; ++idx) {
            const auto x = idx % boardsize;
            const auto y = idx / boardsize;
            const auto vtx = board.GetVertex(x, y);

            if (simple_ownership[idx] == kEmpty ||
                    board.IsCaptureMove(vtx, color) ||
                    board.IsEscapeMove(vtx, color)) {
                movelist.emplace_back(vtx);
            }
        }


        if (!RandomMove(board, movelist)) {
            board.PlayMoveAssumeLegal(kPass, color);
        }

        if (board.GetPasses() >= 4) {
            break;     
        }
    }

    return board.GetSimpleOwnership();
}

bool EndGame::RandomMove(Board &current_board, std::vector<int> &movelist) const {
    auto size = movelist.size();
    auto color = current_board.GetToMove();
    auto mark = std::vector<bool>(size, false);

    Board board;
    auto stop = false;

    if (movelist.empty()) {
        return false;
    }

    while(!stop) {
        stop = true;
        for (auto s = size_t{0}; s < size; ++s) {
            if (!mark[s]) {
                stop = false;
                break;
            }
        }

        auto choice = Random<RandomType::kXoroShiro128Plus>::Get().Generate() % size;
        if (mark[choice]) {
            continue;
        }

        auto vertex = movelist[choice];

        if (!current_board.IsLegalMove(vertex, color)) {
            mark[choice] = true;
            continue;
        }
        board = current_board;
        board.PlayMoveAssumeLegal(vertex, color);

        if (board.GetLiberties(vertex) == 1) {
            mark[choice] = true;
            continue;
        }

        current_board = board; 
        break;
    }

    return !stop;
}

