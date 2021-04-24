#include "search/end_game.h"
#include "game/types.h"

#include <algorithm>

#include <iostream>
#include <iomanip>

EndGame &EndGame::Get(GameState &state) {
    static EndGame end_game(state);
    return end_game;
}

std::vector<int> EndGame::GetFinalOwnership() const {
    auto passes = root_state_.GetPasses();
    if (passes < 2) {
        // The game is not over.
        // return std::vector<int>{};
    }

    auto board = root_state_.board_;
    auto simple_ownership = board.GetSimpleOwnership();

    // We assume that the string is alive if it has at least
    // one eye, first.
    auto lived_groups = GetLivedGroups(board);
    CompareAndRemoveDeadString(board, lived_groups);

    std::cout << board.GetBoardString(board.GetLastMove(), false);
    std::cout << std::endl;
    std::cout << std::endl;
    simple_ownership = board.GetSimpleOwnership();

    auto board_size = board.GetBoardSize();
    for (int y = 0; y < board_size; ++y) {
        for (int x = 0; x < board_size; ++x) {
            auto idx = board.GetIndex(x, board_size - y - 1);
            std::cout << std::setw(4) << simple_ownership[idx];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    return simple_ownership;
}

std::vector<int> EndGame::GatherVertex(std::vector<bool> &buf) const {
    auto result = std::vector<int>{};

    for (auto vtx = 0; vtx < buf.size(); ++vtx) {
        if (buf[vtx]) {
            result.emplace_back(vtx);
        }
    }

    return result;
}

std::vector<int> EndGame::GetLivedGroups(Board &board) const {
    auto board_size = board.GetBoardSize();
    auto num_intersections = board.GetNumIntersections();
    auto num_vertices = board.GetNumVertices();

    auto lived_groups = std::vector<int>(num_vertices);
    auto territory = std::vector<bool>(num_vertices);

    auto simple_ownership = board.GetSimpleOwnership();

    auto group_count = 1;
    std::fill(std::begin(lived_groups), std::begin(lived_groups), -1);
    std::fill(std::begin(territory), std::begin(territory), false);

    for (int index = 0; index < num_intersections; ++index) {
        auto x = index % board_size;
        auto y = index / board_size;
        auto vtx = board.GetVertex(x, y);
        lived_groups[vtx] = 0;
    }

    int directions[4] = {
        board_size+2, 1, -1, -board_size-2
    };

    for (int index = 0; index < num_intersections; ++index) {
        auto x = index % board_size;
        auto y = index / board_size;

        auto vtx = board.GetVertex(x, y);
        auto color = board.GetState(vtx);
        auto owner = simple_ownership[index];

        if (owner != kEmpty &&
                color == kEmpty &&
                territory[vtx] == false) {
            auto territory_buffer = std::vector<bool>{};
            board.ComputeReachGroup(vtx, kEmpty, territory_buffer);

            auto connection_strings = std::vector<bool>(num_vertices);
            auto vertex_group = GatherVertex(territory_buffer);

            for (const auto avtx : vertex_group) {
                territory[avtx] = true;
                lived_groups[avtx] = group_count;

                for (int k = 0; k < 4; ++k) {
                    const auto aavtx = avtx + directions[k];
                    if (!connection_strings[aavtx] && 
                            board.GetState(aavtx) == owner) {
                        board.ComputeReachGroup(aavtx, owner, connection_strings);
                    }
                }
            }

            vertex_group = GatherVertex(connection_strings);

            for (const auto avtx : vertex_group) {
                lived_groups[avtx] = group_count;
            }
            group_count++;
        }
    }


    return lived_groups;
}

void EndGame::FillMoves(Board &board, int color, std::vector<int> &vertex_group) const {
    for (const auto vtx : vertex_group) {
        assert(board.IsLegalMove(vtx, color));
        board.PlayMoveAssumeLegal(vtx, color);
    }
}

void EndGame::CompareAndRemoveDeadString(Board &board,
                                         std::vector<int> &lived_groups) const {
    auto board_size = board.GetBoardSize();
    auto num_intersections = board.GetNumIntersections();
    auto num_vertices = board.GetNumVertices();

    for (int index = 0; index < num_intersections; ++index) {
        auto x = index % board_size;
        auto y = index / board_size;

        auto vtx = board.GetVertex(x, y);
        auto color = board.GetState(vtx);
        auto group = lived_groups[vtx];

        if (group == 0 && color != kEmpty) {
            auto dead_string = std::vector<bool>(num_vertices);
            board.ComputeReachGroup(vtx, kEmpty, dead_string);

            dead_string[vtx] = false;

            auto vertex_group = GatherVertex(dead_string);
            FillMoves(board, !color, vertex_group);
        } 
    }
}

int EndGame::ComputeNumEye(std::vector<int> &eye_group) const {
    return 0;
}
