#include "game/board.h"

#include <algorithm>

int Board::ComputeScoreOnBoard(int black_bonus) const {
    const auto black = ComputeReachColor(kBlack);
    const auto white = ComputeReachColor(kWhite);
    return black - white + black_bonus;
}

float Board::ComputeSimpleFinalScore(float komi) const {
    return static_cast<float>(ComputeScoreOnBoard(0)) - komi;
}

void Board::ComputeSimpleOwnership(std::vector<int> &buffer) const {
    if (buffer.size() != (size_t) num_intersections_) {
        buffer.resize(num_intersections_);
    }
    auto black = std::vector<bool>(num_intersections_, false);
    auto white = std::vector<bool>(num_intersections_, false);
    auto PeekState = [&](int vtx) -> int {
        return state_[vtx];
    };

    ComputeReachColor(kBlack, kEmpty, black, PeekState);
    ComputeReachColor(kWhite, kEmpty, white, PeekState);

    for (int y = 0; y < board_size_; ++y) {
        for (int x = 0; x < board_size_; ++x) {
            const auto idx = GetIndex(x, y);
            const auto vtx = GetVertex(x, y);

            if (black[vtx] && !white[vtx]) {
                buffer[idx] = kBlack;  
            } else if (white[vtx] && !black[vtx]) {
                buffer[idx] = kWhite;
            } else {
                //The point belongs to both.
                buffer[idx] = kEmpty;
            }
       }
    }
}

std::vector<int> Board::GetSimpleOwnership() const {
    auto res = std::vector<int>(num_intersections_, kInvalid);

    ComputeSimpleOwnership(res);

    return res;
}

bool Board::ValidHandicap(int handicap) {
    if (handicap < 2 || handicap > 9) {
        return false;
    }
    if (board_size_ % 2 == 0 && handicap > 4) {
        return false;
    }
    if (board_size_ == 7 && handicap > 4) {
        return false;
    }
    if (board_size_ < 7 && handicap > 0) {
        return false;
    }

    return true;
}

bool Board::SetFixdHandicap(int handicap) {
    if (!ValidHandicap(handicap)) {
        return false;
    }

    int high = board_size_ >= 13 ? 3 : 2;
    int mid = board_size_ / 2;

    int low = board_size_ - 1 - high;
    if (handicap >= 2) {
        PlayMoveAssumeLegal(GetVertex(low, low),  kBlack);
        PlayMoveAssumeLegal(GetVertex(high, high),  kBlack);
    }

    if (handicap >= 3) {
        PlayMoveAssumeLegal(GetVertex(high, low), kBlack);
    }

    if (handicap >= 4) {
        PlayMoveAssumeLegal(GetVertex(low, high), kBlack);
    }

    if (handicap >= 5 && handicap % 2 == 1) {
        PlayMoveAssumeLegal(GetVertex(mid, mid), kBlack);
    }

    if (handicap >= 6) {
        PlayMoveAssumeLegal(GetVertex(low, mid), kBlack);
        PlayMoveAssumeLegal(GetVertex(high, mid), kBlack);
    }

    if (handicap >= 8) {
        PlayMoveAssumeLegal(GetVertex(mid, low), kBlack);
        PlayMoveAssumeLegal(GetVertex(mid, high), kBlack);
    }

    SetToMove(kWhite);
    SetLastMove(kNullVertex);
    SetMoveNumber(0);

    return true;
}

bool Board::SetFreeHandicap(std::vector<int> movelist) {
    for (const auto vtx : movelist) {
        if (IsLegalMove(vtx, kBlack)) {
            PlayMoveAssumeLegal(vtx, kBlack);
        } else {
            return false;
        }
    }

    SetToMove(kWhite);
    SetLastMove(kNullVertex);
    SetMoveNumber(0);

    return true;
}

std::vector<LadderType> Board::GetLadderPlane() const {
    auto res = std::vector<LadderType>(GetNumIntersections(), LadderType::kNotLadder);
    auto ladder = std::vector<int>{};
    auto not_ladder = std::vector<int>{};

    const auto VectorFind = [](std::vector<int> &arr, int element) -> bool {
        auto begin = std::begin(arr);
        auto end = std::end(arr);
        return std::find(begin, end, element) != end;
    };

    auto bsize = GetBoardSize();
    for (int y = 0; y < bsize; ++y) {
        for (int x = 0; x < bsize; ++x) {
            const auto idx = GetIndex(x, y);
            const auto vtx = GetVertex(x, y);

            auto first_found = false;
            int libs = 0;
            auto parent = strings_.GetParent(vtx);

            if (VectorFind(ladder, parent)) {
                // Be found! It is a ladder.
                libs = strings_.GetLiberty(parent);
            } else if (!VectorFind(not_ladder, parent)) {
                // Not be found! Now Search it.
                if (IsLadder(vtx)) {
                    // It is a ladder.
                    ladder.emplace_back(parent);
                    first_found = true; 
                    libs = strings_.GetLiberty(parent);
                } else {
                    // It is not a ladder.
                    not_ladder.emplace_back(parent);
                    continue;
                }
            } else {
                // It is not a ladder.
                continue;
            }

            assert(libs == 1 || libs == 2);
            if (libs == 1) {
                // The ladder string is already death.
                res[idx] = LadderType::kLadderDeath;
            } else {
                // The ladder string has a chance to escape.
                res[idx] = LadderType::kLadderEscapable;
            }

            if (first_found) {
                auto buf = std::vector<int>{};
                auto move_num = FindStringLiberties(vtx, buf);
            #ifdef NDEBUG
                (void)move_num;
            #else
                assert(move_num == libs);
            #endif
                for (const auto &v : buf) {
                    const auto ax = GetX(v);
                    const auto ay = GetY(v);
                    const auto aidx = GetIndex(ax, ay); 
                    if (libs == 1) {
                        // Someone can capture this ladder string.
                        res[aidx] = LadderType::kLadderTake;
                    } else {
                        // Someone can atari this ladder string.
                        res[aidx] = LadderType::kLadderAtari;
                    }
                }
            }
        }
    }

    return res;
}

std::vector<bool> Board::GetOcupiedPlane(const int color) const {
    auto res = std::vector<bool>(GetNumIntersections(), false);
    auto bsize = GetBoardSize();
    for (int y = 0; y < bsize; ++y) {
        for (int x = 0; x < bsize; ++x) {
            const auto state = GetState(x, y);
            const auto index = GetIndex(x, y);
            if (state == color) {
                res[index] = true;
            }
        }
    }

    return res;
}

std::vector<bool> Board::GetPassAlive(const int color) const {
    auto num_vertices = GetNumVertices();

    auto ocupied = std::vector<int>(num_vertices, kInvalid);
    auto eyes = std::vector<bool>(num_vertices, false);
    auto bsize = GetBoardSize();

    for (int y = 0; y < bsize; ++y) {
        for (int x = 0; x < bsize; ++x) {
            auto vertex = GetVertex(x, y);
            auto state = GetState(vertex);
            if (state == color) {
                ocupied[vertex] = color;
            } else {
                ocupied[vertex] = kEmpty;
            }

            if (state == kEmpty && IsSimpleEye(vertex, color)) {
                eyes[vertex] = true;
            }
        }
    }

    auto pass_alive_groups = std::vector<int>(num_vertices, -1);
    auto group_count = ClassifyGroups(ocupied, pass_alive_groups, color);

    auto string_linking = std::vector<int>(group_count);
    for (int i = 0; i < group_count; ++i) {
        int linking = 0;

        for (int v = 0; v < num_vertices; ++v) {
            if (i+1 == pass_alive_groups[v]) {
                linking = v;
                break;
            }
        }
        string_linking[i] = linking;
    }

    // Start the Benson's algorithm.
    // https://senseis.xmp.net/?BensonsAlgorithm
    while(true) {
        auto change = false;

        for (int i = 0; i < group_count; ++i) {
            auto vertex = string_linking[i];
            if (vertex == kNullVertex) {
                // The string is not pass alive.
                continue;
            }
            auto index = pass_alive_groups[vertex];
            auto eye_cnt = 0;
            auto surround = FindStringSurround(pass_alive_groups, index);
            
            for (const auto v : surround) {
                if (eyes[v]) {
                    eye_cnt++;
                }
            }
            if (eye_cnt < 2) {
                // If the string doesn't have enough eyes. It is not
                // a pass alive string.
                for (auto &idx : pass_alive_groups) {
                    if (idx == index) idx = 0;
                }
                string_linking[i] = kNullVertex;
                change = true;

                // The eyes is broken.
                for (const auto v : surround) {
                    if (eyes[v]) {
                        eyes[v] = false;
                    }
                }
            }
        }

        if (!change) break;
    }

    auto result = std::vector<bool>(GetNumIntersections(), false);

    for (int y = 0; y < bsize; ++y) {
        for (int x = 0; x < bsize; ++x) {
            auto index = GetIndex(x, y);
            auto vertex = GetVertex(x, y);
            if (pass_alive_groups[vertex] > 0) {
                result[index] = true;
            } else if (eyes[vertex]) {
                result[index] = true;
            }
        }
    }

    return result;
}

int Board::ClassifyGroups(std::vector<int> &features, std::vector<int> &groups, int target) const {
    if (groups.size() != (size_t)GetNumVertices()) {
        groups.resize(GetNumVertices());
    }

    for (int vtx = 0; vtx < GetNumVertices(); ++vtx) {
        groups[vtx] = -1;
    }

    auto bsize = GetBoardSize();
    for (int y = 0; y < bsize; ++y) {
        for (int x = 0; x < bsize; ++x) {
            auto vtx = GetVertex(x, y);
            groups[vtx] = 0;
        }
    }

    auto marked = std::vector<bool>(GetNumVertices(), false);
    auto groups_index = 1;
    for (int y = 0; y < bsize; ++y) {
        for (int x = 0; x < bsize; ++x) {
            auto vtx = GetVertex(x, y);

            if (!marked[vtx] && features[vtx] == target) {
                auto buf = std::vector<bool>(GetNumVertices(), false);

                ComputeReachGroup(vtx, target, buf, [&](int v){ return features[v]; });

                auto VertexGroups = GatherVertex(buf);
                for (const auto v : VertexGroups) {
                    marked[v] = true;
                    groups[v] = groups_index;
                }
                groups_index += 1;
            }
        }
    }
    return groups_index - 1;
}

std::vector<int> Board::FindStringSurround(std::vector<int> &groups, int index) const {
    auto result = std::vector<int>{};

    for (auto vtx = size_t{0}; vtx < groups.size(); ++vtx) {
        if (groups[vtx] == index) {
            for (int k = 0; k < 4; ++k) {
                auto avtx = directions_[k] + vtx;
                auto it = std::find(std::begin(result), std::end(result), avtx);

                if (groups[avtx] != index && it == std::end(result)) {
                    result.emplace_back(avtx);
                }
            }
        }
    }

    return result;
}

std::vector<int> Board::GatherVertex(std::vector<bool> &buf) const {
    auto result = std::vector<int>{};

    for (auto vtx = size_t{0}; vtx < buf.size(); ++vtx) {
        if (buf[vtx]) {
            result.emplace_back(vtx);
        }
    }

    return result;
}
