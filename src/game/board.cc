#include "game/board.h"

#include <algorithm>
#include <set>

int Board::ComputeScoreOnBoard(int black_bonus) const {
    const auto black = ComputeReachColor(kBlack);
    const auto white = ComputeReachColor(kWhite);
    return black - white + black_bonus;
}

float Board::ComputeSimpleFinalScore(float komi) const {
    return static_cast<float>(ComputeScoreOnBoard(0)) - komi;
}

void Board::ComputeSimpleOwnership(std::vector<int> &result) const {
    if (result.size() != (size_t) num_intersections_) {
        result.resize(num_intersections_);
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
                result[idx] = kBlack;  
            } else if (white[vtx] && !black[vtx]) {
                result[idx] = kWhite;
            } else {
                //The point belongs to both.
                result[idx] = kEmpty;
            }
       }
    }
}

void Board::ComputePassAliveOwnership(std::vector<int> &result) const {

    ComputeSimpleOwnership(result);
    auto pass_alive = std::vector<bool>(num_intersections_);

    for (int c = 0; c < 2; ++c) {

        std::fill(std::begin(pass_alive), std::end(pass_alive), false);
        ComputePassAlive(pass_alive, c, true, true);

        for (int i = 0; i < num_intersections_; ++i) {
            if (pass_alive[i]) {
                result[i] = c;
            }
        }
    }
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

std::vector<LadderType> Board::GetLadderMap() const {
    auto result = std::vector<LadderType>(GetNumIntersections(), LadderType::kNotLadder);
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
                result[idx] = LadderType::kLadderDeath;
            } else {
                // The ladder string has a chance to escape.
                result[idx] = LadderType::kLadderEscapable;
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
                        result[aidx] = LadderType::kLadderTake;
                    } else {
                        // Someone can atari this ladder string.
                        result[aidx] = LadderType::kLadderAtari;
                    }
                }
            }
        }
    }

    return result;
}

void Board::ComputeSekiPoints(std::vector<bool> &result) const {
    for (int y = 0; y < board_size_; ++y) {
        for (int x = 0; x < board_size_; ++x) {
            const auto idx = GetIndex(x, y);
            const auto vtx = GetVertex(x, y);

            if (IsSeki(vtx)) {
                result[idx] = true;
            }
        }
    }
}

void Board::ComputeSafeArea(std::vector<bool> &result, bool mark_seki) const {
    if (result.size() != (size_t) num_intersections_) {
        result.resize(num_intersections_);
    }

    std::fill(std::begin(result), std::end(result), false);

    ComputePassAlive(result, kBlack, true, true);
    ComputePassAlive(result, kWhite, true, true);
    if (mark_seki) {
        ComputeSekiPoints(result);
    }
}

void Board::ComputePassAlive(std::vector<bool> &result,
                                 const int color,
                                 bool mark_vitals,
                                 bool mark_pass_dead) const {
    const auto num_vertices = GetNumVertices();
    const auto bsize = GetBoardSize();
    auto ocupied = std::vector<int>(num_vertices, kInvalid);

    // Mark the color.
    for (int y = 0; y < bsize; ++y) {
        for (int x = 0; x < bsize; ++x) {
            const auto vtx = GetVertex(x, y);
            const auto state = GetState(vtx);
            if (state == color) {
                ocupied[vtx] = color;
            } else {
                ocupied[vtx] = kEmpty;
            }
        }
    }

    auto regions_index = std::vector<int>(num_vertices, -1);
    auto regions_next = std::vector<int>(num_vertices, kNullVertex);
    auto regions_head = ClassifyGroups(kEmpty, ocupied, regions_index, regions_next);
    auto vitals = std::vector<bool>(num_vertices, false);

    // TODO: Do we need to think about sucide move?
    constexpr bool allow_sucide = false;

    // Compute the potential vital regions.
    // TODO: Do we need to compute potential vital regions here?
    for (int vtx : regions_head) {
        bool success = true;
        int pos = vtx;
        do {
            bool is_vital = false;
            int state = allow_sucide == true ? ocupied[pos] : GetState(pos);

            assert(state != color);

            if (state == kEmpty) { 
                for (int k = 0; k < 4; ++k) {
                    const auto apos = directions_[k] + pos;
                    if (ocupied[apos] == color) {
                        is_vital = true;
                        break;
                    }
                }
            } else if (state == !color) {
                is_vital = true;
            }

            if (!is_vital) {
                success = false;
                break;
            }
            pos = regions_next[pos];
        } while(pos != vtx);


        if (success) {
            int pos = vtx;
            do {
                vitals[pos] = true;
                pos = regions_next[pos];
            } while(pos != vtx);
        }
    }

    auto strings_index = std::vector<int>(num_vertices, -1);
    auto strings_next = std::vector<int>(num_vertices, kNullVertex);
    auto strings_head = ClassifyGroups(color, ocupied, strings_index, strings_next);

    int group_cnt = strings_head.size();

    // Start the Benson's algorithm.
    // https://senseis.xmp.net/?BensonsAlgorithm
    while(true) {
        auto change = false;

        for (int i = 0; i < group_cnt; ++i) {
            const auto vtx = strings_head[i];

            if (!IsPassAliveString(vtx, allow_sucide, vitals, ocupied,
                                       regions_index, regions_next, strings_index, strings_next)) {
                // The string is not pass alive. Remove the uncertainty life string.

                int pos = vtx;
                do {
                    strings_index[pos] = 0;
                    ocupied[pos] = kEmpty;
                    pos = strings_next[pos];

                    for (int k = 0; k < 4; ++k) {
                        const auto apos = directions_[k] + pos;
                        if (vitals[apos]) {
                            int rpos = apos;
                            do {
                                vitals[rpos] = false;
                                rpos = regions_next[rpos];
                            } while(rpos != apos);
                        }
                    }
                } while(pos != vtx);

                // Remove the linking.
                std::remove(std::begin(strings_head),
                                std::end(strings_head), vtx);
                group_cnt -= 1;
                change = true;
                break;
            }
        }

        // The algorithm is over if there is no removed string.
        if (!change) break;
    }

    // Fill the pass alive groups.
    for (int i = 0; i < group_cnt; ++i) {
        const auto vtx = strings_head[i];
        int pos = vtx;
        do {
            auto x = GetX(pos);
            auto y = GetY(pos);
            auto index = GetIndex(x, y);
            result[index] = true;
            pos = strings_next[pos];
        } while(pos != vtx);
    }

    // Fill the pass alive vitals.
    if (mark_vitals) {
        for (int vtx : regions_head) {
            int pos = vtx;
            do {
                if (vitals[pos]) {
                    auto x = GetX(pos);
                    auto y = GetY(pos);
                    auto index = GetIndex(x, y);
                    result[index] = true;
                    ocupied[pos] = color;
                }
                pos = regions_next[pos];
            } while(pos != vtx);
        }
    }

    if (mark_pass_dead) {
        // Re-compute the regions for scanning pass-dead regions.
        regions_head = ClassifyGroups(kEmpty, ocupied, regions_index, regions_next);

        // Fill the pass dead regions.
        for (int vtx : regions_head) {
            int pos = vtx;
            bool pass_dead = IsPassDeadRegion(pos, !color, ocupied, regions_next);
            do {
                if (pass_dead) {
                    auto x = GetX(pos);
                    auto y = GetY(pos);
                    auto index = GetIndex(x, y);
                    result[index] = true;
                }
                pos = regions_next[pos];
            } while(pos != vtx);
        }
    }
}

bool Board::IsPassAliveString(const int vtx,
                                  bool allow_sucide,
                                  const std::vector<bool> &vitals,
                                  const std::vector<int> &features,
                                  const std::vector<int> &regions_index,
                                  const std::vector<int> &regions_next,
                                  const std::vector<int> &strings_index,
                                  const std::vector<int> &strings_next) const {
    auto vitals_list = std::set<int>{};
    int my_index = strings_index[vtx];
    int pos = vtx;
    do {
        for (int k = 0; k < 4; ++k) {
            const auto apos = directions_[k] + pos;
            if (vitals[apos]) {
                // This region is potential vital region. Try to search it.
                int rpos = apos;
                bool success = true;
                do {
                    bool is_adjacent = false;
                    int state = allow_sucide == true ? features[rpos] : GetState(rpos);
                    if (state == kEmpty) {
                    for (int k = 0; k < 4; ++k) {
                            // Check points of adjacent empty.
                            const auto aapos = directions_[k] + rpos;                         
                            if(strings_index[aapos] == my_index) {
                                is_adjacent = true;
                                break;
                            }
                        }
                    } else {
                        is_adjacent = true;
                    }
                    if (!is_adjacent) {
                        // Not every empty points are adjacent to my string. The region
                        // is not vital.
                        success = false;
                        break;
                    }
                    rpos = regions_next[rpos];
                } while(rpos != apos);

                if (success) vitals_list.insert(regions_index[apos]);
            }
        }
        pos = strings_next[pos];
    } while(pos != vtx);

    return vitals_list.size() >= 2;
}

bool Board::IsPassDeadRegion(const int vtx,
                                 const int color,
                                 std::vector<int> &features,
                                 const std::vector<int> &regions_next) const {
    const auto IsPotentialEye = [this](const int vertex,
                                           const int color,
                                           std::vector<int> &features,
                                           std::vector<bool> &inner_regions) {
        // The potential eye is possible to become the real eye in the region.
        std::array<int, 4> side_count = {0, 0, 0, 0};

        for (int k = 0; k < 4; ++k) {
            const auto avtx = vertex + directions_[k];
            side_count[features[avtx]]++;
        }

        if (side_count[!color] != 0) {
            return false;
        }

        std::array<int, 4> corner_count = {0, 0, 0, 0};

        for (int k = 4; k < 8; ++k) {
            const auto avtx = vertex + directions_[k];
            if (inner_regions[avtx]) { // The inner region corner is mine.
                corner_count[color]++;
            } else {
                corner_count[features[avtx]]++;
            }
        }
        if (corner_count[kInvalid] == 0) {
            if (corner_count[!color] > 1) {
                return false;
            }
        } else {
            if (corner_count[!color] > 0) {
                return false;
            }
        }
        return true;
    };

    // The inner regions is pass-alive string surrondded by the region which
    // we want to search.
    auto inner_regions = std::vector<bool>(features.size(), false);

    // The inner regions may cause the false-eye life. The false will
    // become the potential eye in this condition.
    ComputeInnerRegions(vtx, color, regions_next, inner_regions);

    auto potentia_eyes = std::vector<int>{};
    int pos = vtx;
    do {
        // Search all potential eyes in this region.
        if (IsPotentialEye(pos, color, features, inner_regions)) {
            potentia_eyes.emplace_back(pos);
        }
        pos = regions_next[pos];
    } while(pos != vtx);

    int eyes_count = potentia_eyes.size();

    if (eyes_count == 2) {
        // It is possible to be pass-dead, if there are only two potential eyes. The
        // case is two eyes are adjacent to each others.
        if (IsNeighbor(potentia_eyes[0], potentia_eyes[1])) {
            eyes_count -= 1;
        }
    }

    return eyes_count < 2;
}

void Board::ComputeInnerRegions(const int vtx,
                                    const int color,
                                    const std::vector<int> &regions_next,
                                    std::vector<bool> &inner_regions) const {
    const auto num_vertices = GetNumVertices();
    const auto bsize = GetBoardSize();
    auto surround = std::vector<int>(num_vertices, kInvalid);

    std::fill(std::begin(inner_regions), std::end(inner_regions), false);

    for (int y = 0; y < bsize; ++y) {
        for (int x = 0; x < bsize; ++x) {
            surround[GetVertex(x, y)] = kEmpty;
        }
    }

    int pos = vtx;
    do {
        surround[pos] = !color;
        pos = regions_next[pos];
    } while(pos != vtx);

    auto epmty_index = std::vector<int>(num_vertices, -1);
    auto epmty_next = std::vector<int>(num_vertices, kNullVertex);
    auto epmty_head = ClassifyGroups(kEmpty, surround, epmty_index, epmty_next);

    int cnt = epmty_head.size();
    for (int i = 0 ; i < cnt; ++i) {
        int v = epmty_head[i];
        pos = v;
        do {
            bool success = false;
            for (int k = 0; k < 4; ++k) {
                const auto apos = pos + directions_[k];
                if (surround[apos] == kInvalid) {
                    success = true;
                    break;
                }
            }
            if (success) {
                std::remove(std::begin(epmty_head),
                                std::end(epmty_head), v);
                cnt -= 1;
                break;
            }
            pos = epmty_next[pos];
        } while(pos != v);
    }

    for (int i = 0 ; i < cnt; ++i) {
        int v = epmty_head[i];
        pos = v;
        do {
            inner_regions[pos] = true;
            pos = epmty_next[pos];
        } while(pos != v);
    }
}

std::vector<int> Board::ClassifyGroups(const int target,
                                           std::vector<int> &features,
                                           std::vector<int> &regions_index,
                                           std::vector<int> &regions_next) const {
    const auto num_vertices = GetNumVertices();
    auto bsize = GetBoardSize();

    std::fill(std::begin(regions_index), std::end(regions_index), -1);
    std::fill(std::begin(regions_next), std::end(regions_next), kNullVertex);

    for (int y = 0; y < bsize; ++y) {
        for (int x = 0; x < bsize; ++x) {
            const auto vtx = GetVertex(x, y);
            regions_index[vtx] = 0;
            regions_next[vtx] = vtx;
        }
    }

    auto head_list = std::vector<int>{};
    auto marked = std::vector<bool>(num_vertices, false);
    auto groups_index = 1;
    for (int y = 0; y < bsize; ++y) {
        for (int x = 0; x < bsize; ++x) {
            const auto vtx = GetVertex(x, y);

            if (!marked[vtx] && features[vtx] == target) {
                auto buf = std::vector<bool>(num_vertices, false);

                ComputeReachGroup(vtx, target, buf, [&](int v){ return features[v]; });

                auto vertices = GatherVertices(buf);
                auto next_vertex = kNullVertex;
                for (const auto v : vertices) {
                    regions_next[v] = next_vertex;
                    regions_index[v] = groups_index;
                    marked[v] = true;
                    next_vertex = v;
                }
                if (!vertices.empty()) {
                    regions_next[vertices[0]] = next_vertex;
                }

                groups_index += 1;
                head_list.emplace_back(vtx);
            }
        }
    }
    return head_list;
}

std::vector<int> Board::GatherVertices(std::vector<bool> &buf) const {
    auto result = std::vector<int>{};

    for (auto vtx = size_t{0}; vtx < buf.size(); ++vtx) {
        if (buf[vtx]) {
            result.emplace_back(vtx);
        }
    }

    return result;
}
