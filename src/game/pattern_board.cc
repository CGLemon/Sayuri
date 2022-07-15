#include <algorithm>

#include "game/simple_board.h"
#include "pattern/pattern.h"

std::uint64_t SimpleBoard::GetPatternHash(const int vtx, const int color, const int dist) const {
    std::uint64_t hash = PatternHash[0][kInvalid][0];

    constexpr int color_map[2][4] = {
                    {kBlack, kWhite, kEmpty, kInvalid},
                    {kWhite, kBlack, kEmpty, kInvalid}
    };

    const int cx = GetX(vtx);
    const int cy = GetY(vtx);

    for (int i = kPointIndex[2]; i < kPointIndex[dist + 1]; ++i) {
        const int px = cx + kPointCoords[i].x;
        const int py = cy + kPointCoords[i].y;
        if (px >= board_size_ ||
                py >= board_size_ ||
                px < 0 || py < 0) {
            continue;
        }
        const int pvtx = GetVertex(px,py);
        const int c = color_map[color][state_[pvtx]];

        hash ^= PatternHash[0][c][i];
    }
    return hash;
}

std::uint64_t SimpleBoard::GetSurroundPatternHash(std::uint64_t hash,
                                                      const int vtx,
                                                      const int color,
                                                      const int dist) const {
    if (dist == 2) {
        hash = PatternHash[0][kInvalid][0];
    }
    constexpr int color_map[2][4] = {
                    {kBlack, kWhite, kEmpty, kInvalid},
                    {kWhite, kBlack, kEmpty, kInvalid}
    };
    const int cx = GetX(vtx);
    const int cy = GetY(vtx);

    for (int i = kPointIndex[dist]; i < kPointIndex[dist + 1]; ++i) {
        const int px = cx + kPointCoords[i].x;
        const int py = cy + kPointCoords[i].y;
        if (px >= board_size_ ||
                py >= board_size_ ||
                px < 0 || py < 0) {
            continue;
        }
        const int pvtx = GetVertex(px,py);
        const int c = color_map[color][state_[pvtx]];

        hash ^= PatternHash[0][c][i];
    }
    return hash;
}

bool SimpleBoard::GetBorderLevel(const int vtx, int &dist) const {
    if (vtx == kPass) {
        return false;
    }

    const int center = board_size_/2 + board_size_%2;

    int x_dist = GetX(vtx) + 1;
    if (x_dist > center) {
        x_dist = board_size_ + 1 - x_dist;
    }

    int y_dist = GetY(vtx) + 1;
    if (y_dist > center) {
        y_dist = board_size_ + 1 - y_dist;
    }

    dist = std::min(x_dist, y_dist) - 1;
    if (dist >= 5) {
        return false;
    }
    return true;
}

bool SimpleBoard::GetDistLevel(const int vtx, int &dist) const {
    if (last_move_ == kNullVertex) {
        dist = 0;
        return true;
    }

    if (last_move_ == kPass) {
        dist = 1;
        return true;
    }

    const int dx = std::abs(GetX(last_move_) - GetX(vtx));
    const int dy = std::abs(GetY(last_move_) - GetY(vtx));

    dist = dx + dy + std::max(dx, dy);
    if (dist >= 17) {
        dist = 17;
    }
    return true;
}
