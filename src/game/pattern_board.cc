#include <algorithm>
#include <vector>
#include <string>

#include "game/simple_board.h"
#include "pattern/pattern.h"

const
std::vector<
    std::vector<
        std::string
   >
>
kPattern3Src =
{  // 3x3 playout patterns;
   // X,O are colors, x,o are their inverses,
   // # is off boarder, . is empty, ? is any color

    {"XOX",  // hane pattern - enclosing hane
     "...",
     "???"},

    {"XO.",  // hane pattern - non-cutting hane
     "...",
     "?.?"},

    {"XO?",  // hane pattern - magari
     "X..",
     "x.?"},

    {".O.",  // generic pattern - katatsuke or diagonal attachment; similar to magari
     "X..",
     "..."},

    {"XO?",  // cut1 pattern (kiri] - unprotected cut
     "O.o",
     "?o?"},

    {"XO?",  // cut1 pattern (kiri] - peeped cut
     "O.X",
     "???"},

    {"?X?",  // cut2 pattern (de]
     "O.O",
     "ooo"},

    {"OX?",  // cut keima
     "o.O",
     "???"},

    {"X.?",  // side pattern - chase
     "O.?",
     "###"},

    {"OX?",  // side pattern - block side cut
     "X.O",
     "###"},

    {"?X?",  // side pattern - block side connection
     "x.O",
     "###"},

    {"?XO",  // side pattern - sagari
     "x.x",
     "###"},

    {"?OX",  // side pattern - cut
     "X.O",
     "###"},
};

//Hard-coded patterns, a bit nasty.
bool SimpleBoard::MatchPattern3(const int vtx, const int color) const {
    int size = letter_box_size_;
    int raw[3][3];

    raw[0][0] = state_[vtx+size-1];
    raw[0][1] = state_[vtx+size];
    raw[0][2] = state_[vtx+size+1];

    raw[1][0] = state_[vtx-1];
    raw[1][1] = state_[vtx];
    raw[1][2] = state_[vtx+1];

    raw[2][0] = state_[vtx-size-1];
    raw[2][1] = state_[vtx-size];
    raw[2][2] = state_[vtx-size+1];

    constexpr int color_map[2][4] = {
        {kBlack, kWhite, kEmpty, kInvalid},
        {kWhite, kBlack, kEmpty, kInvalid}
    };

    for (int y=0; y<3; ++y) {
        for (int x=0; x<3; ++x) {
            raw[y][x] = color_map[color][raw[y][x]];
        }
    }

    int symm_p[3][3];

#define PTH_VMIRROR	1
#define PTH_HMIRROR	2
#define PTH_90ROT	4
	for (int r = 0; r < 8; ++r) {
        for (int y=0; y<3; ++y) {
            for (int x=0; x<3; ++x) {
			    int rx = x;
			    int ry = y;

			    if (r & PTH_VMIRROR) ry = -ry;
			    if (r & PTH_HMIRROR) rx = -rx;
			    if (r & PTH_90ROT) {
				    int rs = rx; rx = -ry; ry = rs;
			    }
                symm_p[ry][rx] = raw[y][x];
            }
        }
        for (auto &pat3 : kPattern3Src) {
            bool match = true;
            for (int y=0; y<3; ++y) {
                for (int x=0; x<3; ++x) {
                    int s = symm_p[y][x];
                    char m = pat3[y][x];

                    switch (m) {
                        case 'X': if (s != kInvalid)              { match = false; } break;
                        case 'O': if (s != kBlack)                { match = false; } break;
                        case 'x': if (s != kWhite && s != kEmpty) { match = false; } break;
                        case 'o': if (s != kBlack && s != kEmpty) { match = false; } break;
                        case '.': if (s != kEmpty)                { match = false; } break;
                        case '#': if (s != kInvalid)              { match = false; } break;
                        case '?': ;
                        default : ;
                    }
                    if (!match) break;
                }
                if (!match) break;
            }
            if (match) return true;
        }
    }
#undef PTH_VMIRROR
#undef PTH_HMIRROR
#undef PTH_90ROT

    return false;
}

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
