#include <algorithm>
#include <vector>
#include <queue>
#include <string>
#include <sstream>
#include <cassert>

#include "game/board.h"
#include "pattern/pattern.h"

#define PTH_VMIRROR	1
#define PTH_HMIRROR	2
#define PTH_180ROT	4

using Pattern3Str = std::array<std::string, 3>;
bool kPattern3Matched[65535];

const std::vector<Pattern3Str> kPattern3Src =
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

constexpr std::uint16_t ComputePattern3HashFromList(int *buf) {
    const std::uint16_t hash =
        buf[0] << 0  | buf[1] << 2  | buf[2] << 4 |
        buf[3] << 6  /* raw[4] */   | buf[5] << 8 |
        buf[6] << 10 | buf[7] << 12 | buf[8] << 14
    ;
    return hash;
}

void Board::InitPattern3() {
    std::fill(std::begin(kPattern3Matched), std::end(kPattern3Matched), false);

    auto Pattern3Replacement = [](Pattern3Str pat3, int y, int x, std::vector<char> list) {
        auto out = std::vector<Pattern3Str>{};
        for (auto v : list) {
            Pattern3Str rpat3 = pat3;
            rpat3[y][x] = v;
            out.emplace_back(rpat3);
        }
        return out;
    };

    auto replace_pat3_buf = std::vector<Pattern3Str>{};
    for (auto &pat3 : kPattern3Src) {
        auto pat3_raw_que = std::queue<Pattern3Str>{};
        pat3_raw_que.emplace(pat3);

        while (!pat3_raw_que.empty()) {
            Pattern3Str pat3_raw = pat3_raw_que.front();
            pat3_raw_que.pop();

            bool success = true;
            for (int y=0; y<3; ++y) {
                for (int x=0; x<3; ++x) {
                    char m = pat3_raw[y][x];
                    auto extension = std::vector<Pattern3Str>{};

                    switch (m) {
                        case 'x': extension = Pattern3Replacement(pat3_raw, y, x, {'O', '.'});           break;
                        case 'o': extension = Pattern3Replacement(pat3_raw, y, x, {'X', '.'});           break;
                        case '?': extension = Pattern3Replacement(pat3_raw, y, x, {'X', 'O', '#', '.'}); break;
                        default : ;
                    }
                    if (!extension.empty()) {
                        for (auto ext_pat3 : extension) {
                            pat3_raw_que.emplace(ext_pat3);
                        }
                        success = false;
                        x = y = 999; // end the loop...
                    }
                }
            }
            if (success) {
                replace_pat3_buf.emplace_back(pat3_raw);
            }
        }
    }

    auto invert_pat3_buf = std::vector<Pattern3Str>{};
    for (auto &pat3 : replace_pat3_buf) {
        Pattern3Str invert_pat3 = pat3;
        bool is_invert = false;
        for (int y=0; y<3; ++y) {
            for (int x=0; x<3; ++x) {
                if (invert_pat3[y][x] == 'X') {
                    invert_pat3[y][x] = 'O';
                    is_invert = true;
                } else if (invert_pat3[y][x] == 'O') {
                    invert_pat3[y][x] = 'X';
                    is_invert = true;
                }
            }
        }
        invert_pat3_buf.emplace_back(pat3);
        if (is_invert) {
            invert_pat3_buf.emplace_back(invert_pat3);
        }
    }

    auto symm_pat3_buf = std::vector<Pattern3Str>{};
    for (auto &pat3 : invert_pat3_buf) {
	    for (int r=0; r<8; ++r) {
            Pattern3Str symm_pat3 = pat3;

            for (int y=0; y<3; ++y) {
                for (int x=0; x<3; ++x) {
			        int rx = x;
			        int ry = y;

	                if (r & PTH_180ROT) std::swap(rx, ry);
			        if (r & PTH_HMIRROR) rx = 2-rx;
			        if (r & PTH_VMIRROR) ry = 2-ry;

                    symm_pat3[ry][rx] = pat3[y][x];
                }
            }
            symm_pat3_buf.emplace_back(symm_pat3);
        }
    }

    for (auto &pat3 : symm_pat3_buf) {
        auto list = std::vector<int>{};
        for (int y=0; y<3; ++y) {
            for (int x=0; x<3; ++x) {
                char m = pat3[y][x];
                switch (m) {
                    case 'X': list.emplace_back(kBlack); break;
                    case 'O': list.emplace_back(kWhite); break;
                    case '.': list.emplace_back(kEmpty); break;
                    case '#': list.emplace_back(kInvalid); break;
                    default : throw "unknown Patterns...";
                }
            }
        }

        kPattern3Matched[ComputePattern3HashFromList(list.data())] = true;
    }
}

bool Board::MatchPattern3(const int vtx) const {
#ifdef USE_DIRTY_HARD_CODED
    int color = to_move_;
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

    constexpr int kColorMap[2][4] = {
        {kBlack, kWhite, kEmpty, kInvalid},
        {kWhite, kBlack, kEmpty, kInvalid}
    };

    for (int y=0; y<3; ++y) {
        for (int x=0; x<3; ++x) {
            raw[y][x] = kColorMap[color][raw[y][x]];
        }
    }

    int symm_p[3][3];
	for (int r = 0; r < 8; ++r) {
        for (int y=0; y<3; ++y) {
            for (int x=0; x<3; ++x) {
			    int rx = x;
			    int ry = y;

			    if (r & PTH_180ROT) {
				    std::swap(rx, ry);
			    }
			    if (r & PTH_HMIRROR) rx = 2-rx;
			    if (r & PTH_VMIRROR) ry = 2-ry;

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
                        case 'X': if (s != kBlack)                { match = false; } break;
                        case 'O': if (s != kWhite)                { match = false; } break;
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

    return false;

#else
    return kPattern3Matched[GetPattern3Hash(vtx)];
#endif
}

std::uint16_t Board::GetPattern3Hash(const int vtx) const {
    int size = letter_box_size_;
    int buf[9];

    buf[0] = state_[vtx+size-1];
    buf[1] = state_[vtx+size];
    buf[2] = state_[vtx+size+1];
    buf[3] = state_[vtx-1];
    buf[4] = state_[vtx];
    buf[5] = state_[vtx+1];
    buf[6] = state_[vtx-size-1];
    buf[7] = state_[vtx-size];
    buf[8] = state_[vtx-size+1];

    return ComputePattern3HashFromList(buf);
}

std::uint16_t Board::GetSymmetryPattern3Hash(const int vtx,
                                             const int color,
                                             const int symmetry) const {
    int size = letter_box_size_;
    int buf[9];

    buf[0] = state_[vtx+size-1];
    buf[1] = state_[vtx+size];
    buf[2] = state_[vtx+size+1];
    buf[3] = state_[vtx-1];
    buf[4] = state_[vtx];
    buf[5] = state_[vtx+1];
    buf[6] = state_[vtx-size-1];
    buf[7] = state_[vtx-size];
    buf[8] = state_[vtx-size+1];

    constexpr int kColorMap[2][4] = {
        {kBlack, kWhite, kEmpty, kInvalid},
        {kWhite, kBlack, kEmpty, kInvalid}
    };

    for (int i=0; i<9; ++i) {
        buf[i] = kColorMap[color][buf[i]];
    }

    int symm_buf[9];

    for (int y=0; y<3; ++y) {
        for (int x=0; x<3; ++x) {
	        int rx = x;
	        int ry = y;

	        if (symmetry & PTH_180ROT) std::swap(rx, ry);
	        if (symmetry & PTH_HMIRROR) rx = 2-rx;
	        if (symmetry & PTH_VMIRROR) ry = 2-ry;

            symm_buf[3 * ry + rx] = buf[3 * y + x];
        }
    }

    return ComputePattern3HashFromList(symm_buf);
}

std::string Board::GetPatternSpat(const int vtx, const int color, const int dist) const {
    auto out = std::ostringstream{};

    const int cx = GetX(vtx);
    const int cy = GetY(vtx);

    out << '.'; // center

    for (int i = kPointIndex[2]; i < kPointIndex[dist + 1]; ++i) {
        const int px = cx + kPointCoords[i].x;
        const int py = cy + kPointCoords[i].y;

        if (px >= board_size_ ||
                py >= board_size_ ||
                px < 0 || py < 0) {
            out << '#'; // invalid
        } else {
            const int state = state_[GetVertex(px,py)];
            if (state == kEmpty) {
                out << '.';
            } else if (color == state) {
                // my color
                out << 'X';
            } else if (color != state) {
                // opp color
                out << 'O';
            }
        }
    }

    return out.str();
}

std::uint64_t Board::GetPatternHash(const int vtx, const int color, const int dist) const {
    std::uint64_t hash = PatternHash[0][kInvalid][0];

    constexpr int kColorMap[2][4] = {
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
        const int c = kColorMap[color][state_[pvtx]];

        hash ^= PatternHash[0][c][i];
    }
    return hash;
}

std::uint64_t Board::GetSymmetryPatternHash(const int vtx, const int color,
                                            const int dist, const int symmetry) const {
    std::uint64_t hash = PatternHash[0][kInvalid][0];

    constexpr int kColorMap[2][4] = {
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
        const int c = kColorMap[color][state_[pvtx]];

        hash ^= PatternHash[symmetry][c][i];
    }
    return hash;
}


std::uint64_t Board::GetSurroundPatternHash(std::uint64_t hash,
                                            const int vtx,
                                            const int color,
                                            const int dist) const {
    if (dist == 2) {
        hash = PatternHash[0][kInvalid][0];
    }
    constexpr int kColorMap[2][4] = {
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
        const int c = kColorMap[color][state_[pvtx]];

        hash ^= PatternHash[0][c][i];
    }
    return hash;
}

bool Board::GetBorderLevel(const int vtx, const int /* color */, std::uint64_t &hash) const {
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

    const int dist = std::min(x_dist, y_dist);
    if (dist >= 6) {
        return false;
    }

    hash = 0ULL << 32 | (std::uint64_t)dist;

    return true;
}

bool Board::GetDistLevel(const int vtx, const int /* color */, std::uint64_t &hash) const {
    if (vtx == kPass) {
        return false;
    }

    int dist;

    if (last_move_ == kNullVertex) {
        dist = 0;
    } else if (last_move_ == kPass) {
        dist = 1;
    } else {
        const int dx = std::abs(GetX(last_move_) - GetX(vtx));
        const int dy = std::abs(GetY(last_move_) - GetY(vtx));

        dist = dx + dy + std::max(dx, dy);
        if (dist >= 17) {
            dist = 17;
        }
    }

    hash = 1ULL << 32 | (std::uint64_t)dist;

    return true;
}

bool Board::GetDistLevel2(const int vtx, const int /* color */, std::uint64_t &hash) const {
    if (vtx == kPass) {
        return false;
    }

    int dist;

    if (last_move_2_ == kNullVertex) {
        dist = 0;
    } else if (last_move_2_ == kPass) {
        dist = 1;
    } else {
        const int dx = std::abs(GetX(last_move_2_) - GetX(vtx));
        const int dy = std::abs(GetY(last_move_2_) - GetY(vtx));

        dist = dx + dy + std::max(dx, dy);
        if (dist >= 17) {
            dist = 17;
        }
    }

    hash = 2ULL << 32 | (std::uint64_t)dist;

    return true;
}

bool Board::GetCapureLevel(const int vtx, const int color, std::uint64_t &hash) const {
    if (vtx == kPass) {
        return false;
    }
    if (!IsCaptureMove(vtx, color)) {
        return false;
    }

    int opp_color = !color;
    int num_cap_stones = 0;
    auto pset = std::vector<int>{};
    bool is_ladder = false;

    for (int k = 0; k < 4; ++k) {
        const auto avtx = vtx + directions_[k];

        if (state_[avtx] == opp_color && GetLiberties(avtx) == 1) {
            const auto parent = strings_.GetParent(avtx);

            if (std::end(pset) ==
                    std::find(std::begin(pset), std::end(pset), parent)) {
                pset.emplace_back(parent);
                num_cap_stones += GetStones(avtx);

                auto vital_moves = std::vector<int>{};
                if (IsLadder(avtx, vital_moves)) {
                    if (std::end(vital_moves) !=
                            std::find(std::begin(vital_moves),
                                          std::end(vital_moves), vtx)) {
                        // The vtx is in the vital_moves.
                        is_ladder = true;
                    }
                }
            }
        }
    }

    int level = 0;
    assert(num_cap_stones >= 1);

    if (IsAtariMove(vtx, color)) {
        // string contiguous to new string in atari
        level = 1;
    } else if (is_ladder) {
        // string in a ladder
        level = 2;
    } else {
        // string not in a ladder
        if (num_cap_stones <= 6) {
            level = 3;
        } else {
            level = 4;
        }
    }

    hash = 3ULL << 32 | (std::uint64_t)level;
    return true;
}

bool Board::GetAtariLevel(const int vtx, const int color, std::uint64_t &hash) const {
    if (vtx == kPass) {
        return false;
    }
    if (!IsAtariMove(vtx, color)) {
        return false;
    }

    int opp_color = !color;
    int num_atari_stones = 0;
    bool is_ladder = false;

    auto pset = std::vector<int>{};

    for (int k = 0; k < 4; ++k) {
        const auto avtx = vtx + directions_[k];

        if (state_[avtx] == opp_color && GetLiberties(avtx) == 2) {
            const auto parent = strings_.GetParent(avtx);

            if (std::end(pset) ==
                    std::find(std::begin(pset),
                                  std::end(pset), parent)) {
                pset.emplace_back(parent);
                num_atari_stones += GetStones(avtx);

                auto vital_moves = std::vector<int>{};
                if (IsLadder(avtx, vital_moves)) {
                    if (std::end(vital_moves) !=
                            std::find(std::begin(vital_moves),
                                          std::end(vital_moves), vtx)) {
                        // The vtx is in the vital_moves.
                        is_ladder = true;
                    }
                }
            }
        }

    }

    int level = 0;
    assert(num_atari_stones >= 1);
#ifdef NDEBUG
    (void) num_atari_stones;
#endif

    if (ko_move_ != kNullVertex) {
        // atari when there is a ko
        level = 1;
    } else if (is_ladder) {
        // ladder atari
        level = 2;
    } else {
        if (pset.size() == 1) {
            // simple atari
            level = 3;
        } else {
            // double atari
            level = 4;
        }
    }

    hash = 4ULL << 32 | (std::uint64_t)level;
    return true;
}

bool Board::GetSelfAtariLevel(const int vtx, const int color, std::uint64_t &hash) const {
    if (vtx == kPass) {
        return false;
    }
    if (!IsSelfAtariMove(vtx, color)) {
        return false;
    }

    int level = 1;
    hash = 5ULL << 32 | (std::uint64_t)level;

    return true;
}

bool Board::GetFeatureWrapper(const int f, const int vtx,
                              const int color, std::uint64_t &hash) const {
    switch (f) {
        case 0: return GetBorderLevel(vtx, color, hash);
        case 1: return GetDistLevel(vtx, color, hash);
        case 2: return GetDistLevel2(vtx, color, hash);
        case 3: return GetCapureLevel(vtx, color, hash);
        case 4: return GetAtariLevel(vtx, color, hash);
        case 5: return GetSelfAtariLevel(vtx, color, hash);
    }
    return false;
}

int Board::GetMaxFeatures() {
    return 6;
}

#undef PTH_VMIRROR
#undef PTH_HMIRROR
#undef PTH_180ROT
