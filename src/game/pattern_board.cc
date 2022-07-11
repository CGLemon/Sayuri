#include "game/simple_board.h"
#include <algorithm>

LocPattern SimpleBoard::GetPattern3x3(const int vtx, const int color) const {
    const int size = letter_box_size_;

    /*
     color order, each color takes 2 bits, totally take 16 bits

     1 2 3
     4 . 5
     6 7 8

     libs order, each libs takes 2 bits, totally take 8 bits

       1
     2 . 3
       4 

     */

    int color_invert[4] = {kWhite, kBlack, kEmpty, kInvalid};
    int color_cnt[4] = {0,0,0,0};

    // TODO: Maybe we can apply Zobrist hash insteading of it
    //       colors
    int center = state_[vtx];
    int temp0 = state_[vtx - size - 1];
    int temp1 = state_[vtx - size];
    int temp2 = state_[vtx - size + 1];
    int temp3 = state_[vtx - 1];
    int temp4 = state_[vtx + 1];
    int temp5 = state_[vtx + size - 1];
    int temp6 = state_[vtx + size];
    int temp7 = state_[vtx + size + 1];

    if (color == kWhite) {
        temp0 = color_invert[temp0];
        temp1 = color_invert[temp1];
        temp2 = color_invert[temp2];
        temp3 = color_invert[temp3];
        temp4 = color_invert[temp4];
        temp5 = color_invert[temp5];
        temp6 = color_invert[temp6];
        temp7 = color_invert[temp7];
    }

    color_cnt[temp0]++;
    color_cnt[temp1]++;
    color_cnt[temp2]++;
    color_cnt[temp3]++;
    color_cnt[temp4]++;
    color_cnt[temp5]++;
    color_cnt[temp6]++;
    color_cnt[temp7]++;

    if (color_cnt[kEmpty] + color_cnt[kInvalid] >= 8 || center != kEmpty) {
        return LocPattern::GetNoFeature();
    }

    int libs0 = 0, libs1 = 0, libs2 = 0, libs3 = 0; 
    if (temp1 == kBlack || temp1 == kWhite) {
        libs0 = std::min(3, GetLiberties(vtx - size));
    }
    if (temp3 == kBlack || temp3 == kWhite) {
        libs1 = std::min(3, GetLiberties(vtx - 1));
    }
    if (temp4 == kBlack || temp4 == kWhite) {
        libs2 = std::min(3, GetLiberties(vtx + 1));
    }
    if (temp6 == kBlack || temp6 == kWhite) {
        libs3 = std::min(3, GetLiberties(vtx + size));
    }

    std::uint32_t hash_arr[8];

    hash_arr[0] = (temp0 << 14) | (temp1 << 12) | (temp2 << 10) |
                  (temp3 <<  8) |                 (temp4 <<  6) |
                  (temp5 <<  4) | (temp6 <<  2) | (temp7 <<  0)

                                | (libs0 << 22) | 
                  (libs1 << 20)                 | (libs2 << 18)
                                | (libs3 << 16)
                ;
              

    hash_arr[1] = (temp5 << 14) | (temp3 << 12) | (temp0 << 10) |
                  (temp6 <<  8) |                 (temp1 <<  6) |
                  (temp7 <<  4) | (temp4 <<  2) | (temp2 <<  0)

                                | (libs1 << 22) | 
                  (libs3 << 20)                 | (libs0 << 18)
                                | (libs2 << 16)
                ;


    hash_arr[2] = (temp7 << 14) | (temp6 << 12) | (temp5 << 10) |
                  (temp4 <<  8) |                 (temp3 <<  6) | 
                  (temp2 <<  4) | (temp1 <<  2) | (temp0 <<  0)

                                | (libs3 << 22) | 
                  (libs2 << 20)                 | (libs1 << 18)
                                | (libs0 << 16)
                ;

    hash_arr[3] = (temp2 << 14) | (temp4 << 12) | (temp7 << 10) |
                  (temp1 <<  8) |                 (temp6 <<  6) |
                  (temp0 <<  4) | (temp3 <<  2) | (temp5 <<  0)

                                | (libs2 << 22) | 
                  (libs0 << 20)                 | (libs3 << 18)
                                | (libs1 << 16)
                ;


    hash_arr[4] = (temp0 << 14) | (temp3 << 12) | (temp5 << 10) |
                  (temp1 <<  8) |                 (temp6 <<  6) |
                  (temp2 <<  4) | (temp4 <<  2) | (temp7 <<  0)

                                | (libs1 << 22) | 
                  (libs0 << 20)                 | (libs3 << 18)
                                | (libs2 << 16)
                ;


    hash_arr[5] = (temp2 << 14) | (temp1 << 12) | (temp0 << 10) |
                  (temp4 <<  8) |                 (temp3 <<  6) |
                  (temp7 <<  4) | (temp6 <<  2) | (temp5 <<  0)

                                | (libs0 << 22) | 
                  (libs2 << 20)                 | (libs1 << 18)
                                | (libs3 << 16)
                ;
 
    hash_arr[6] = (temp7 << 14) | (temp4 << 12) | (temp2 << 10) |
                  (temp6 <<  8) |                 (temp1 <<  6) | 
                  (temp5 <<  4) | (temp3 <<  2) | (temp0 <<  0)

                                | (libs2 << 22) | 
                  (libs3 << 20)                 | (libs0 << 18)
                                | (libs1 << 16)
                ;


    hash_arr[7] = (temp5 << 14) | (temp6 << 12) | (temp7 << 10) |
                  (temp3 <<  8) |                 (temp4 <<  6) |
                  (temp0 <<  4) | (temp1 <<  2) | (temp2 <<  0)

                                | (libs3 << 22) | 
                  (libs1 << 20)                 | (libs2 << 18)
                                | (libs0 << 16)
                ;

    return LocPattern::GetSpatial3x3(*std::min_element(hash_arr, hash_arr+7));
}

LocPattern SimpleBoard::GetPatternDistBorder(const int vtx) const {
    // distance to boarder
    const int center = board_size_/2 + board_size_%2;

    int x_dist = GetX(vtx) + 1;
    if (x_dist > center) {
        x_dist = board_size_ + 1 - x_dist;
    }

    int y_dist = GetY(vtx) + 1;
    if (y_dist > center) {
        y_dist = board_size_ + 1 - y_dist;
    }

    int dist = std::min(x_dist, y_dist);
    if (dist >= 5) {
        return LocPattern::GetNoFeature();
    }
    return LocPattern::GetDistToBorder(dist);
}

LocPattern SimpleBoard::GetPatternDistLastMove(const int vtx) const {
    if (last_move_ == kNullVertex || last_move_ == kPass) {
        return LocPattern::GetNoFeature();
    }

    const int dx = std::abs(GetX(last_move_) - GetX(vtx));
    const int dy = std::abs(GetY(last_move_) - GetY(vtx));

    int dist = dx + dy + std::max(dx, dy);
    if (dist >= 17) {
        dist = 17;
    }
    return LocPattern::GetDistToLastMove(dist);
}

LocPattern SimpleBoard::GetPatternAtari(const int vtx, const int color) const {
    if (IsAtariMove(vtx, color)) {
        return  LocPattern::GetAtari(1);
    }
    return LocPattern::GetAtari(2);
}

std::vector<LocPattern> SimpleBoard::GetAllPatterns(const int vtx, const int color) const {
    auto patterns = std::vector<LocPattern>{};
    if (vtx == kPass || state_[vtx] != kEmpty) {
        return patterns;
    }

    patterns.emplace_back(GetPattern3x3(vtx, color));
    // patterns.emplace_back(GetPatternDistBorder(vtx));
    // patterns.emplace_back(GetPatternDistLastMove(vtx));
    // patterns.emplace_back(GetPatternAtari(vtx, color));

    return patterns;
}
