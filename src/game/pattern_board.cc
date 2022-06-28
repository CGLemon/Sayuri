#include "game/simple_board.h"
#include <algorithm>

LocPattern SimpleBoard::GetPattern3x3(const int vtx, const int color) const {
    const int size = letter_box_size_;

    /*
     color order

     1 2 3
     4 . 5
     6 7 8

     each color takes 2 bits, totally take 16 bits

     */

    int color_invert[4] = {kWhite, kBlack, kEmpty, kInvalid};

    // TODO: Maybe we can apply Zobrist hash insteading of it
    // colors
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

    std::uint32_t hash_arr[8];

    hash_arr[0] = (temp0 << 14) | (temp1 << 12) | (temp2 << 10) |
                  (temp3 <<  8) |                 (temp4 <<  6) |
                  (temp5 <<  4) | (temp6 <<  2) | (temp7 <<  0)
                ;
              

    hash_arr[1] = (temp5 << 14) | (temp3 << 12) | (temp0 << 10) |
                  (temp6 <<  8) |                 (temp1 <<  6) |
                  (temp7 <<  4) | (temp4 <<  2) | (temp2 <<  0)
                ;


    hash_arr[2] = (temp7 << 14) | (temp6 << 12) | (temp5 << 10) |
                  (temp4 <<  8) |                 (temp3 <<  6) | 
                  (temp2 <<  4) | (temp1 <<  2) | (temp0 <<  0)
                ;

    hash_arr[3] = (temp2 << 14) | (temp4 << 12) | (temp7 << 10) |
                  (temp1 <<  8) |                 (temp6 <<  6) |
                  (temp0 <<  4) | (temp3 <<  2) | (temp5 <<  0)
                ;


    hash_arr[4] = (temp0 << 14) | (temp3 << 12) | (temp5 << 10) |
                  (temp1 <<  8) |                 (temp6 <<  6) |
                  (temp2 <<  4) | (temp4 <<  2) | (temp7 <<  0)
                ;


    hash_arr[5] = (temp2 << 14) | (temp1 << 12) | (temp0 << 10) |
                  (temp4 <<  8) |                 (temp3 <<  6) |
                  (temp7 <<  4) | (temp6 <<  2) | (temp5 <<  0)
                ;
 
    hash_arr[6] = (temp7 << 14) | (temp4 << 12) | (temp2 << 10) |
                  (temp6 <<  8) |                 (temp1 <<  6) | 
                  (temp5 <<  4) | (temp3 <<  2) | (temp0 <<  0)
                ;


    hash_arr[7] = (temp5 << 14) | (temp6 << 12) | (temp7 << 10) |
                  (temp3 <<  8) |                 (temp4 <<  6) |
                  (temp0 <<  4) | (temp1 <<  2) | (temp2 <<  0)
                ;

    return LocPattern::GetSpatial3x3(*std::min_element(hash_arr, hash_arr+8));
}

LocPattern SimpleBoard::GetPatternDistBorderX(const int vtx) const {
    // distance to boarder
    const int center = board_size_/2 + board_size_%2;

    int x_dist = GetX(vtx) + 1;
    if (x_dist > center) {
        x_dist = board_size_ + 1 - x_dist;
    }

    return LocPattern::GetDistToBorder(x_dist);
}

LocPattern SimpleBoard::GetPatternDistBorderY(const int vtx) const {
    // distance to boarder
    const int center = board_size_/2 + board_size_%2;

    int y_dist = GetY(vtx) + 1;
    if (y_dist > center) {
        y_dist = board_size_ + 1 - y_dist;
    }
    return LocPattern::GetDistToBorder(y_dist);
}

LocPattern SimpleBoard::GetPatternLiberties(const int vtx) const {
    int libs = 0;
    if (state_[vtx] == kBlack || state_[vtx] == kWhite) {
        libs = GetLiberties(vtx);

        if (libs >= 6) libs = 6;
    }
    return LocPattern::GetLiberties(libs);
}

LocPattern SimpleBoard::GetPatternDistLastMove(const int vtx) const {
    if (last_move_ == kPass) {
        return LocPattern::GetDistToLastMove(100 + 100 + 100); // give it crazy big value
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
    return LocPattern{}; // invalid pattern
}

std::vector<LocPattern> SimpleBoard::GetAllPatterns(const int vtx, const int color) const {
    auto features = std::vector<LocPattern>{};

    features.emplace_back(GetPattern3x3(vtx, color));
    features.emplace_back(GetPatternDistBorderX(vtx));
    features.emplace_back(GetPatternDistBorderY(vtx));
    features.emplace_back(GetPatternDistLastMove(vtx));
    features.emplace_back(GetPatternLiberties(vtx));
    features.emplace_back(GetPatternAtari(vtx, color));

    return features;
}
