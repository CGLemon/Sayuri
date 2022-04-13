#include "game/simple_board.h"

Pattern SimpleBoard::GetPattern3x3(const int vtx, const int color) const {
    const int size = letter_box_size_;

    /*
     color order

     1 2 3
     4 . 5
     6 7 8

     each color takes 2 bits, totally take 16 bits


     liberties order
       1
     2 . 3
       4
     each liberties takes 2 bits, totally take 8 bits

     */

    int color_invert[4] = {kWhite, kBlack, kEmpty, kInvalid};

    // colors
    int temp0 = state_[vtx - size - 1];
    int temp1 = state_[vtx - size];
    int temp2 = state_[vtx - size + 1];
    int temp3 = state_[vtx - 1];
    int temp4 = state_[vtx + 1];
    int temp5 = state_[vtx + size - 1];
    int temp6 = state_[vtx + size];
    int temp7 = state_[vtx + size + 1];

    // liberties
    int temp8  = std::min(strings_.GetLiberty(strings_.GetParent(vtx - size)), 3);
    int temp9  = std::min(strings_.GetLiberty(strings_.GetParent(vtx - 1))   , 3);
    int temp10 = std::min(strings_.GetLiberty(strings_.GetParent(vtx + 1))   , 3);
    int temp11 = std::min(strings_.GetLiberty(strings_.GetParent(vtx + size)), 3);

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
                  (temp5 <<  4) | (temp6 <<  2) | (temp7 <<  0) |

                                  (temp8 << 22) |
                  (temp9 << 20) |                 (temp10<< 18) |
                                  (temp11<< 16);
              

    hash_arr[1] = (temp5 << 14) | (temp3 << 12) | (temp0 << 10) |
                  (temp6 <<  8) |                 (temp1 <<  6) |
                  (temp7 <<  4) | (temp4 <<  2) | (temp2 <<  0) |

                                  (temp9 << 22) |
                  (temp11<< 20) |                 (temp8 << 18) |
                                  (temp10<< 16);


    hash_arr[2] = (temp7 << 14) | (temp6 << 12) | (temp5 << 10) |
                  (temp4 <<  8) |                 (temp3 <<  6) | 
                  (temp2 <<  4) | (temp1 <<  2) | (temp0 <<  0) |

                                  (temp11<< 22) |
                  (temp10<< 20) |                 (temp9 << 18) |
                                  (temp8 << 16);

    hash_arr[3] = (temp2 << 14) | (temp4 << 12) | (temp7 << 10) |
                  (temp1 <<  8) |                 (temp6 <<  6) |
                  (temp0 <<  4) | (temp3 <<  2) | (temp5 <<  0) |

                                  (temp10<< 22) |
                  (temp8 << 20) |                 (temp11<< 18) |
                                  (temp9 << 16);


    hash_arr[4] = (temp0 << 14) | (temp3 << 12) | (temp5 << 10) |
                  (temp1 <<  8) |                 (temp6 <<  6) |
                  (temp2 <<  4) | (temp4 <<  2) | (temp7 <<  0) |

                                  (temp9 << 22) |
                  (temp8 << 20) |                 (temp11<< 18) |
                                  (temp10<< 16);


    hash_arr[5] = (temp2 << 14) | (temp1 << 12) | (temp0 << 10) |
                  (temp4 <<  8) |                 (temp3 <<  6) |
                  (temp7 <<  4) | (temp6 <<  2) | (temp5 <<  0) |

                                  (temp8 << 22) |
                  (temp10<< 20) |                 (temp9 << 18) |
                                  (temp11<< 16);
 
    hash_arr[6] = (temp7 << 14) | (temp4 << 12) | (temp2 << 10) |
                  (temp6 <<  8) |                 (temp1 <<  6) | 
                  (temp5 <<  4) | (temp3 <<  2) | (temp0 <<  0) |

                                  (temp10<< 22) |
                  (temp11<< 20) |                 (temp8 << 18) |
                                  (temp9 << 16);


    hash_arr[7] = (temp5 << 14) | (temp6 << 12) | (temp7 << 10) |
                  (temp3 <<  8) |                 (temp4 <<  6) |
                  (temp0 <<  4) | (temp1 <<  2) | (temp2 <<  0) |

                                  (temp11<< 22) |
                  (temp9 << 20) |                 (temp10<< 18) |
                                  (temp8 << 16);

    return Pattern::GetSpatial3x3(*std::min_element(hash_arr, hash_arr+8));
}
