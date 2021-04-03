#include "game/board.h"

int Board::ComputeFinalScore(float komi) const {
    const auto black = ComputeReachColor(kBlack);
    const auto white = ComputeReachColor(kWhite);
    return static_cast<float>(black - white) - komi;
}

std::vector<int> Board::GetSimpleOwnership() const {
    auto res = std::vector<int>(num_intersections_, kInvalid);

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

            if ((black[vtx] && white[vtx]) || (!black[vtx] && !white[vtx])) {
                //The point is seki if there are no stones on board.
                res[idx] = kEmpty;
            } else if (black[vtx]) {
                res[idx] = kBlack;  
            } else if (white[vtx]) {
                res[idx] = kWhite;
            }
       }
    }

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
