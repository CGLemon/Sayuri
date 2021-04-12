#pragma once

#include <vector>

#include "game/simple_board.h"
#include "game/types.h"

class Board : public SimpleBoard {
public:
    float ComputeFinalScore(float komi) const;

    int ComputeScoreOnBoard(int black_bonus) const;

    std::vector<int> GetSimpleOwnership() const;

    bool SetFixdHandicap(int handicap);

    bool SetFreeHandicap(std::vector<int> movelist);

    std::vector<LadderType> GetLadderPlane() const;

    std::vector<bool> GetOcupiedPlane(const int color) const;

private:
    bool ValidHandicap(int handicap);
};
