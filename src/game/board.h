#pragma once

#include <vector>

#include "game/simple_board.h"
#include "game/types.h"

class Board : public SimpleBoard {
public:
    int ComputeFinalScore(float komi) const;

    std::vector<int> GetSimpleOwnership() const;

    bool SetFixdHandicap(int handicap);

    bool SetFreeHandicap(std::vector<int> movelist);

    std::vector<LadderType> GetLadderMap() const;

private:
    bool ValidHandicap(int handicap);


};
