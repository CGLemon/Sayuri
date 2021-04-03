#ifndef GAME_BOARD_H_INCLUDE
#define GAME_BOARD_H_INCLUDE

#include <vector>

#include "game/simple_board.h"
#include "game/types.h"

class Board : public SimpleBoard {
public:
    int ComputeFinalScore(float komi) const;

    std::vector<int> GetSimpleOwnership() const;

    bool SetFixdHandicap(int handicap);

    bool SetFreeHandicap(std::vector<int> movelist);

private:
    bool ValidHandicap(int handicap);


};

#endif
