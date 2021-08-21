#pragma once

#include <vector>

#include "game/simple_board.h"
#include "game/types.h"

class Board : public SimpleBoard {
public:
    // Compute final score by Tromp Taylor
    float ComputeSimpleFinalScore(float komi) const;

    int ComputeScoreOnBoard(int black_bonus) const;

    // Compute ownership by Tromp Taylor
    void ComputeSimpleOwnership(std::vector<int> &buffer) const;

    std::vector<int> GetSimpleOwnership() const;

    bool SetFixdHandicap(int handicap);

    bool SetFreeHandicap(std::vector<int> movelist);

    std::vector<LadderType> GetLadderPlane() const;

    std::vector<bool> GetOcupiedPlane(const int color) const;

    std::vector<bool> GetPassAlive(const int color) const;

private:
    bool ValidHandicap(int handicap);

    std::vector<int> FindStringSurround(std::vector<int> &groups, int index) const;

    std::vector<int> GatherVertex(std::vector<bool> &buf) const;

    int ClassifyGroups(std::vector<int> &features, std::vector<int> &groups, int target) const;

};
