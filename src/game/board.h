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

    void ComputeRemovedOneLibertyOwnership(std::vector<int> &buffer) const; // experiment function

    std::vector<int> GetPassAliveOwnership() const; // experiment function

    bool SetFixdHandicap(int handicap);

    bool SetFreeHandicap(std::vector<int> movelist);

    std::vector<LadderType> GetLadderPlane() const;

    std::vector<bool> GetOcupiedPlane(const int color) const;

    std::vector<bool> GetPassAlive(const int color) const;

private:
    bool ValidHandicap(int handicap);

    bool IsPassAliveString(const int vertex,
                               bool allow_sucide,
                               const std::vector<bool> &vitals,
                               const std::vector<int> &features,
                               const std::vector<int> &regions_index,
                               const std::vector<int> &regions_next,
                               const std::vector<int> &strings_index,
                               const std::vector<int> &strings_next) const;

    bool IsPassDeadRegion(const int vertex,
                               const int color,
                               std::vector<int> &features,
                               const std::vector<int> &regions_next) const;

    std::vector<int> GatherVertices(std::vector<bool> &buf) const;

    std::vector<int> ClassifyGroups(const int target,
                                        std::vector<int> &features,
                                        std::vector<int> &regions_index,
                                        std::vector<int> &regions_next) const;

    void ComputationInnerRegions(const int vtx,
                                     const int color,
                                     const std::vector<int> &regions_next,
                                     std::vector<bool> &inner_regions) const;
};
