#pragma once

#include <vector>

#include "game/simple_board.h"
#include "game/types.h"

class Board : public SimpleBoard {
public:
    // Compute final score by Tromp Taylor rule.
    float ComputeSimpleFinalScore(float komi) const;

    int ComputeScoreOnBoard(int black_bonus) const;

    // Compute ownership by Tromp Taylor rule.
    void ComputeSimpleOwnership(std::vector<int> &result) const;

    void ComputePassAliveOwnership(std::vector<int> &result) const;

    bool SetFixdHandicap(int handicap);

    bool SetFreeHandicap(std::vector<int> movelist);

    std::vector<LadderType> GetLadderMap() const;

    // Compute all pass-alive string.
    // Mark all vital regions of pass-alive string if mark_vitals is true.
    // Mark all pass-dead regions if mark_pass_dead is true.
    void ComputePassAlive(std::vector<bool> &result,
                              const int color,
                              bool mark_vitals,
                              bool mark_pass_dead) const;

    // Compute all safe area which both players don't need to play move in.
    // Mark all seki points if mark_seki is true.
    void ComputeSafeArea(std::vector<bool> &result, bool mark_seki) const;

    void ComputeSekiPoints(std::vector<bool> &result) const;

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
