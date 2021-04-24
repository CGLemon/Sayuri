#pragma once

#include "game/game_state.h"

class EndGame {
public:
    enum FinalState : int {
        kBlackStone = 0,
        kBlackTerritory,

        kWhiteStone,
        kWhiteTerritory,

        kDame,
        kSeki
    };


    static EndGame &Get(GameState &state);

    EndGame(GameState &state) : root_state_(state) {}

    std::vector<int> GetFinalOwnership() const;

private:
    std::vector<int> GatherVertex(std::vector<bool> &buf) const;

    void AssignVertex(std::vector<int>, std::vector<bool> &buf) const;

    std::vector<int> GetLivedGroups(Board &board) const;

    int ComputeNumEye(std::vector<int> &eye_group) const;

    void FillMoves(Board &board, int color, std::vector<int> &vertex_group) const;

    void CompareAndRemoveDeadString(Board &board,
                                    std::vector<int> &lived_groups) const;

    GameState &root_state_;
};
