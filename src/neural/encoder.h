#pragma once

#include <vector>
#include <array>
#include "game/symmetry.h"
#include "game/game_state.h"
#include "neural/network_basic.h"

class Encoder {
public:
    static constexpr auto kPlaneChannels = kInputChannels;
    static constexpr auto kHistoryMove = 3;

    static Encoder& Get();

    InputData GetInputs(GameState &state, int symmetry = Symmetry::kIdentitySymmetry) const;

private:
    void FillColorStones(std::shared_ptr<const Board> board,
                         std::vector<float>::iterator it,
                         int color, int symmetry) const;
    void EncoderHistoryMove(GameState &state,
                            std::vector<float>::iterator it,
                            int symmetry) const;

};
