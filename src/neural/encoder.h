#pragma once

#include <vector>
#include <array>
#include "game/symmetry.h"
#include "game/game_state.h"
#include "neural/network_basic.h"

class Encoder {
public:
    static constexpr int kPlaneChannels = kInputChannels;
    static constexpr int kHistoryMoves = 8;
    static constexpr int kNumFeatures = 14;
    static_assert(kPlaneChannels == 3*kHistoryMoves + kNumFeatures, "");

    static Encoder& Get();

    // Get the Network input datas.
    InputData GetInputs(const GameState &state, int symmetry = Symmetry::kIdentitySymmetry) const;

    /*
     * Get the Network input planes.
     *
     * planes 1 -24 : last 8 history moves
     * plane     25 :
     * plane     26 :
     * planes 27-30 :
     * planes 31-34 :
     * plane     35 : komi/20
     * plane     36 : -komi/20
     * plane     37 : intersections/361
     * plane     38 : fill ones
     *
     */
    std::vector<float> GetPlanes(const GameState &state, int symmetry = Symmetry::kIdentitySymmetry) const;

    std::string GetPlanesString(const GameState &state, int symmetry = Symmetry::kIdentitySymmetry) const;

private:
    void SymmetryPlanes(const GameState &state, std::vector<float> &planes, int symmetry) const;

    void FillColorStones(const Board* board,
                         std::vector<float>::iterator black_it,
                         std::vector<float>::iterator white_it) const;

    void FillMove(const Board* board,
                  std::vector<float>::iterator move_it) const;

    void EncoderHistoryMove(const GameState &state,
                            int counter,
                            std::vector<float>::iterator it) const;

    void FillFeatures(const Board* board,
                      const int to_move,
                      std::vector<float>::iterator feat_it) const;

    void FillMisc(const Board* board,
                  const int color,
                  float komi,
                  std::vector<float>::iterator color_it) const;

    void EncoderFeatures(const GameState &state,
                         std::vector<float>::iterator it) const;

};
